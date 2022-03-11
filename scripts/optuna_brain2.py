import torch, torch_geometric.transforms as T, torch.nn.functional as F
import matplotlib.pyplot as plt, numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import wandb
import optuna

## import cancer-net files
import sys
sys.path.append('../')
import TCGAData
from arch.net import *

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0.005):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Objective(object):
    def __init__(self,arch,root,rng,batch,epochs,device):
        self.arch=arch
        self.root=root
        self.rng=rng
        self.batch=batch
        self.epochs=epochs
        self.device=device
        
        ## hardcoding this false for now
        self.parall=False
        
        ## Should be able to construct the dataset before __call__
        ## as this won't change for different trials
        label_mapping = ["LGG", "GBM"]
        pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
        self.dataset = TCGAData.TCGADataset(
            root=self.root,
            files=self.root+"/samples.txt",
            label_mapping=label_mapping,
            gene_graph="brain.geneSymbol.gz",
            transform=pre_transform,
            suffix="sparse",
        )

        rng = np.random.default_rng(self.rng)
        rnd_perm = rng.permutation(len(self.dataset))
        self.train_indices = list(rnd_perm[: 3 * len(self.dataset) // 4])
        self.test_indices = list(rnd_perm[3 * len(self.dataset) // 4 :])
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch,
            sampler=SubsetRandomSampler(self.train_indices),
            drop_last=True,
        )
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch,
            sampler=SubsetRandomSampler(self.test_indices),
            drop_last=True,
        )

        assert len(self.train_indices) + len(self.test_indices) == len(
            self.dataset
        ), "Train test split with overlap or unused samples!"
    
    def train(self, epoch, report=True):
        self.model.train()
        total_loss = 0
        correct = 0
        num_samps = 0
        for data in self.train_loader:
            if not self.parall:
                data = data.to(device)
            self.optimizer.zero_grad()

            output = self.model(data)
            output = output.squeeze()

            if self.parall:
                y = torch.cat([d.y for d in data]).to(output.device)
            else:
                y = data.y

            if len(output.shape) == 1:
                output = output.unsqueeze(0)
            loss = self.criterion(output, y)

            pred = output.max(1)[1]
            correct += pred.eq(y).sum().item()
            total_loss += loss
            loss.backward()
            self.optimizer.step()
            num_samps += len(y)
        if report:
            print(
                "Epoch: {:02d}, Loss: {:.3g}, Train Acc: {:.4f}".format(
                    epoch, total_loss / num_samps, correct / num_samps
                )
            )

        return total_loss / num_samps, correct / num_samps
    
    def test(self):
        self.model.eval()
        correct = 0

        total_loss = 0
        num_samps = 0
        for data in self.test_loader:
            if not self.parall:
                data = data.to(device)
            output = self.model(data)
            output = output.squeeze()

            pred = output.max(1)[1]
            if self.parall:
                y = torch.cat([d.y for d in data]).to(output.device)
            else:
                y = data.y
            loss = self.criterion(output, y)
            total_loss += loss.item()

            correct += pred.eq(y).sum().item()
            num_samps += len(y)
        return total_loss / num_samps, correct / num_samps
    
    def __call__(self,trial):
        print("Suggesting trial")
        # get the value of the hyperparameters
        lr        = trial.suggest_float("lr", 5e-5, 5e-2, log=True)
        self.alpha     = trial.suggest_float("alpha", 0.2, 0.8)
        self.dropout   = trial.suggest_float("dropout", 0.1, 0.4)
        print("Suggested trial")
        ## Store hyperparams in a config for wandb
        config = {"learning rate": lr,
                 "epochs": self.epochs,
                 "batch size": self.batch,
                 "arch": self.arch,
                 "alpha": self.alpha,
                 "dropout": self.dropout}

        print("\nTrial number: {}".format(trial.number))
        print("lr: {}".format(lr))
        print("alpha: {}".format(self.alpha))
        print("dropout: {}".format(self.dropout))

        wandb.login()
        wandb.init(project="brain-GCN2", entity="chris-pedersen",config=config)
        self.model = GCN2Net(
            hidden_channels=2048,
            num_layers=4,
            alpha=self.alpha,
            theta=1.0,
            shared_weights=False,
            dropout=self.dropout).to(device)
        wandb.watch(self.model, log_freq=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.3, patience=7)
        early_stopping=EarlyStopping()
        self.criterion = F.nll_loss
        train_losses = []
        train_acces = []
        test_acces = []
        test_losses = []
        for epoch in range(1, self.epochs):
            report = (epoch) % 10 == 0
            train_loss, train_acc = self.train(epoch, report=report)
            test_loss, test_acc = self.test()
            train_losses.append(train_loss.cpu().detach().numpy())
            test_losses.append(test_loss)
            train_acces.append(train_acc)
            test_acces.append(test_acc)
            wandb.log({"train loss": train_loss,
                       "test loss": test_loss,
                       "train accuracy": train_acc,
                       "test accuracy": test_acc,
                       "learning rate": self.optimizer.param_groups[0]["lr"]})
            if report:
                print("Test Loss: {:.3g}, Acc: {:.4f}".format(test_loss, test_acc))
            if epoch>50:
                early_stopping(test_loss)
                if early_stopping.early_stop:
                    wandb.finish()
                    #trial.study.stop()
                    break
        wandb.finish()

arch = "GCN2"
batch = 10
rng = 2022
parall = False
epochs=200

if torch.cuda.is_available():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

root = "/mnt/home/sgolkar/projects/cancer-net/data/brain"

## Optuna params
study_name = "optuna/brain-GCN2"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
n_trials=30

# train networks with bayesian optimization
objective = Objective(arch,root,rng,batch,epochs,device)
sampler = optuna.samplers.TPESampler(n_startup_trials=30)
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage_name,
                            load_if_exists=True)
study.optimize(objective, n_trials, gc_after_trial=False)
