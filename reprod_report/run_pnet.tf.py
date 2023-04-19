try:
    import silence_tensorflow

    silence_tensorflow.silence_tensorflow()
except ImportError:
    print("failed to silence TF")
import os
import importlib
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from cancernet.pnet.model.nn import Model

from cancernet.pnet.data.data_access import Data
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import argparse

parser = argparse.ArgumentParser(description="run P-Net for prostate cancer discovery")
parser.add_argument(
    "--n-hidden", required=True, type=int, help="number of hidden layers"
)
parser.add_argument(
    "--n-repeats",
    required=False,
    type=int,
    default=1,
    help="number of repeated measures",
)
parser.add_argument(
    "--disable-earlystop",
    action="store_true",
    help="disable early stopping based on val loss",
)
args = parser.parse_args()

params_file = ".onsplit_average_reg_10_tanh_large_testing"
params_file_full = "cancernet.pnet.train.params.P1000.pnet" + params_file
params = importlib.import_module(params_file_full)

# manually adjusted P-net model params
n_hidden_layers = args.n_hidden
assert 2 <= n_hidden_layers < 8, "unsupported hidden layers number"
params.models[0]["params"]["model_params"]["n_hidden_layers"] = n_hidden_layers
params.models[0]["params"]["model_params"]["w_reg"] = [0.001] * (n_hidden_layers + 1)
params.models[0]["params"]["model_params"]["w_reg_outcomes"] = [0.001] * (
    n_hidden_layers + 1
)
params.models[0]["params"]["model_params"]["dropout"] = [0.5] + [0.1] * (
    n_hidden_layers
)
params.models[0]["params"]["model_params"]["loss_weights"] = (
    [2, 7, 20, 54, 148, 400][: (n_hidden_layers + 1)]
    if n_hidden_layers <= 5
    else [2] * (n_hidden_layers - 5) + [2, 7, 20, 54, 148, 400]
)
# new input genes
params.models[0]["params"]["model_params"]["data_params"]["params"][
    "selected_genes"
] += "matched_pyg.csv"
# change molecular data type order to match PyG dataset
params.models[0]["params"]["model_params"]["data_params"]["params"]["data_type"] = [
    "mut_important",
    "cnv_amp",
    "cnv_del",
]

data = Data(**params.data[0])

(
    x_train,
    x_validate_,
    x_test_,
    y_train,
    y_validate_,
    y_test_,
    info_train,
    info_validate_,
    info_test_,
    cols,
) = data.get_train_validate_test()
results = []

for _ in tqdm(range(args.n_repeats)):
    # model inherits a sklearn BaseEstimator
    model = Model(**params.models[0]["params"])
    ret = model.build_fn(**model.model_params)
    pnet_mod, feature_names, reactome_map = ret

    verbose = 2
    if args.disable_earlystop:
        callback_list = []
    else:
        callback_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, verbose=verbose
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "pnet_mod.h5", save_best_only=True, verbose=verbose
            ),
        ]
    pnet_mod.fit(
        x_train,
        [y_train] * (n_hidden_layers + 1),
        validation_data=(x_validate_, [y_validate_] * (n_hidden_layers + 1)),
        callbacks=callback_list,
        epochs=300,
        batch_size=10,
        verbose=verbose,
    )

    # export weights and bias
    if not args.disable_earlystop:
        pnet_mod.load_weights("pnet_mod.h5")
    y_hat = pnet_mod.predict(x_test_)
    print([y.shape for y in y_hat])
    ys = y_test_.flatten()
    outs = y_hat[-1]

    fpr_valid, tpr_valid, _ = roc_curve(ys, outs)
    test_auc = auc(fpr_valid, tpr_valid)
    result = {}
    result["auc"] = test_auc
    result["accuracy"] = accuracy_score(ys, outs > 0.5)
    result["aupr"] = average_precision_score(ys, outs)
    result["f1"] = f1_score(ys, outs > 0.5)
    result["precision"] = precision_score(ys, outs > 0.5)
    result["recall"] = recall_score(ys, outs > 0.5)

    for k, v in result.items():
        print(k, v)

    # result['y_hat'] = y_hat
    # result['ys'] = ys
    # result['y_info'] = info_test_
    # result['outs'] = outs.flatten()
    result["pred"] = pd.DataFrame(
        np.vstack([np.array(y_hat).squeeze(), ys]).T,
        index=info_test_,
        columns=["pred_%i" % i for i in range(len(y_hat))] + ["y"],
    )
    results.append(result)

    with open(f"pnet_results.h{n_hidden_layers}.pkl", "wb") as f:
        pickle.dump(results, f)

    tf.keras.backend.clear_session()
