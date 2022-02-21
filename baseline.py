import numpy as np
import deepdish as dd

from os import listdir
from os.path import isfile, join
import h5py
import deepdish as dd
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ---------- Load Preprocessed ---------#
# ==== RUN 02-filter_gene.py which saves 'graph/gene.h5' -----#
data = dd.io.load("graph/gene.h5")
gene_names = data["gene_name"]
embed_id = data["embed_id"]
gene_id = data["gene_id"]

# ------ Read column names from file
path = "/data/xiaoxiaol/data/frank/embedded/raw"
samples = [f for f in listdir(path) if isfile(join(path, f))]

gene_ids = []
for k in list(embed_id.keys()):
    if k in gene_names:
        gene_ids.append(gene_id[k])
gene_ids = np.array(gene_ids)

X = np.zeros((len(samples), len(gene_names), 64, 2))
y = np.zeros(len(samples))
ids = []
for i, sample in enumerate(samples):
    data = h5py.File(join(path, sample))
    fea1 = data["data"]["promoter"][()]  # np array, dim (19201, 64), noncoding feature
    fea2 = data["data"]["protein"][()]  # np array, dim (19201, 64), coding feature
    X[i, :, :, 0] = fea1[gene_ids, :]  # np array, dim (18749, 64)
    X[i, :, :, 1] = fea2[gene_ids, :]  # np array, dim (18749, 64)
    id = data["label"]["sample_id"][()]  # bytes, patient id
    ids.append(id)
    if data["label"]["sample_meta"]["tumor"][()] == b"GBM":
        y[i] = 1  # GBM: glioblastoma and patients die within a few months.

# === split data for baseline ===#
start_time = time.time()
score = []
train_id, test_id, y_train, y_test = train_test_split(
    np.arange(len(y)), y, test_size=0.3, random_state=123
)
# ===== Run some baselines =====#
ids = np.array(ids)
X1 = X[:, ids, :, :]
X1 = X.reshape(901, -1)
X1_train = X1[train_id]
X1_test = X1[test_id]

from sklearn.decomposition import PCA

pca = PCA(n_components=128)
pca.fit(X1_train)
X2_train = pca.transform(X1_train)
X2_test = pca.transform(X1_test)

lr = LogisticRegression(random_state=123)
lr.fit(X2_train, y_train)
print("f% accuracy of LR" % (lr.score(X2_test, y_test)))
svm = SVC(gamma=2, C=1)
svm.fit(X2_train, y_train)
print("f% accuracy of SVM" % (svm.score(X2_test, y_test)))
rf = RandomForestClassifier(max_depth=3, n_estimators=10, random_state=123)
rf.fit(X2_train, y_train)
print("f% accuracy of RF" % (rf.score(X2_test, y_test)))
mlp = MLPClassifier(random_state=1, max_iter=300)
mlp.fit(X2_train, y_train)
print("f% accuracy of MLP" % (mlp.score(X2_test, y_test)))

