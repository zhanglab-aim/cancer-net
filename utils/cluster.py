import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import k_means


def cluster_dict(labels):
    community = dict()
    for c in set(labels):
        community[c] = np.where(labels==c)[0]
    return community

def kmeans(cmat):
    N = cmat.shape[0] // 50
    clustering1 = k_means(cmat, n_clusters=N)
    label1 = clustering1[1]
    community1 = cluster_dict(label1)

    # removing the across community edges
    cmat3 = np.zeros(cmat.shape)
    for l in label1:
        cmat3[np.ix_(community1[l], community1[l])] = np.copy(cmat[np.ix_(community1[l], community1[l])])

    # thresholding on cmat3
    res = np.copy(cmat3)
    res[res < 0.05] = 0

    return res
