import numpy as np
import matplotlib.image as img
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.io import loadmat
from statistics import mode



### Lab 1 : Source coding
## Memoryless entropy calculation

def rand_vector(set, l):
    """Generates a vector of length l, taking random values in set."""
    return [set[np.random.randint(0, len(set))] for _ in range(l)]

def estimate_P(X, setX=[]):
    """Returns the proportion of each element of setX in vector X."""
    if not setX : setX = list(set(X))
    P = np.array([0 for _ in setX])
    for x in X :
        P[setX.index(x)] += 1
    return list(P/len(X))

def entropy(X, setX=[]):
    """Returns the entropy of vector X taking values in setX."""
    h = 0
    if not setX : setX = list(set(X))
    P = estimate_P(X, setX)
    for p in P :
        h += 0 if p == 0 else p * np.log2(p)
    return -h


## Entropy as a function of P(X=0)

def draw_h(steps, length, setX):
    """Displays the evolution of the entropy of a length-long vector (taking values in setX), as a function of the proportion of 0 in said vector. steps is the number of points to compute."""
    x_axis = [k/steps for k in range(steps+1)]
    y_axis = []
    cset = setX.copy()
    cset.remove(0)
    for k in range(steps+1):
        tx = int(length * x_axis[k])
        X = [0 for i in range(tx)] + rand_vector(cset, length-tx)
        y_axis.append(entropy(X, setX))
    plt.plot(x_axis, y_axis)
    plt.xlabel("Proportion of 0 in X")
    plt.ylabel("Entropy of X")
    plt.show()


## Image processing using first order stationary Markov chain

def estimate_M(I):
    """Treats I (a 2D B&W image) as a first order Markov chain. Estimates four probabilities pXY of reading X after Y in I, where X and Y are either 0 or 1 (black or white pixel)."""
    p00, p01, p10, p11 = 0, 0, 0, 0
    prev, curr = I[0][0], I[0][1]
    i, j = 0, 2
    l, c = np.shape(I)
    while i < l :
        if prev == 0 :
            if curr == 0 : p00 += 1
            else :         p01 += 1
        else :
            if curr == 0 : p10 += 1
            else :         p11 += 1
        prev = curr
        curr = I[i][j]
        j += 1
        if j == c :
            j = 2
            i += 1
            if i < l :
                prev, curr = I[i][0], I[i][1]
    p0x, p1x = p00+p01, p10+p11
    p00 /= p0x
    p01 /= p0x
    p10 /= p1x
    p11 /= p1x
    return p00, p10, p01, p11

def run_length_encoder(I):
    """Returns a 2D array encoding I (a 2D array of labels). The first sublist contains the dimensions of I. The next sublists are [LENGTH, LABEL], representing a series of LENGTH consecutive LABEL."""
    curr = I[0][0]
    l, c = np.shape(I)
    i, j = 0, 1
    code = [[l, c]]
    while i < l :
        rl = 1
        prev = curr
        curr = I[i][j]
        while prev == curr and i < l :
            rl += 1
            j += 1
            if j == c :
                j = 0
                i += 1
            if i < l :
                prev = curr
                curr = I[i][j]
        code.append([rl, prev])
    return code

def code_length(RLE):
    """Returns the bitwise length of RLE, a list of couples."""
    l = 0
    for e in RLE :
        l += int(e[0]).bit_length() + int(e[1]).bit_length()
    return l



### Lab 2 : Lossy image compression

def k_means(ImgFlat, n_clusters):
    """Runs the KMeans algorithm on ImgFlat, a 2D array, with n_clusters clusters.
    Returns the labels and clusters computed by KMeans."""
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(ImgFlat)
    return kmeans.labels_, kmeans.cluster_centers_

def images_from_kmeans(labels, clusters, n_cols):
    """Creates two 2D arrays with n_cols columns, from the result of KMeans. ImgLabels contains the labels, ImgClustersFlat contains the corresponding clusters."""
    i, j = 0, 0
    ImgLabels = []
    ImgClusters = []
    for label in labels :
        if j == 0 :
            ImgLabels.append([])
            ImgClusters.append([])
        ImgLabels[-1].append(label)
        ImgClusters[-1].append(clusters[label].tolist())
        j += 1
        if j == n_cols :
            j = 0
            i += 1
    ImgClustersFlat = []
    for line in ImgClusters :
        for pixel in line :
            ImgClustersFlat.append(pixel)
    return ImgLabels, ImgClustersFlat



### Lab 3 : Mutual information and NMI calculation

def estimate_joint_P(X, Y, setX=[], setY=[]):
    """Estimates the joint probability matrix of vectors X and Y, respectively taking values in setX and setY."""
    lX = len(X)
    if not setX : setX = list(set(X))
    if not setY : setY = list(set(Y))
    PXY = np.zeros((len(setX), len(setY)))
    for k in range(lX):
        PXY[setX.index(X[k])][setY.index(Y[k])] += 1
    return PXY/lX

def mutual_info_quantity(X, Y, setX=[], setY=[]):
    """Returns the mutual information quantity of vectors X and Y, respectively taking values in setX and setY."""
    if not setX : setX = list(set(X))
    if not setY : setY = list(set(Y))
    l1, l2 = len(setX), len(setY)
    if len(X) != len(Y) :
        raise ValueError("The two vectors must have the same dimensions.")
    PXY = estimate_joint_P(X, Y, setX, setY)
    PX = [sum(PXY[i]) for i in range(l1)]
    PY = [sum([PXY[i][j] for i in range(l1)]) for j in range(l2)]
    info = 0
    for i in range(l1) :
        for j in range(l2) :
            xy = PXY[i][j]
            info += 0 if xy == 0 else xy * np.log2(xy / PX[i] / PY[j])
    return info

def NMI(X, Y, setX=[], setY=[]):
    """Returns the normalized mutual information of vec tors X and Y, respectively taking values in setX and setY."""
    return 2*mutual_info_quantity(X, Y, setX, setY) / (entropy(X, setX) + entropy(Y, setY))

def cross_NMI(Labels, Data):
    """Takes a list of equally sized vectors Data, labelled by Labels, and computes the NMI between each pair of vectors."""
    l = len(Labels)
    nmis = [[1.0 if i == j else 0 for i in range(l)] for j in range(l)]
    for i in range(l):
        for j in range(i):
            nmi = NMI(Data[i], Data[j])
            nmis[i][j] = nmi
            nmis[j][i] = nmi
    return nmis



### Lab 4 : IT-based feature selection

def mRMR_feature_selection(Img, Y, n):
    """Uses the mRMR algorithm to select the n most representative bands of Img, according to the target class vector Y.
    Note that it calls the mutual_info_score function of sklearn (which uses C) instead of mutual_information_quantity, because Python is far too slow to compute everything this algorithm needs."""
    Img_tmp = deepcopy(Img)
    l = len(Img)
    Features_index, Features = [], []
    MI_xz = [[] for _ in range(l)]
    MI_xy = [mutual_info_score(Img[k], Y) for k in range(l)]
    for k in range(1, n+1):
        Z = []
        for i in range(l-k+1) :
            S = 0 if k == 1 else 1/(k-1) * sum(MI_xz[i])
            Z.append(MI_xy[i] - S)
        zk = Z.index(max(Z))
        Features_index.append(Img.index(Img_tmp[zk]))
        Features.append(Img_tmp[zk])
        MI_xz.pop(zk)
        MI_xy.pop(zk)
        Img_tmp.pop(zk)
        if k < n :
            for j in range(l-k) :
                MI_xz[j].append(mutual_info_score(Img[j], Features[-1]))
    return Features_index, Features

def distance(X, Y):
    """Computes a distance between X and Y, from the relative distances between each attribute."""
    dist = 0
    for i in range(len(X)):
        dist += abs(X[i]-Y[i])
    return dist/len(X)


class kNN_Classifier(object):
    """Uses the result of feature selection to classify data according to the nearest neighbors method."""
    samples_in = []
    samples_out = []
    k = 1

    def __init__(self, k, samples_in, samples_out):
        """k is the number of neighbors considered for classification.
        samples_in should be a matrix of samples of the features selected by mRMR_feature_selection.
        samples_out should be the corresponding target class vector."""
        self.samples_in = samples_in
        self.samples_out = samples_out
        self.k = k

    def k_closest(self, x):
        """Returns the k nearest neighbors of x in the training data, taking every feature into account."""
        Dist = [distance(x, y) for y in self.samples_in]
        Closest = []
        for _ in range(k):
            i = Dist.index(min(Dist))
            Closest.append(i)
            Dist.pop(i)
        return Closest

    def predict(self, x):
        """Predicts the class of x with a majority vote using its k nearest neighbors."""
        neighbors = self.k_closest(x)
        outs = [self.samples_out[neighbor] for neighbor in neighbors]
        return mode(outs)

    def predict_all(self, X):
        """Predicts classes for all elements of X."""
        return [self.predict(x) for x in X]

    def overall_accuracy(self, X, Y):
        """Returns the overall accuracy (trace of the confusion matrix) of the classifier, for testing data X, with expected class vector Y."""
        correct = 0
        Pred = self.predict_all(X)
        for i in range(len(Pred)):
            if Pred[i] == Y[i] :
                correct += 1
        return correct/len(Pred)
