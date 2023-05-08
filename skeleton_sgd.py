#################################
# name: Aviv Avraham
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import random

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels
# train data is ndarray(6000, 784=28X28 pixels),
# train labels is from {-1,1} where -1 stands for "0" and 1 stands for "8".
# validation data is ndarray(5774, 784=28X28 pixels),validation labels...
# test data is ndarray(1954, 784=28X28 pixels),validation labels...


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
"""
ℓ(w, x, y) = C(max{0, 1 −y⟨w, x⟩}) + 0.5∥w∥2)
sgd stands for stochastic gradient descent algorithm which is as follow:
initialize w0 = 0
on each iteration t 0,1,... : 
we sample i uniformly; and if ((yi) (wt)· (xi)) < 1, we update:
wt+1 = (1 − (ηt))wt + (ηt) C (yi) (xi)
and wt+1 = (1 − ηt)(wt) otherwise
where ηt = η0/t, and η0 is a constant.
"""


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w = np.zeros(len(data[0]))
    for t in range(T):
        i = random.randint(0, len(data) - 1)
        if labels[i] * (numpy.dot(data[i], w)) < 1:
            w = (1 - eta_0) * w + eta_0 * C * labels[i] * data[i]
        else:
            w *= (1 - eta_0)
        eta_0 /= (t + 1)
    return w

def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    # TODO: Implement me
    pass

#################################


# cross validation to find the best eta.
def q1a(T=1000, C=1, runs=10):
    accuracy = [0] * 10
    eta_0 = 10 ^ -5
    for run in range(runs):
        eta = eta_0 * (10 ^ run)
        w = SGD_hinge(train_data, train_labels, C, eta, T)
        loss = loss_calculator(w)
        return loss


"""
ℓ(w, x, y) = C(max{0, 1 −y⟨w, x⟩}) + 0.5∥w∥2)
"""
def loss_calculator(w):
    pass
    # TODO: implement me

#################################
