#################################
# name: Aviv Avraham
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import random
import matplotlib.pyplot as plt
import scipy
from scipy import special

"""
this function extracts data from the MNIST data base, the data is images of the
 8 and 0 digits, 28X28 pixels for a data point represents as nd array of size (1,784)
 as well as label the 8 digit as 1 and the 0 digit as -1.
"""


def extracting_data():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels_ = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels_ = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels_ = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data_ = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data_ = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data_ = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data_, train_labels_, validation_data_, validation_labels_, test_data_, test_labels_


# train data is ndarray(6000, 784=28X28 pixels),
# train labels is from {-1,1} where -1 stands for "0" and 1 stands for "8".
# validation data is ndarray(5774, 784=28X28 pixels),validation labels...
# test data is ndarray(1954, 784=28X28 pixels),validation labels...


# extracting data
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = extracting_data()


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    ℓ(w, x, y) = C(max{0, 1 −y⟨w, x⟩}) + 0.5∥w∥2)
    sgd stands for stochastic gradient descent algorithm which is as follow:
    initialize w0 = 0
    on each iteration t 0,1,... :
    we sample i uniformly; and if ((yi) (wt)· (xi)) < 1, we update:
    wt+1 = (1 − (ηt))wt + (ηt) C (yi) (xi)
    and wt+1 = (1 − ηt)(wt) otherwise
    where ηt = η0/t, and η0 is a constant.
    """
    eta = eta_0
    w = np.zeros(len(data[0]))
    for t in range(T):
        i = random.randint(0, len(data) - 1)
        if labels[i] * (numpy.dot(data[i], w)) < 1:
            w = (1 - eta) * w + eta * C * labels[i] * data[i]
        else:
            w *= (1 - eta)
        eta = eta_0 / (t + 1)
    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    ℓlog (w, x, y) = log(1 + e^(-y<w,x>))
    sgd stands for stochastic gradient descent algorithm which is as follow:
    initialize w0 = 0
    on each iteration t 0,1,... :
    we sample i uniformly; and if ((yi) (wt)· (xi)) < 1, we update:
    wt+1 = wt + (ηt) gradient(fi(wt))
    gradient(fi(w))= -y_i x_i  (e^(-y_i w∙x_i )/(1+ e^(-y_i w∙x_i )
    where ηt = η0/t, and η0 is a constant.
    """
    eta = eta_0
    w = np.zeros(len(data[0]))
    # q2c:
    # norms = np.zeros(T)
    for t in range(T):
        i = random.randint(0, len(data) - 1)
        w -= eta * -labels[i] * data[i] * (1 - scipy.special.expit(labels[i] * numpy.dot(w, data[i])))
        eta = eta_0 / (t + 1)
        # q2c:
        # norms[t] = np.linalg.norm(w)
    # q2c:
    # return w, norms
    return w


#################################
# cross validation to find the best eta.
def q1_2_a_b(T=1000, C=1, runs=10, scale=11, eta=1):
    """
    Train the classifier on the training set. Use cross-validation on the
    validation set to find the best parameters.
    """
    accuracy = [0.0] * scale
    for i in range(-5, -5 + scale):
        sum_ = 0
        eta = pow(10, i)
        for run in range(runs):
            w = SGD_log(train_data, train_labels, eta, T)
            sum_ += accurate_calculator(w, validation_data, validation_labels)
        accuracy[i + 5] = sum_ / runs
    return accuracy


def q1c_q2b():
    """
    Show the resulting w as an image
    you can adjust for Hinge lost by changing function call to w
    from SGD_log to hinge, and you can pass C value if wanted.
    """
    w = SGD_log(train_data, train_labels, pow(10, -5), 20000)
    plt.imshow(w.reshape((28, 28)), interpolation='nearest')
    plt.colorbar()
    plt.title("weight vector w as an image for log loss")
    plt.show()


def q1d_q2b():
    """
    prints the accuracy of w on the test set
    """
    eta = pow(10, -5)
    w = SGD_log(train_data, train_labels, eta, 20000)
    return accurate_calculator(w, test_data, test_labels)


def q2c():
    """
    calculates w weight vector norm along its calculation on the SGD algorithm.
    """
    w, norms = SGD_log(train_data, train_labels, pow(10, -5), 20000)
    plot(np.array([i for i in range(20000)]), norms, "t", "w norm", "w norm", "q2c: visualizing w norm for SGD with "
                                                                              "log loss")


def plot(x, y, x_label, y_label, label, title):
    """
    plots the data
    """
    plt.plot(x, y, color='r', label=label)
    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.xscale('log')
    # plt.ylim((0.8, 1))
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    # To load the display window
    plt.show()


def accurate_calculator(w, data, labels):
    """
    calculates the ratio of success for the linear classifier
    """
    sum_ = 0
    for i in range(len(data)):
        if ((labels[i] == 1 and numpy.dot(data[i], w) >= 0) or
                (labels[i] == -1 and numpy.dot(data[i], w) < 0)):
            sum_ += 1
    return sum_ / len(data)


if __name__ == '__main__':
    # q1a:
    # plot(np.array([pow(10, i) for i in range(-5, 4)]), q1a(), "eta", "ratio of success", "accuracy", "q1a")
    # q1b:
    # plot(np.array([pow(10, i) for i in range(-5, 6)]), q1ab(), "C", "ratio of success", "accuracy", "q1b")
    # q1c:
    # q1c()
    # q1d:
    # print(q1d())
    # q2a:
    # plot(np.array([pow(10, i) for i in range(-5, 6)]), q1_2_a_b(), "eta", "ratio of success", "accuracy", "q2a")
    # q2b:
    # q1c_q2b()
    # print(q1d_q2b())
    # q2c:
    # q2c()
    pass

#################################
