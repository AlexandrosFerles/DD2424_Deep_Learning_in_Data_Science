### Imports

import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical as make_class_categorical

### Load Batch
def LoadBatch(filename):
    # borrowed from https://www.cs.toronto.edu/~kriz/cifar.html
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    dictionary = unpickle(filename)

    # borrowed from https://stackoverflow.com/questions/16977385/extract-the-nth-key-in-a-python-dictionary?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    def ix(dic, n):  # don't use dict as  a variable name
        try:
            return list(dic)[n]  # or sorted(dic)[n] if you want the keys to be sorted
        except IndexError:
            print('not enough keys')

    garbage = ix(dictionary, 1)
    y = dictionary[garbage]
    Y = np.transpose(make_class_categorical(y, 10))
    garbage = dictionary['data']
    X = np.transpose(garbage) / 255.0

    return X, Y, y

### Initialize weights
def initialize_weights(d, m, K, variance=0.01):

    W1= np.random.normal(0, variance, size=(m, d) )
    b1= np.random.normal(0, variance, size=(m, 1) )

    W2 = np.random.normal(0, variance, size=(K, m))
    b2 = np.random.normal(0, variance, size=(K, 1))

    return W1, b1, W2, b2

### ReLU
def ReLU(x):

    return max(0,x)

### Softmax
def softmax(X, theta=1.0, axis=None):

    # Softmax over numpy rows and columns, taking care for overflow cases
    # Many thanks to https://nolanbconaway.github.io/blog/2017/softmax-numpy
    # Usage: Softmax over rows-> axis =0, softmax over columns ->axis =1

    """
    Compute the softmax of each element along an axis of X.
    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.
    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

### Evaluate Classifier
def EvaluateClassifier(X, W1, b1, W2, b2):

    s1= np.dot(W1,X)+ b1
    h=ReLU(s1)
    s=np.dot(W2,h)+b2
    p= softmax(s, axis=1)

    return p, h, s1, s

### Predict classes
def predictClasses(p):

    return np.argmax(p, axis=0)

### Compute Gradients
def ComputeGradients(X, Y, W1, b1, W2, b2, regularization_term):

    grad_W1= np.zeros(W1.shape)
    grad_b1= np.zeros(b1.shape)
    grad_W2= np.zeros(W2.shape)
    grad_b2= np.zeros(b2.shape)

    p, h, s1, s = EvaluateClassifier(X, W1, b1, W2, b2)
    for datum_index in range(X.shape[1]):

        g= (Y[:,datum_index] - p[:,datum_index]).T

        #  Add gradient of l w.r.t. b2 & W2 computed at (x, y)
        grad_b2+= g
        grad_W2+= np.dot(g.T, h.T)

        # Back-propagate gradient through 2nd fully connected layer

        g= np.dot(g, W2)
        g= np.dot(g, np.diag(s1.clip(0)) )

        # Add gradient of l w.r.t.first layer parameters computed at(x, y)
        grad_b1+= g

        grad_W1+= np.dot(g.T, X[:,datum_index]).T

    grad_W1/=X.shape[1]
    grad_W2/=X.shape[1]
    grad_b1/=X.shape[1]
    grad_b2/=X.shape[1]

    grad_W1+= 2*regularization_term*W1
    grad_W2+= 2*regularization_term*W2

    return grad_W1, grad_b1, grad_W2, grad_b2



