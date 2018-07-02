import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical as make_class_categorical

def LoadBatch(filename):
    """
    Loads batch based on the given filename and produces the X, Y, and y arrays

    :param filename: Path of the file
    :return: X, Y and y arrays
    """
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

def initialize_weights(d, m, K, std=0.001):
    """
    Initializes the weight and bias arrays for the 2 layers of the network

    :param d: Dimensionality of the input data
    :param m: Number of nodes in the first layer
    :param K: Number of different classes (K=10 for the CIFAR-10 dataset)
    :param variance (optional): The variance of the normal distribution that will be used for the initialization of the weights

    :return: Weights and bias arrays for the first and second layer of the neural network
    """

    W1= np.random.normal(0, std, size=(m, d) )
    b1= np.zeros(shape=(m,1))

    W2 = np.random.normal(0, std, size=(K, m))
    b2 = np.zeros(shape=(K,1))

    return W1, b1, W2, b2

def ReLU(x):
    """
    Rectified Linear Unit function

    :param x: Input to the function

    :return: Output of ReLU(x)
    """

    return max(0,x)


def softmax(X, theta=1.0, axis=None):
    """
    Softmax over numpy rows and columns, taking care for overflow cases
    Many thanks to https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Usage: Softmax over rows-> axis =0, softmax over columns ->axis =1

    :param X: ND-Array. Probably should be floats.
    :param theta: float parameter, used as a multiplier prior to exponentiation. Default = 1.0
    :param axis (optional): axis to compute values along. Default is the first non-singleton axis.

    :return: An array the same size as X. The result will sum to 1 along the specified axis
    """

    # make X at least 2d
    y= np.atleast_2d(X)

    # find axis
    if axis is None:
        axis= next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y= y * float(theta)

    # subtract the max for numerical stability
    y= y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y= np.exp(y)

    # take the sum along the specified axis
    ax_sum= np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p= y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def EvaluateClassifier(X, W1, b1, W2, b2):
    """
    Computes the Softmax output of the 2 layer network, based on input data X and trained weight and bias arrays

    :param X: Input data
    :param W1: Weight array of the first layer
    :param b1: Bias vector of the first layer
    :param W2: Weight array of the second layer
    :param b2: Bias vector of the second layer

    :return: Softmax output of the trained network
    """

    s1= np.dot(W1,X) + b1
    h= ReLU(s1)
    s= np.dot(W2,h) + b2
    p= softmax(s, axis=1)

    return p, h, s1

def predictClasses(p):
    """
    Predicts classes based on the softmax output of the network

    :param p: Softmax output of the network
    :return: Predicted classes
    """

    return np.argmax(p, axis=0)

def ComputeAccuracy(X, y, W1, b1, W2, b2):
    """
    Computes the accuracy of the feed-forward 2-layer network

    :param X: Input data
    :param y: Labels of the ground truth
    :param W1: Weight matrix of the first layer
    :param b1: Bias vector of the first layer
    :param W2: Weight matrix of the second layer
    :param b2: Bias vector of the second layer

    :return: Accuracy metric of the neural network.
    """
    p, _, _ = EvaluateClassifier(X=X, W1=W1, b1=b1, W2=W2, b2=b2)
    predictions = predictClasses(p)

    accuracy= np.sum(np.where(predictions-y==0, 1, 0))

    return accuracy

def ComputeCost(X, Y, W1, b1, W2, b2, regularization_term):

    p, _, _ = EvaluateClassifier(X=X, W1=W1, b1=b1, W2=W2, b2=b2)

    cross_entropy_loss = np.sum(-np.log(np.dot(Y.T, p)), axis=1) / float(X.shape[0])

    weight_sum = np.power(W1,2).sum() + np.power(W2,2).sum()

    return cross_entropy_loss + regularization_term*weight_sum

def ComputeGradsNum(X, Y, W1, b1, W2, b2, regularization_term, h=1e-5):

    """
    Computes gradient descent updates on a batch of data with numerical computations.
    Contributed by Josephine Sullivan for educational purposes for the DD2424 Deep Learning in Data Science course.

    :param X: Input data
    :param Y: One-hot representation of the true labels of input data X
    :param W1: Weight matrix of the first layer
    :param b1: Bias vector of the first layer
    :param W2: Weight matrix of the second layer
    :param b2: Bias vector of the second layer
    :param regularization_term: Contribution of the regularization in the weight updates

    :return: Weight and bias updates of the first and second layer of our network computed with numerical computations
    """

    grad_W1= np.zeros(W1.shape)
    grad_b1= np.zeros(b1.shape)
    grad_W2= np.zeros(W2.shape)
    grad_b2= np.zeros(b2.shape)
    
    c = ComputeCost(X=X, Y=Y, W1=W1, b1=b1, W2=W2, b2=b2, regularization_term=regularization_term)
    
    for i in range(b1.shape[0]):
        b1_try = b1[i,0]
        b1_try[i,0] += h
        c2 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1_try, W2=W2, b2=b2, regularization_term=regularization_term)
        grad_b1[i,0] = (c2-c) / h

    for i in range(b2.shape[0]):
        b2_try = b2[i,0]
        b2_try[i,0] += h
        c2 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1, W2=W2, b2=b2_try, regularization_term=regularization_term)
        grad_b2[i,0] = (c2-c) / h
        
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            
            W1_try= W1
            W1_try[i,j] += h
            c2= ComputeCost(X=X, Y=Y, W1=W1_try, b1=b1, W2=W2, regularization_term=regularization_term)
            
            grad_W1[i,j] = (c2-c) / h

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = W2
            W2_try[i, j] += h
            c2 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1, W2=W2_try, regularization_term=regularization_term)

            grad_W2[i, j] = (c2 - c) / h

    return W1, b1, W2, b2

def ComputeGradsNumSlow(X, Y, W1, b1, W2, b2, regularization_term, h=1e-5):
    """
    Computes gradient descent updates on a batch of data with numerical computations of great precision, thus slower computations.
    Contributed by Josephine Sullivan for educational purposes for the DD2424 Deep Learning in Data Science course.

    :param X: Input data
    :param Y: One-hot representation of the true labels of input data X
    :param W1: Weight matrix of the first layer
    :param b1: Bias vector of the first layer
    :param W2: Weight matrix of the second layer
    :param b2: Bias vector of the second layer
    :param regularization_term: Contribution of the regularization in the weight updates

    :return: Weight and bias updates of the first and second layer of our network computed with numerical computations with high precision.
    """

    grad_W1= np.zeros(W1.shape)
    grad_b1= np.zeros(b1.shape)
    grad_W2= np.zeros(W2.shape)
    grad_b2= np.zeros(b2.shape)

    for i in range(b1.shape[0]):

        b1_try = b1[i,0]
        b1_try[i,0] -= h
        c1 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1_try, W2=W2, b2=b2, regularization_term=regularization_term)

        b1_try = b1[i,0]
        b1_try[i,0] += h
        c2 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1_try, W2=W2, b2=b2, regularization_term=regularization_term)
        grad_b1[i,0] = (c2-c1) / (2*h)

    for i in range(b2.shape[0]):
        b2_try = b2[i, 0]
        b2_try[i, 0] -= h
        c1 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1, W2=W2, b2=b2_try, regularization_term=regularization_term)

        b2_try = b2[i, 0]
        b2_try[i, 0] += h
        c2 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1, W2=W2, b2=b2_try, regularization_term=regularization_term)
        grad_b2[i, 0] = (c2 - c1) / (2 * h)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):

            W1_try = W1
            W1_try[i, j] -= h
            c1 = ComputeCost(X=X, Y=Y, W1=W1_try, b1=b1, W2=W2, regularization_term=regularization_term)

            W1_try = W1
            W1_try[i, j] += h
            c2 = ComputeCost(X=X, Y=Y, W1=W1_try, b1=b1, W2=W2, regularization_term=regularization_term)

            grad_W1[i, j] = (c2 - c1) / (2 * h)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):

            W2_try = W2
            W2_try[i, j] -= h
            c1 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1, W2=W2_try, regularization_term=regularization_term)

            W2_try = W2
            W2_try[i, j] += h
            c2 = ComputeCost(X=X, Y=Y, W1=W1, b1=b1, W2=W2_try, regularization_term=regularization_term)

            grad_W2[i, j] = (c2 - c1) / (2 * h)

    return W1, b1, W2, b2

def ComputeGradients(X, Y, W1, b1, W2, b2, regularization_term):

    """
    Computes gradient descent updates on a batch of data

    :param X: Input data
    :param Y: One-hot representation of the true labels of input data X
    :param W1: Weight matrix of the first layer
    :param b1: Bias vector of the first layer
    :param W2: Weight matrix of the second layer
    :param b2: Bias vector of the second layer
    :param regularization_term: Contribution of the regularization in the weight updates

    :return: Weight and bias updates of the first and second layer of our network
    """

    # Evaluate the classifier to the batch
    p, h, s1 = EvaluateClassifier(X=X, W1=W1, b1=b1, W2=W2, b2=b2)

    # Back-propagate second layer at first

    # Gradient of J w.r.t second bias vector is the g vector:
    g = (Y-p).T
    grad_b2= np.sum(g, axis=1).reshape(b2.shape[0],1) / X.shape[0]
    # Gradient of J w.r.t second weight matrix is the matrix:
    grad_W2 = np.dot(g.T, h.T) / 255 + 2 * regularization_term * W2

    # Back-propagate the gradient vector g to the first layer
    g= np.dot(g,W2) * np.where(s1 >0, 1, 0)

    grad_b1= np.sum(g, axis=1).reshape(b1.shape[0], 1) / X.shape[0]
    grad_W1= np.dot(g.T , X.T) / X.shape[0]

    # Add regularizers
    grad_W1+= 2 * regularization_term * W1
    grad_W2+= 2 * regularization_term * W2

    return grad_W1, grad_b1, grad_W2, grad_b2




