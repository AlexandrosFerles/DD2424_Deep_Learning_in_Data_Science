import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical as make_class_categorical
import _pickle as pickle
import pdb
from tqdm import tqdm
from scipy import ndarray
# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

# ------------------------------------------ LOAD BATCH -------------------------------------------

def LoadBatch(filename):
    """
    Loads a CIFAR-10 batch of data.

    :param filename: The path of the file in your local computer.
    :return: CIFAR-10 data X, their one-hot representation Y, and their true labels y
    """
    # borrowed from https://www.cs.toronto.edu/~kriz/cifar.html
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
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

# ------------------------------------------ WEIGHT INITIALIZATION --------------------------------


def initialize_weights(d, K, variance, mode):
    """
    Initializes the weight and bias matrices of the single-layer network
    through a Gaussian/Normal distribution.

    :param d: The dimensionality of the input data.
    :param K: The number of classes.
    :param variance: The variance of the Gaussian-distribution.
    :param mode: Selecting if instead of a normal distribution, a distribution based on
                 Xavier initialization will be applied.

    :return: Randomly initialized weight and bias matrices of the single-layer network.
    """

    np.random.seed(400)

    if mode =="simple xavier":
        variance = 1.0/float(d)
    elif mode == "xavier":
        variance = 2.0 / float(K+d)

    weight_matrix = np.random.normal(0,variance, (K,d))
    bias = np.random.normal(0 ,variance, (K,1))

    return weight_matrix, bias

# ------------------------------------------ NON-OVERFLOWING SOFTMAX IMPLEMENTATION ---------------


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

# ------------------------------------------ ASSIGNMENT INSTRUCTION FUNCTIONS ---------------------


def EvaluateClassifier(X, W, b):
    """
    Computes the output class probabilities of the network.

    :param X: Input data of the network.
    :param W: Weight matrix of the network.
    :param b: Bias vector of the network.

    :return: Softmax probabilities that predict the true class of the input data.
    """

    s = np.dot(W,X) + b
    P = softmax(s)
    return P

def ComputeCost(X, Y, W, b, regularization_term):
    """
    Computes the cross-entropy loss of the predictions derived using the
    W and b matrices on input data X.

    :param X: Input data of the single-layer network.
    :param Y: One-hot representation of the true labels of the data
    :param W: Weight matrix of the single-layer network.
    :param b: Bias vector of the single-layer network.
    :param regularization_term: Amount of regularization applied on the predictions.

    :return: Cross-entropy loss between the predictions and the ground truth.
    """

    P = EvaluateClassifier(X,W,b)

    cross_entropy_loss = 0
    for i in range(0, X.shape[1]):
        cross_entropy_loss-= np.log(np.dot(Y[:,i].T,P[:,i]))
    cross_entropy_loss/=float(X.shape[1])

    weight_sum = np.power(W,2).sum()

    return cross_entropy_loss + regularization_term*weight_sum


def ComputeGradients(X, Y, P, W, regularization_term):
    """
    Computes the gradient updates derived by minimizng the cross-entropy loss of the network.

    :param X: Input data.
    :param Y: One-hot representation of the true labels of the data.
    :param P: Softmax predictions derived by the forward pass of the network.
    :param W: Weight matrix of the network.
    :param regularization_term: Amount of regularization applied on the gradient updates.

    :return: Graient upodates to be applied in order to minimize the cross-entropy loss.
    """

    gradW = np.zeros((W.shape[0],W.shape[1]))
    gradb = np.zeros((W.shape[0], 1))

    for i in range(X.shape[1]):

        g = -(Y[:,i] - P[:,i]).T
        gradb += g.T.reshape(g.shape[0], 1)
        gradW += np.dot(g.T.reshape(g.shape[0], 1), X[:, i].T.reshape(1, X.shape[0]))

    # normalize
    gradW /= X.shape[1]
    gradW += 2*regularization_term*W
    gradb /= X.shape[1]

    return gradW, gradb

# ------------------------------------------ MINI-BATCH GRADIENT DESCENT FUNCTIONS ----------------

def MiniBatchGD(X, Y, validation_X, validation_Y, y, y_validation, GDparams, W, b, regularization_term,
                with_early_stopping, with_patience, patience=10, with_factor_decaying=False):
    """
    Performs mini batch-gradient descent computations.

    :param X: Input batch of data
    :param Y: One-hot representation of the true labels of the data.
    :param validation_X: Input batch of validation data.
    :param validation_Y: One-hot representation of the true labels of the validation data.
    :param y: True labels of the data.
    :param y_validation: True labels of the validation data.
    :param GDparams: Gradient descent parameters (number of mini batches to construct, learning rate, epochs)
    :param W: Weight matrix of the network.
    :param b: Bias vector of the network.
    :param regularization_term: Amount of regularization applied.
    :param with_early_stopping: Decides whether early stopping will be applied or not.
    :param with_patience: Decides whether patience will be taken into acoount when using early stopping.
    :param patience: Amount of patience epochs.
    :param with_factor_decaying: Decides whether decaying of the learning will be applied or not.

    :return: The weight and bias matrices learnt (trained) from the training process.
    """
    number_of_mini_batches = GDparams[0]
    if with_factor_decaying:
        eta = 0.1
    else:
        eta = GDparams[1]
    epoches = GDparams[2]

    cost = []
    val_cost = []


    early_stopping = 10e8
    early_stopping_cnt = 0

    save_W = np.copy(W)
    save_b = np.copy(b)

    temp_W = np.copy(W)
    temp_b = np.copy(b)

    for epoch in range(epoches):

        for batch in range(1, int(X.shape[1] / number_of_mini_batches)):
            start = (batch - 1) * number_of_mini_batches + 1
            end = batch * number_of_mini_batches + 1

            P = EvaluateClassifier(X[:, start:end], W, b)

            gradW, gradb = ComputeGradients(X=X[:, start:end], Y=Y[:, start:end], P=P, W=W,
                                            regularization_term=regularization_term)

            # Compute similarity between analytic and numerical computation
            # gradW_num, gradb_num = ComputeGradsNumSlow(X=X[:, start:end], Y=Y[:, start:end], y=y, W=W, b=b, regularization_term=regularization_term)
            # check_similarity(gradb, gradW, gradb_num, gradW_num)

            temp_W = np.copy(W)
            temp_b = np.copy(b)

            W -= eta * gradW
            b -= eta * gradb

        epoch_cost = ComputeCost(X=X, Y=Y, y=y, W=W, b=b, regularization_term=regularization_term)
        val_epoch_cost = ComputeCost(X=validation_X, Y=validation_Y, y=y_validation, W=W, b=b, regularization_term=regularization_term)

        # ---------EARLY STOPPING----------
        if with_early_stopping:
            if with_patience:
                if early_stopping - val_epoch_cost > 1e-6:
                    early_stopping_cnt = 0
                    early_stopping = val_epoch_cost
                else:
                    if early_stopping_cnt == patience:
                        print("Early stopping after " + str(epoch) + " epochs")
                        return save_W, save_b, cost, val_epoch_cost
                    else:
                        if early_stopping_cnt == 0:
                            save_W = temp_W
                            save_b = temp_b
                        early_stopping_cnt += 1
            else:
                if early_stopping - val_epoch_cost > 1e-6:
                    early_stopping = val_epoch_cost
                else:
                    print("Early stopping after " + str(epoch) + " epochs")
                    return temp_W, temp_b, cost, val_cost

        cost.append(epoch_cost)
        val_cost.append(val_epoch_cost)

        if with_factor_decaying:
            eta *= 0.9

    return W, b, cost, val_cost

# ------------------------------------------ VISUALIZING FUNCTIONS ----------------

def visualize_costs(cost, val_cost, regularization_term, eta, vis, name=None):
    """
    Visualizes the training-set and validation-set cost evolution.

    :param cost: Loss on the training-set trhough a period of epochs.
    :param val_cost: Loss on the validation-set trhough a period of epochs.
    :param regularization_term: Amount of regularization applied.
    :param eta: Learning rate
    :param vis: Decides whether the created image will be shown as an output.
    :param name: Name of the file that the image is stored (also serves for a decision whether the plot will
                 be stored at a local file or not)

    :return: None
    """
    if vis:
        plt.title('Cost function loss per epoch, $\lambda$=' + str(regularization_term) + ', $\eta$=' + str(eta))
        plt.plot(cost, 'g', label='Training set ')
        plt.plot(val_cost, 'r', label='Validation set')
        plt.legend(loc='upper right')
        plt.show()
        if name is not None:
            plt.savefig(name+".png")
        plt.clf()

def weights_vis(W, vis, name=None):
    """
    Visualizes the weights learned from a training process of the single-layer network.

    :param W: Weight matrix.
    :param vis: Decides whether the created image will be shown as an output.
    :param name: Name of the file that the image is stored (also serves for a decision whether the plot will
                 be stored at a local file or not)

    :return: None
    """
    if vis:
        # weight visualization
        images = []
        for img in W:
            raw_img=np.rot90(img.reshape(3, 32, 32).T, -1)
            image = ((raw_img-np.min(raw_img))/(np.max(raw_img)-np.min(raw_img)))
            images.append(image)
        fig1 = plt.figure(figsize = (20, 5))
        for idx in range(len(W)):
            ax = fig1.add_subplot(2, 5, idx+1)
            ax.set_title('Class %s'%(idx+1))
            ax.imshow(images[idx])
            ax.axis('off')
        plt.show()
        if name is not None:
            plt.savefig(name+".png")
        plt.clf()

# Function to display CIFAR-10 images, borrowed from
# https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib

def visualize_raw_image(image_array, display=True):
    """
    Visualizes a CIFAR-10 image in raw form (flattened 3072 numpy array).

    :param image_array: The numpy array representing the image
    :param display: Decides whether the image will be displayed or not.

    :return: None
    """
    img = image_array.reshape(3,32,32).transpose([1, 2, 0])
    if display:
        plt.imshow(img)

def visualize_image(image_array, display=True):
    """
    Visualizes a CIFAR-10 image in 3D form.

    :param image_array: The numpy array representing the image
    :param display: Decides whether the image will be displayed or not.

    :return: None
    """
    img = image_array.transpose([1, 2, 0])
    if display:
        plt.imshow(img)

# ------------------------------------------ IMAGE AUGMENTATION -----------------------------------

def random_rotation(image_array: ndarray, degree = 90):
    """
    Rotates image.

    :param image_array: The numpy array of the image.
    :param degree: Amount of degrees that the image will be rotated.

    :return: Rotated image.
    """
    return sk.transform.rotate(image_array, degree)
