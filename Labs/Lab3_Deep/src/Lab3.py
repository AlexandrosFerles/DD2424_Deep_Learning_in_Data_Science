import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical as make_class_categorical
import _pickle as pickle
from tqdm import tqdm

def LoadBatch(filename):
    """
    Loads batch based on the given filename and produces the X, Y, and y arrays

    :param filename: Path of the file
    :return: X, Y and y arrays
    """

    # borrowed from https://www.cs.toronto.edu/~kriz/cifar.html
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
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
    garbage = ix(dictionary, 2)
    X = np.transpose(dictionary[garbage]) / 255

    return X, Y, y

def initialize_weights(shapes_list, std=0.001):
    """
    Initializes the weight and bias arrays for the 2 layers of the network

    :param shapes_list: List that contains the shapes of the weight matrices of each layer. The number of layers can be found through
                        estimating the length of this list.
    :param variance (optional): The variance of the normal distribution that will be used for the initialization of the weights

    :return: Weights and bias arrays for each layer of the network stored in lists
    """

    np.random.seed(400)

    weights = []
    biases = []

    for shape in shapes_list:

        W = np.random.normal(0, std, size=(shape[0], shape[1]))
        b = np.zeros(shape=(shape[0], 1))

        weights.append(W)
        biases.append(b)

    return weights, biases

def he_initialization_k_layers(shapes_list):
    """
    He initialization on the weight matrices.

    :param shapes_list: List that contains the dimensions of each layer of the network.

    :return: Initialized weight and bias matrices based on He initialization of the weights.
    """

    weights = []
    biases = []

    for pair in shapes_list:

        weights.append(np.random.randn(pair[0], pair[1]) * np.sqrt(2 / float(pair[0])))
        biases.append(np.zeros(shape=(pair[0], 1)))

    return weights, biases

def ReLU(x):
    """
    Rectified Linear Unit function

    :param x: Input to the function

    :return: Output of ReLU(x)
    """

    return np.maximum(0, x)

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

def EvaluateClassifier(X, weights, biases):
    """
    Computes the Softmax output of the k-layer network, based on input data X and trained weight and bias arrays

    :param X: Input data
    :param weights: Weights arrays of the k layers
    :param biases: Bias vectors of the k layers

    :return: Softmax output of the trained network, along with the intermediate layer outpouts and activations
    """

    intermediate_outputs = [] # s's
    intermediate_activations = [] #h's

    s = np.dot(weights[0], X) + biases[0]
    intermediate_outputs.append(s)
    h = ReLU(s)
    intermediate_activations.append(h)

    for i in range(1, len(weights) - 1):

        s = np.dot(weights[i], intermediate_activations[-1]) + biases[i]
        intermediate_outputs.append(s)
        h = ReLU(s)
        intermediate_activations.append(h)

    s = np.dot(weights[-1], intermediate_activations[-1]) + biases[-1]
    p = softmax(s, axis=0)

    return p, intermediate_activations, intermediate_outputs

def predictClasses(p):
    """
    Predicts classes based on the softmax output of the network

    :param p: Softmax output of the network
    :return: Predicted classes
    """

    return np.argmax(p, axis=0)

def ComputeAccuracy(X, y, weights, biases):
    """
    Computes the accuracy of the feed-forward 2-layer network

    :param X: Input data
    :param weights: Weights arrays of the k layers
    :param biases: Bias vectors of the k layers

    :return: Accuracy performance of the neural network.
    """
    p, _, _ = EvaluateClassifier(X, weights, biases)
    predictions = predictClasses(p)

    accuracy = round(np.sum(np.where(predictions - y == 0, 1, 0)) * 100 / len(y), 2)

    return accuracy

def ComputeCost(X, Y, weights, biases, regularization_term=0):
    """
    Computes the cross-entropy loss on a batch of data.

    :param X: Input data
    :param y: Labels of the ground truth
    :param weights: Weights arrays of the k layers
    :param biases: Bias vectors of the k layers
    :param regularization_term: Amount of regularization applied.

    :return: Cross-entropy loss.
    """

    p, _, _ = EvaluateClassifier(X, weights, biases)

    cross_entropy_loss = -np.log(np.diag(np.dot(Y.T, p))).sum() / float(X.shape[1])

    weight_sum = 0
    for weight in weights:

        weight_sum += np.power(weight, 2).sum()

    return cross_entropy_loss + regularization_term * weight_sum

def ComputeGradsNumSlow(X, Y, weights, biases, start_index=0, h=1e-5):
    """
    Computes gradient descent updates on a batch of data with numerical computations of great precision, thus slower computations.
    Contributed by Josephine Sullivan for educational purposes for the DD2424 Deep Learning in Data Science course.

    :param X: Input data.
    :param Y: One-hot representation of the true labels of input data X.
    :param weights: Weights arrays of the k layers.
    :param biases: Bias vectors of the k layers.
    :param start_index: In case there are already some weights and bias precomputed, we need to compute the numerical gradients for
                        those weights and bias that have other shapes (the 2 last layers in fact).

    :return: Weight and bias updates of the k layers of our network computed with numerical computations with high precision.
    """

    grad_weights = []
    grad_biases = []

    for layer_index in range(start_index, len(weights)):

        W = weights[layer_index]
        b = biases[layer_index]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros(b.shape)

        for i in tqdm(range(b.shape[0])):
            b_try = np.copy(b)
            b_try[i, 0] -= h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c1 = ComputeCost(X=X, Y=Y, weights=weights, biases=temp_biases)
            b_try = np.copy(b)
            b_try[i, 0] += h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c2 = ComputeCost(X=X, Y=Y, weights=weights, biases=temp_biases)

            grad_b[i, 0] = (c2 - c1) / (2 * h)

        grad_biases.append(grad_b)

        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i, j] -= h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c1 = ComputeCost(X=X, Y=Y, weights=temp_weights, biases=biases)
                W_try = np.copy(W)
                W_try[i, j] += h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c2 = ComputeCost(X=X, Y=Y, weights=temp_weights, biases=biases)

                grad_W[i, j] = (c2 - c1) / (2 * h)

        grad_weights.append(grad_W)

    return grad_weights, grad_biases

def ComputeGradsNumSlowBatchNorm(X, Y, weights, biases, start_index=0, h=1e-5):
    """
    Computes gradient descent updates on a batch of data with numerical computations of great precision, thus slower computations.
    Contributed by Josephine Sullivan for educational purposes for the DD2424 Deep Learning in Data Science course.

    :param X: Input data.
    :param Y: One-hot representation of the true labels of input data X.
    :param weights: Weights arrays of the k layers.
    :param biases: Bias vectors of the k layers.
    :param start_index: In case there are already some weights and bias precomputed, we need to compute the numerical gradients for
                        those weights and bias that have other shapes (the 2 last layers in fact).

    :return: Weight and bias updates of the k layers of our network computed with numerical computations with high precision.
    """

    grad_weights = []
    grad_biases = []

    for layer_index in range(start_index, len(weights)):

        W = weights[layer_index]
        b = biases[layer_index]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros(b.shape)

        for i in tqdm(range(b.shape[0])):
            b_try = np.copy(b)
            b_try[i, 0] -= h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c1 = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases)
            b_try = np.copy(b)
            b_try[i, 0] += h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c2 = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases)

            grad_b[i, 0] = (c2 - c1) / (2 * h)

        grad_biases.append(grad_b)

        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i, j] -= h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c1 = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases)
                W_try = np.copy(W)
                W_try[i, j] += h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c2 = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases)

                grad_W[i, j] = (c2 - c1) / (2 * h)

        grad_weights.append(grad_W)

    return grad_weights, grad_biases

def ComputeGradients(X, Y, weights, biases, p, outputs, activations, regularization_term=0):
    """
    Computes gradient descent updates on a batch of data

    :param X: Input data
    :param Y: One-hot representation of the true labels of input data X
    :param weights: Weight matrices of the k layers
    :param biases: Bias vectors of the k layers
    :param p: Softmax probabilities (predictions) of the network over classes.
    :param outputs: True outputs of the intermediate layers of the network.
    :param activations: ReLU activations of the intermediate layers of the network.
    :param regularization_term: Contribution of the regularization in the weight updates

    :return: Weight and bias updates of the first and second layer of our network
    """

    # Back-propagate output layer at first

    weight_updates = []
    bias_updates = []

    g = p - Y
    bias_updates.append(g.sum(axis=1).reshape(biases[-1].shape))
    weight_updates.append(np.dot(g, activations[-1].T))

    for i in reversed(range(len(weights) -1)):
    # Back-propagate the gradient vector g to the layer before

        g = np.dot(g.T, weights[i+1])
        ind = 1 * (outputs[i] > 0)
        g = g.T * ind

        if i == 0:
            weight_updates.append(np.dot(g, X.T))
        else:
            weight_updates.append(np.dot(g, activations[i-1].T))

        bias_updates.append(np.sum(g, axis=1).reshape(biases[i].shape))

    for elem in weight_updates:
        elem /= X.shape[1]

    for elem in bias_updates:
        elem /= X.shape[1]

    # Reverse the updates to match the order of the layers
    weight_updates = list(reversed(weight_updates)).copy()
    bias_updates = list(reversed(bias_updates)).copy()

    # Add regularizers
    for index in range(len(weight_updates)):
        weight_updates[index] += 2*regularization_term * weight_updates[index]

    return weight_updates, bias_updates

def check_similarity(grad_weights, grad_biases, num_weights, num_biases):
    """
    Compares the gradients of both the analytical and numerical method and prints out a message of result
    or failure, depending on how close these gradients are between each other.

    :param grad_weights: Analytically computed gradients of the weights
    :param grad_biases: Analytically computed gradients of the biases
    :param num_weights: Numerically computed gradients of the weights
    :param num_biases: Numerically computed gradients of the biases

    :return: None
    """

    for layer_index in range(len(grad_weights)):

        print('-----------------')
        print(f'Layer no. {layer_index+1}:')

        weight_abs = np.abs(grad_weights[layer_index] - num_weights[layer_index])
        bias_abs = np.abs(grad_biases[layer_index] - num_biases[layer_index])

        weight_nominator = np.average(weight_abs)
        bias_nominator = np.average(bias_abs)

        grad_weight_abs = np.absolute(grad_weights[layer_index])
        grad_weight_num_abs = np.absolute(num_weights[layer_index])

        grad_bias_abs = np.absolute(grad_biases[layer_index])
        grad_bias_num_abs = np.absolute(num_biases[layer_index])

        sum_weight = grad_weight_abs + grad_weight_num_abs
        sum_bias = grad_bias_abs + grad_bias_num_abs

        print(f'Deviation on weight matrix: {weight_nominator / np.amax(sum_weight)}')
        print(f'Deviation on bias vector: {bias_nominator / np.amax(sum_bias)}')

def initialize_momentum(arrays):
    """
    Initializes the momentum arrays to zero numpy arrays.

    :param matrices: Weights or bias that need corresponding momentum arrays.
    :return: Numpy zeros for each layer of the same shape
    """
    momentum_matrices = []
    for elem in arrays:
        momentum_matrices.append(np.zeros(elem.shape))
    return momentum_matrices

def add_momentum(weights, grad_weights, momentum_weights, biases, grad_biases, momentum_biases, eta, momentum_term=0.9):
    """
    Add momentum to an array (weight or bias) of the network.

    :param weights: The weight matrices of the k layers.
    :param grad_weights: The gradient updatea of the weights.
    :param momentum_weights: Momentum arrays (v) of the weights.
    :param biases: The bias vector of the k layers.
    :param grad_biases: The gradient updates for the biases.
    :param momentum_biases: Momentum vectors (v) of the weights.
    :param eta: Learning rate of the network.
    :param momentum_term: Amount of momentum to be taken into account in the updates.

    :return: Updated weights and biases of the network with momentum contribution, updated momentumm arrays for the
             weights and biases of the network.
    """

    updated_weights = []
    updated_biases = []

    for index in range(len(weights)):

        new_momentum_weight = momentum_term * momentum_weights[index] + eta * grad_weights[index]
        momentum_weights[index] = new_momentum_weight
        updated_weights.append(weights[index] - new_momentum_weight)

        new_momentum_bias = momentum_term * momentum_biases[index] + eta * grad_biases[index]
        momentum_biases[index] = new_momentum_bias
        updated_biases.append(biases[index] - new_momentum_bias)


    return updated_weights, updated_biases, momentum_weights, momentum_biases

def MiniBatchGDwithMomentum(X, Y, X_validation, Y_validation, y_validation, GDparams, weights, biases,
                            regularization_term=0, momentum_term=0.9):
    """
    Performs mini batch-gradient descent computations.

    :param X: Input batch of data
    :param Y: One-hot representation of the true labels of the data.
    :param X_validation: Input batch of validation data.
    :param Y_validation: One-hot representation of the true labels of the validation data.
    :param y_validation: True labels of the validation data.
    :param GDparams: Gradient descent parameters (number of mini batches to construct, learning rate, epochs)
    :param weights: Weight matrices of the k layers
    :param biases: Bias vectors of the k layers
    :param regularization_term: Amount of regularization applied.

    :return: The weight and bias matrices learnt (trained) from the training process, loss in training and validation set.
    """
    number_of_mini_batches = GDparams[0]
    eta = GDparams[1]
    epoches = GDparams[2]

    cost = []
    val_cost = []

    momentum_weights = initialize_momentum(weights)
    momentum_biases = initialize_momentum(biases)

    original_training_cost= ComputeCost(X, Y, weights, biases, regularization_term)
    # print('Training set loss before start of training process: '+str(ComputeCost(X, Y, W1, W2, b1, b2, regularization_term)))

    best_weights = weights
    best_biases = biases

    best_validation_set_accuracy = 0

    for _ in tqdm(range(epoches)):
        # for epoch in range(epoches):

        for batch in range(1, int(X.shape[1] / number_of_mini_batches)):
            start = (batch - 1) * number_of_mini_batches + 1
            end = batch * number_of_mini_batches + 1

            p, intermediate_activations, intermediate_outputs = EvaluateClassifier(X[:, start:end], weights, biases)

            grad_weights, grad_biases = ComputeGradients(X[:, start:end], Y[:, start:end], weights, biases, p, intermediate_outputs, intermediate_activations, regularization_term)

            weights, biases, momentum_weights, momentum_biases = add_momentum(weights, grad_weights, momentum_weights, biases, grad_biases, momentum_biases, eta, momentum_term)

        validation_set_accuracy = ComputeAccuracy(X_validation, y_validation, weights, biases)

        if validation_set_accuracy > best_validation_set_accuracy:

            best_weights = weights
            best_biases = biases
            best_validation_set_accuracy = validation_set_accuracy

        epoch_cost = ComputeCost(X, Y, weights, biases, regularization_term)
        # print('Training set loss after epoch number '+str(epoch)+' is: '+str(epoch_cost))
        if epoch_cost > 3 * original_training_cost:
            break
        val_epoch_cost = ComputeCost(X_validation, Y_validation, weights, biases, regularization_term)

        cost.append(epoch_cost)
        val_cost.append(val_epoch_cost)

        # Decay the learning rate
        eta *= 0.95

    return best_weights, best_biases, cost, val_cost

# ---------------------- BATCH NORMALIZATION FUNCTIONS ---------------------- #

def BatchNormalize(s, mean_s, var_s, epsilon=1e-10):
    """
    Normalizes the scores of a batch based on their mean and variance.

    :param s: Scores evaluated as output of a layer of the network.
    :param mean_s: Mean of the scores.
    :param var_s: Variance of the scores.
    :param epsilon: A small number that is present to ensure that no division by zero will be performed.

    :return: The normalized scores,
    """

    diff = s - mean_s

    return diff / (np.sqrt(var_s + epsilon))

def ForwardPassBatchNormalization(X, weights, biases, exponentials= None):
    """
    Evaluates the forward pass result of the classifier network using batch normalization.

    :param X: Input data.
    :param weights: Weight arrays of the k-layer network.
    :param biases: Bias vectors of the k-layer network.

    :return: Softmax probabilities (predictions) of the true labels of the data.
    """

    s = np.dot(weights[0], X) + biases[0]

    intermediate_outputs = [s]

    if exponentials is not None:

        exponential_means = exponentials[0]
        exponential_variances = exponentials[1]

        mean_s = exponential_means[0]
        var_s = exponential_variances[0]

    else:

        mean_s = s.mean(axis=1).reshape(s.shape[0], 1)
        var_s = s.var(axis=1).reshape(s.shape[0], 1)

        means = [mean_s]
        variances = [var_s]

    normalized_score = BatchNormalize(s, mean_s, var_s)

    batch_normalization_outputs = [normalized_score]
    batch_normalization_activations = [ReLU(normalized_score)]

    for index in range(1, len(weights) - 1):

        s = np.dot(weights[index], batch_normalization_activations[-1]) + biases[index]

        intermediate_outputs.append(s)

        if exponentials is None:
            mean_s = s.mean(axis=1).reshape(s.shape[0], 1)
            var_s = s.var(axis=1).reshape(s.shape[0], 1)

            means.append(mean_s)
            variances.append(var_s)

        else:

            mean_s = exponential_means[index]
            var_s = exponential_variances[index]

        normalized_score = BatchNormalize(s, mean_s, var_s)

        batch_normalization_outputs.append(normalized_score)
        batch_normalization_activations.append(ReLU(normalized_score))

    s = np.dot(weights[-1], batch_normalization_activations[-1]) + biases[-1]

    p = softmax(s, axis=0)

    if exponentials is not None:
        return p
    else:
        return p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances

def ComputeAccuracyBatchNormalization(X, y, weights, biases, exponentials = None):
    """
    Computes the accuracy of the feed-forward k-layer network

    :param X: Input data
    :param weights: Weights arrays of the k layers
    :param biases: Bias vectors of the k layers
    :param exponentials: Contains the exponential means and variances computed, they are used in call after training.

    :return: Accuracy performance of the neural network.
    """
    if exponentials is not None:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)
    else:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)[0]
    predictions = predictClasses(p)

    accuracy = round(np.sum(np.where(predictions - y == 0, 1, 0)) * 100 / len(y), 2)

    return accuracy

def ComputeCostBatchNormalization(X, Y, weights, biases, regularization_term, exponentials=None):
    """
    Computes the cross-entropy loss on a batch of data.

    :param X: Input data
    :param y: Labels of the ground truth
    :param weights: Weights arrays of the k layers
    :param biases: Bias vectors of the k layers
    :param regularization_term: Amount of regularization applied.
    :param exponentials: Contains the exponential means and variances computed, they are used in call after training.

    :return: Cross-entropy loss.
    """
    if exponentials is None:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)[0]
    else:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)

    cross_entropy_loss = 0
    for datum_index in range(X.shape[1]):

        cross_entropy_loss -= np.log(np.dot(Y[:, datum_index].T, p[:, datum_index]))

    weight_sum = 0
    for weight in weights:

        weight_sum += np.power(weight, 2).sum()

    return (cross_entropy_loss / float(X.shape[1]))+ regularization_term * weight_sum

def BatchNormBackPass(g, s, mean_s, var_s, epsilon=1e-10):

    # First part of the gradient:
    V_b = (var_s+ epsilon) ** (-0.5)
    V_b = np.power( (var_s+ epsilon), -0.5)
    part_1 = g * V_b

    # Second part pf the gradient
    diff = s - mean_s
    grad_J_vb = -0.5 * np.sum(g * np.power((var_s+epsilon), -1.5) * diff, axis=1)
    grad_J_vb = -0.5 * np.sum(g * (var_s+epsilon) ** (-1.5) * diff, axis=1)
    grad_J_vb = np.expand_dims(grad_J_vb, axis=1)
    part_2 = (2/float(s.shape[1])) * grad_J_vb * diff

    # Third part of the gradient
    grad_J_mb = -np.sum(g * V_b, axis=1)
    grad_J_mb = np.expand_dims(grad_J_mb, axis=1)
    part_3 = grad_J_mb / float(s.shape[1])

    return part_1 + part_2 + part_3

def BackwardPassBatchNormalization(X, Y, weights, biases, p, bn_outputs, bn_activations, intermediate_outputs, means, variances, regularization_term):

    # Output layer
    g = p - Y

    bias_updates = [np.sum(g, axis=1).reshape(biases[-1].shape)]
    weight_updates = [np.dot(g, bn_activations[-1].T)]

    g = np.dot(g.T, weights[-1])
    ind = 1 * (bn_outputs[-1] > 0)
    g = g.T * ind

    for layer in reversed(range(len(weights) -1)):

        g = BatchNormBackPass(g, intermediate_outputs[layer], means[layer], variances[layer])

        if layer > 0:

            bias_updates.append(np.sum(g, axis=1).reshape(biases[layer].shape))
            weight_updates.append(np.dot(g, bn_activations[layer-1].T))

            g = np.dot(g.T, weights[layer])
            ind = 1 * (bn_outputs[layer - 1] > 0)
            g = g.T * ind

        else:

            bias_updates.append(np.sum(g, axis=1).reshape(biases[layer].shape))
            weight_updates.append(np.dot(g, X.T))

    weight_updates=list(reversed(weight_updates))
    bias_updates=list(reversed(bias_updates))

    for index, elem in enumerate(weight_updates):

        elem /= X.shape[1]
        bias_updates[index] /= X.shape[1]
        elem += 2 * regularization_term * weights[index]

    return weight_updates, bias_updates

def ComputeGradsNumSlowBatchNorm(X, Y, weights, biases, start_index=0, h=1e-5):
    """
    Computes gradient descent updates on a batch of data with numerical computations of great precision, thus slower computations.
    Contributed by Josephine Sullivan for educational purposes for the DD2424 Deep Learning in Data Science course.

    :param X: Input data.
    :param Y: One-hot representation of the true labels of input data X.
    :param weights: Weights arrays of the k layers.
    :param biases: Bias vectors of the k layers.
    :param start_index: In case there are already some weights and bias precomputed, we need to compute the numerical gradients for
                        those weights and bias that have other shapes (the 2 last layers in fact).

    :return: Weight and bias updates of the k layers of our network computed with numerical computations with high precision.
    """

    grad_weights = []
    grad_biases = []

    for layer_index in range(start_index, len(weights)):

        W = weights[layer_index]
        b = biases[layer_index]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros(b.shape)

        for i in tqdm(range(b.shape[0])):
            b_try = np.copy(b)
            b_try[i, 0] -= h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c1 = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases, regularization_term=0)
            b_try = np.copy(b)
            b_try[i, 0] += h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c2 = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases, regularization_term=0)

            grad_b[i, 0] = (c2 - c1) / (2 * h)

        grad_biases.append(grad_b)

        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i, j] -= h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c1 = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases, regularization_term=0)
                W_try = np.copy(W)
                W_try[i, j] += h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c2 = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases, regularization_term=0)

                grad_W[i, j] = (c2 - c1) / (2 * h)

        grad_weights.append(grad_W)

    return grad_weights, grad_biases

def ExponentialMovingAverage(means, exponential_means, variances, exponential_variances, a=0.99):

    for index, elem in enumerate(exponential_means):

        exponential_means[index] = a * elem + (1-a) * means[index]
        exponential_variances[index] = a * exponential_variances[index] + (1-a) * variances[index]

    return [exponential_means, exponential_variances]


def MiniBatchGDBatchNormalization(X, Y, X_validation, Y_validation, y_validation, GDparams, weights, biases, regularization_term, momentum_term=0.9):
    """
    Performs mini-gradient descent training in the weights and biases of the network, and computes
    the loss on training and validation sets, and the exponential moving averages.

    :param X: Input training data of the network.
    :param Y: One-hot representations of X.
    :param X_validation: Validation input data.
    :param Y_validation: One-hot representations of X_validation.
    :param y_validation: True labels of the validation data
    :param GDparams: Gradient-descent parameters (number of mini batches per epoch, learning rate, epochs)
    :param weights: Original pre-training weights of the network.
    :param biases: Original pre-training biases of the network.
    :param regularization_term: Amount of regularization applied to the weights.
    :param momentum_term: Percentage of previous state to account for in every weight and bias update.

    :return: Trained weights and biases, loss on training and validation set per training epoch, expoenntial moving means and variances.
    """

    train_loss = [ComputeCostBatchNormalization(X, Y, weights, biases, regularization_term)]
    validation_loss = [ComputeCostBatchNormalization(X_validation, Y_validation, weights, biases, regularization_term)]

    momentum_weights, momentum_biases = initialize_momentum(weights), initialize_momentum(biases)

    for epoch in tqdm(range(GDparams[2])):

        for batch in range(1, int(X.shape[1] / GDparams[0])):

            start = (batch - 1) * GDparams[0]
            end = batch * GDparams[0]

            p, bn_activations, bn_outputs, intermediate_scores, mean, var = ForwardPassBatchNormalization(X[:, start:end], weights, biases)
            grad_LW, grad_Lb = BackwardPassBatchNormalization(X[:, start:end], Y[:, start:end], weights, biases, p, intermediate_scores, bn_activations, intermediate_scores, mean, var, regularization_term)

            if (epoch == 0) and (start == 0):

                exponentials = [mean, var]

                for i in range(len(weights)):
                    weights[i] -= momentum_weights[i]
                    biases[i] -= momentum_biases[i]

            else:

                exponentials = ExponentialMovingAverage(mean, exponentials[0], var, exponentials[1])

                weights, biases, momentum_weights, momentum_biases = add_momentum(weights, grad_LW, momentum_weights, biases, grad_Lb, momentum_biases, GDparams[1])

        train_loss.append(ComputeCostBatchNormalization(X, Y, weights, biases, regularization_term))
        if train_loss[-1] > 3 * train_loss[0]:
            print(f'Train loss at epoch no.{epoch} loss exceeded by 3 times or more the original train loss. Exiting training process...')
            break
        validation_loss.append(ComputeCostBatchNormalization(X_validation, Y_validation, weights, biases, regularization_term))

        GDparams[1] *= 0.95

    return weights, biases, train_loss, validation_loss, exponentials

def visualize_costs(loss, val_loss, display=False, title=None, save_name=None, save_path='../figures/'):
    """
    Visualization and saving the losses of the network.

    :param loss: Loss of the network.
    :param val_loss: Loss of the network in the validation set.
    :param display: (Optional) Boolean, set to True for displaying the loss evolution plot.
    :param title: (Optional) Title of the plot.
    :param save_name: (Optional) name of the file to save the plot.
    :param save_path: (Optional) Path of the folder to save the plot in your local computer.

    :return: None

    """

    if title is not None:
        plt.title(title)

    plt.plot(loss, 'g', label='Training set ')
    plt.plot(val_loss, 'r', label='Validation set')
    plt.legend(loc='upper right')

    if save_name is not None:
        if save_path[-1] != '/':
            save_path += '/'
        plt.savefig(save_path + save_name)

    if display:
        plt.show()

    plt.clf()

def create_sets():
    """
    Creates the full dataset, containing all the available data for training except 1000 images
    used for the validation set.

    :return: Training, validation and test sets (features, ground-truth labels, and their one-hot representation
    """

    X_training_1, Y_training_1, y_training_1 = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_training_2, Y_training_2, y_training_2 = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_training_3, Y_training_3, y_training_3 = LoadBatch('../../cifar-10-batches-py/data_batch_3')
    X_training_4, Y_training_4, y_training_4 = LoadBatch('../../cifar-10-batches-py/data_batch_4')
    X_training_5, Y_training_5, y_training_5 = LoadBatch('../../cifar-10-batches-py/data_batch_5')

    X_training = np.concatenate((X_training_1, X_training_3), axis=1)
    X_training = np.copy(np.concatenate((X_training, X_training_4), axis=1))
    X_training = np.copy(np.concatenate((X_training, X_training_5), axis=1))

    X_training = np.concatenate((X_training, X_training_2[:, :9000]), axis=1)

    Y_training = np.concatenate((Y_training_1, Y_training_3), axis=1)
    Y_training = np.copy(np.concatenate((Y_training, Y_training_4), axis=1))
    Y_training = np.copy(np.concatenate((Y_training, Y_training_5), axis=1))

    Y_training = np.concatenate((Y_training, Y_training_2[:, :9000]), axis=1)

    y_training = y_training_1 + y_training_3 + y_training_4 + y_training_5 + y_training_2[:9000]

    X_validation = np.copy(X_training_2[:, 9000:])
    Y_validation = np.copy(Y_training_2[:, 9000:])
    y_validation = y_training_2[9000:]

    X_test, _, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    mean = np.mean(X_training)
    X_training -= mean
    X_validation -= mean
    X_test -= mean

    return [X_training, Y_training, y_training], [X_validation, Y_validation, y_validation], [X_test, y_test]

def exercise_1():

    X_training_1, Y_training_1, y_training_1 = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_training_2, Y_training_2, y_training_2 = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, _, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    mean = np.mean(X_training_1)
    X_training_1 -= mean
    X_training_2 -= mean
    X_test -= mean

    """
    Check with numerically computed gradients for a 2-layer network
    """

    # weights, biases = initialize_weights([[50, 3072], [10, 50]])
    # W1_num = np.load('grad_W1_num.npy')
    # W2_num = np.load('grad_W2_num.npy')
    #
    # b1_num = np.load('grad_b1_num.npy')
    # b2_num = np.load('grad_b2_num.npy')
    #
    # num_weights = [W1_num, W2_num]
    # num_biases = [b1_num, b2_num]
    #
    # p, activations, outputs = EvaluateClassifier(X_training_1[:, 0:2], weights, biases)
    # grad_weights, grad_biases = ComputeGradients(X_training_1[:, 0:2], Y_training_1[:, 0:2], weights, biases, p, outputs, activations)
    #
    # check_similarity(grad_weights, grad_biases, num_weights, num_biases)

    """
    Check with numerically computed gradients for a 3-layer network
    """

    # grad_weights_3_num, grad_bias_3_num = ComputeGradsNumSlow(X_training_1[:, 0:2], Y_training_1[:, 0:2], weights, biases)

    # weights, biases = initialize_weights([[50, 3072], [20, 50], [10, 20]])
    #
    # w1_num = np.load('3_layers_num_weights0.npy')
    # w2_num = np.load('3_layers_num_weights1.npy')
    # w3_num = np.load('3_layers_num_weights2.npy')
    #
    # b1_num = np.load('3_layers_num_bias0.npy')
    # b2_num = np.load('3_layers_num_bias1.npy')
    # b3_num = np.load('3_layers_num_bias2.npy')
    #
    # grad_weights_3_num = [w1_num, w2_num, w3_num]
    # grad_bias_3_num = [b1_num, b2_num, b3_num]
    #
    # p, activations, outputs = EvaluateClassifier(X_training_1[:, 0:2], weights, biases)
    # grad_weights, grad_biases = ComputeGradients(X_training_1[:, 0:2], Y_training_1[:, 0:2], weights, biases, p, outputs, activations)
    #
    # check_similarity(grad_weights, grad_biases, grad_weights_3_num, grad_bias_3_num)

    """
    Check with numerically computed gradients for a 4-layer network
    """
    weights, biases = initialize_weights([[50, 3072], [20, 50], [15, 20], [10, 15]])

    grad_weights_4_num, grad_bias_4_num = ComputeGradsNumSlow(X_training_1[:, 0:2], Y_training_1[:, 0:2], weights,
                                                              biases, start_index=len(weights) - 2)

    p, activations, outputs = EvaluateClassifier(X_training_1[:, 0:2], weights, biases)
    grad_weights, grad_biases = ComputeGradients(X_training_1[:, 0:2], Y_training_1[:, 0:2], weights, biases, p,
                                                 outputs, activations)

    check_similarity(grad_weights, grad_biases, grad_weights_4_num, grad_bias_4_num)

def exercise_2():

    X_training_1, Y_training_1, y_training_1 = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_training_2, Y_training_2, y_training_2 = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, _, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    mean = np.mean(X_training_1)
    X_training_1 -= mean
    X_training_2 -= mean
    X_test -= mean
    """
    Test that you are able to replicate the results of a 2-layer network
    """
    # weights, biases = initialize_weights([[50, 3072], [10, 50]])
    #
    # GD_params = [100, 0.0171384811847413, 10]
    #
    # weights, biases, training_cost, validation_cost = MiniBatchGDwithMomentum(X_training_1,
    #                                                                           Y_training_1,
    #                                                                           X_training_2,
    #                                                                           Y_training_2,
    #                                                                           y_training_2,
    #                                                                           GD_params,
    #                                                                           weights, biases,
    #                                                                           regularization_term=0.0001)
    #
    # validation_set_accuracy = ComputeAccuracy(X_training_2, y_training_2, weights, biases)
    # print('Validation set accuracy of this setting: ', validation_set_accuracy)

    """
    Try with a 3-layer network
    """

    # What happens after a few epochs? Are you learning anything?
    # weights, biases = initialize_weights([[50, 3072], [30, 50], [10,30]])
    #
    # GD_params = [100, 0.0171384811847413, 15]
    #
    # weights, biases, training_cost, validation_cost = MiniBatchGDwithMomentum(X_training_1,
    #                                                                           Y_training_1,
    #                                                                           X_training_2,
    #                                                                           Y_training_2,
    #                                                                           y_training_2,
    #                                                                           GD_params,
    #                                                                           weights, biases,
    #                                                                           regularization_term=0.0001)
    #
    # for i in range(len(training_cost)):
    #
    #     print(f'Cost at training epoch {i+1} is {training_cost[i]}')

    # What happens if you play around with the learning rate?

    # for eta in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
    #
    #     weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])
    #
    #     print('------------------')
    #     print('Eta: ', eta)
    #     GD_params = [100, eta, 10]
    #
    #     weights, biases, training_cost, validation_cost = MiniBatchGDwithMomentum(X_training_1,
    #                                                                               Y_training_1,
    #                                                                               X_training_2,
    #                                                                               Y_training_2,
    #                                                                               y_training_2,
    #                                                                               GD_params,
    #                                                                               weights, biases,
    #                                                                               regularization_term=0.0001)
    #
    #     for i in range(len(training_cost)):
    #         print(f'Cost at training epoch {i+1} is {training_cost[i]}')

    # What happens if you use He initialization?

    weights_he, biases_he = he_initialization_k_layers([[50, 3072], [50, 30], [10, 30]])

    GD_params = [100, 0.0171384811847413, 15]

    weights_he, biases_he, training_cost_he, validation_cost_he = MiniBatchGDwithMomentum(X_training_1,
                                                                              Y_training_1,
                                                                              X_training_2,
                                                                              Y_training_2,
                                                                              y_training_2,
                                                                              GD_params,
                                                                              weights_he, biases_he,
                                                                              regularization_term=0.0001)

    for i in range(len(training_cost_he)):
        print(f'Cost of training with He initialization at epoch {i+1} is {training_cost_he}')

def exercise_3():

    X_training_1, Y_training_1, y_training_1 = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_training_2, Y_training_2, y_training_2 = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, _, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    mean = np.mean(X_training_1, axis=0).reshape(1, X_training_1.shape[1])
    X_training_1 -= mean
    X_training_2 -= mean
    X_test -= mean

    def compare_with_numerical_gradients():

        """
        Convince yourself that the backward pass works by comparing with
        numerical computed gradients.
        """

        """
        Compare for 2 layers
        """
        weights, biases = initialize_weights([[50, 3072], [10, 50]])

        p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances = ForwardPassBatchNormalization(X_training_1[:, :4], weights, biases)
        weights_2, biases_2 = BackwardPassBatchNormalization(X_training_1[:, :4], Y_training_1[:, :4], weights, biases, p, batch_normalization_outputs, batch_normalization_activations, intermediate_outputs, means, variances, regularization_term=0)

        two_layers = np.load('2_layers.npz')

        check_similarity(weights_2, biases_2, [two_layers['w0'], two_layers['w1'] ], [two_layers['b0'], two_layers['b1']])

        """
        Compare for 3 layers
        """

        weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])
        w3_num, b3_num = ComputeGradsNumSlowBatchNorm(X_training_1[:, :2], Y_training_1[:, :2], weights, biases)

        np.savez('3_layers_num', w0=w3_num[0], w1=w3_num[1], w2=w3_num[2], b0=b3_num[0], b1=b3_num[1], b2=b3_num[2])

        weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

        p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances = ForwardPassBatchNormalization(X_training_1[:, :2], weights, biases)

        weights_3, biases_3 = BackwardPassBatchNormalization(X_training_1[:, :2], Y_training_1[:, :2], weights, biases,
                                                             p, batch_normalization_outputs,
                                                             batch_normalization_activations, intermediate_outputs,
                                                             means, variances, regularization_term=0)

        check_similarity(weights_3, biases_3, w3_num, b3_num)
        """
        Compare for 4 layers
        """
        weights, biases = initialize_weights([[50, 3072], [20, 50], [15, 20], [10, 15]])

        p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances = ForwardPassBatchNormalization(X_training_1[:, :4], weights, biases)
        weights_4, biases_4 = BackwardPassBatchNormalization(X_training_1[:, :4], Y_training_1[:, :4], weights, biases, p, batch_normalization_outputs, batch_normalization_activations, intermediate_outputs, means, variances, regularization_term=0)

        four_layers = np.load('4_layers.npz')

        check_similarity(weights_4, biases_4, [four_layers['w0'], four_layers['w1'], four_layers['w2'], four_layers['w3']], [four_layers['b0'], four_layers['b1'], four_layers['b2'], four_layers['b3']])

    def sanity():

        weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])
        GD_params = [100, 0.05, 200]

        best_weights, best_biases, training_loss, validation_loss, exponentials = MiniBatchGDBatchNormalization(X_training_1[:, :1000], Y_training_1[:, :1000], X_training_2[:, :1000], Y_training_2[:, :1000], y_training_2[:1000], GD_params, weights, biases, regularization_term=1e-6)

        visualize_costs(training_loss, validation_loss, display=True, title='Overfitting at a small dataset')
        print(f'Training set accuracy: {ComputeAccuracyBatchNormalization(X_training_1[:, :100], y_training_1[:100], best_weights, best_biases)}')
        print(f'Validation set accuracy: {ComputeAccuracyBatchNormalization(X_training_2[:, :100], y_training_2[:100], best_weights, best_biases)}')

    def random_search():

        for eta in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
            print('-----------------------')
            print('eta: ', eta)

            GD_params = [100, eta, 5]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization( X_training_1,
                                                                                                                                 Y_training_1,
                                                                                                                                 X_training_2,
                                                                                                                                 Y_training_2,
                                                                                                                                 y_training_2,
                                                                                                                                 GD_params,
                                                                                                                                 weights,
                                                                                                                                 biases,
                                                                                                                                 regularization_term=0.000001)

            print()
            for epoch, loss in enumerate(cost):
                print(f'Cross-entropy loss at epoch no.{epoch}: {loss}')
            print()



            print(f'Validation set accuracy: {ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights, best_biases, exponentials=[exponential_means, exponential_variances])}')

        for eta in np.arange(0.15, 1.05, 0.05):

            print('-----------------------')
            print('eta: ', eta)

            GD_params = [100, eta, 5]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(    X_training_1,
                                                                                                                                    Y_training_1,
                                                                                                                                    X_training_2,
                                                                                                                                    Y_training_2,
                                                                                                                                    y_training_2,
                                                                                                                                    GD_params,
                                                                                                                                    weights,
                                                                                                                                    biases,
                                                                                                                                    regularization_term=0.000001)
            print()
            for epoch, loss in enumerate(cost):
                print(f'Cross-entropy loss at epoch no.{epoch}: {loss}')
            print()

            print(f'Validation set accuracy: {ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights, best_biases, exponentials=[exponential_means, exponential_variances])}')

    def coarse_search():
        """
        First step of coarse search, where good values for eta derived from random_search are tested along
        with many tries for the amount of regularization.

        :return: None
        """

        accuracies = []
        etas = []
        lambdas = []

        for regularization_term in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:

            e_min = np.log(0.01)
            e_max = np.log(0.15)

            for _ in range(15):
                eta_term = np.random.rand(1, 1).flatten()[0]
                e = e_min + (e_max - e_min) * eta_term
                eta = np.exp(e)
                etas.append(eta)

                lambdas.append(regularization_term)

                GD_params = [100, eta, 10]

                weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

                best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(
                    X_training_1,
                    Y_training_1,
                    X_training_2,
                    Y_training_2,
                    y_training_2,
                    GD_params,
                    weights,
                    biases,
                    regularization_term)

                print('---------------------------------')
                print('Learning rate: ' + str(eta) + ', amount of regularization term: ' + str(regularization_term))
                accuracy_on_validation_set = ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights,
                                                                               best_biases, [exponential_means,
                                                                                             exponential_variances])
                accuracies.append(accuracy_on_validation_set)
                print('Accuracy performance on the validation set: ', accuracy_on_validation_set)

        sort_them_all = sorted(zip(accuracies, etas, lambdas))

        best_accuracies = [x for x, _, _ in sort_them_all]
        best_etas = [y for _, y, _ in sort_them_all]
        best_lambdas = [z for _, _, z in sort_them_all]

        print('---------------------------------')
        print('BEST PERFORMANCE: ', str(best_accuracies[-1]))
        print('Best eta: ', best_etas[-1])
        print('Best lambda: ', best_lambdas[-1])

        print('---------------------------------')
        print('SECOND BEST PERFORMANCE: ', str(best_accuracies[-2]))
        print('Second best eta: ', best_etas[-2])
        print('Second best lambda: ', best_lambdas[-2])

        print('---------------------------------')
        print('THIRD BEST PERFORMANCE: ', str(best_accuracies[-3]))
        print('Third best eta: ', best_etas[-3])
        print('Third best lambda: ', best_lambdas[-3])

    def fine_search():
        """
        Fine search based on the results of coarse search

        """

        accuracies = []
        etas = []
        lambdas = []

        # Lambda = 10^(-6)
        regularization_term = 1e-6

        e_min = np.log(0.027)
        e_max = np.log(0.06)

        for _ in range(20):
            eta_term = np.random.rand(1, 1).flatten()[0]
            e = e_min + (e_max - e_min) * eta_term
            eta = np.exp(e)
            etas.append(eta)

            lambdas.append(regularization_term)

            GD_params = [100, eta, 10]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(
                X_training_1,
                Y_training_1,
                X_training_2,
                Y_training_2,
                y_training_2,
                GD_params,
                weights,
                biases,
                regularization_term)

            print('---------------------------------')
            print('Learning rate: ' + str(eta) + ', amount of regularization term: ' + str(regularization_term))
            accuracy_on_validation_set = ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights,
                                                                           best_biases,
                                                                           [exponential_means, exponential_variances])
            accuracies.append(accuracy_on_validation_set)
            print('Accuracy performance on the validation set: ', accuracy_on_validation_set)

        # Lambda = 10^(-5)
        regularization_term = 1e-5

        e_min = np.log(0.005)
        e_max = np.log(0.015)

        for _ in range(20):
            eta_term = np.random.rand(1, 1).flatten()[0]
            e = e_min + (e_max - e_min) * eta_term
            eta = np.exp(e)
            etas.append(eta)

            lambdas.append(regularization_term)

            GD_params = [100, eta, 10]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(
                X_training_1,
                Y_training_1,
                X_training_2,
                Y_training_2,
                y_training_2,
                GD_params,
                weights,
                biases,
                regularization_term)

            print('---------------------------------')
            print('Learning rate: ' + str(eta) + ', amount of regularization term: ' + str(regularization_term))
            accuracy_on_validation_set = ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights,
                                                                           best_biases,
                                                                           [exponential_means, exponential_variances])
            accuracies.append(accuracy_on_validation_set)
            print('Accuracy performance on the validation set: ', accuracy_on_validation_set)

        e_min = np.log(0.06)
        e_max = np.log(0.07)

        for _ in range(20):
            eta_term = np.random.rand(1, 1).flatten()[0]
            e = e_min + (e_max - e_min) * eta_term
            eta = np.exp(e)
            etas.append(eta)

            lambdas.append(regularization_term)

            GD_params = [100, eta, 10]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(
                X_training_1,
                Y_training_1,
                X_training_2,
                Y_training_2,
                y_training_2,
                GD_params,
                weights,
                biases,
                regularization_term)

            print('---------------------------------')
            print('Learning rate: ' + str(eta) + ', amount of regularization term: ' + str(regularization_term))
            accuracy_on_validation_set = ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights,
                                                                           best_biases,
                                                                           [exponential_means, exponential_variances])
            accuracies.append(accuracy_on_validation_set)
            print('Accuracy performance on the validation set: ', accuracy_on_validation_set)

        # Lambda = 10^(-4)
        regularization_term = 1e-4

        e_min = np.log(0.01)
        e_max = np.log(0.02)

        for _ in range(10):
            eta_term = np.random.rand(1, 1).flatten()[0]
            e = e_min + (e_max - e_min) * eta_term
            eta = np.exp(e)
            etas.append(eta)

            lambdas.append(regularization_term)

            GD_params = [100, eta, 10]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(
                X_training_1,
                Y_training_1,
                X_training_2,
                Y_training_2,
                y_training_2,
                GD_params,
                weights,
                biases,
                regularization_term)

            print('---------------------------------')
            print('Learning rate: ' + str(eta) + ', amount of regularization term: ' + str(regularization_term))
            accuracy_on_validation_set = ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights,
                                                                           best_biases,
                                                                           [exponential_means, exponential_variances])
            accuracies.append(accuracy_on_validation_set)
            print('Accuracy performance on the validation set: ', accuracy_on_validation_set)

        # Lambda = 10^(-3)
        regularization_term = 1e-3

        e_min = np.log(0.005)
        e_max = np.log(0.015)

        for _ in range(10):
            eta_term = np.random.rand(1, 1).flatten()[0]
            e = e_min + (e_max - e_min) * eta_term
            eta = np.exp(e)
            etas.append(eta)

            lambdas.append(regularization_term)

            GD_params = [100, eta, 10]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(
                X_training_1,
                Y_training_1,
                X_training_2,
                Y_training_2,
                y_training_2,
                GD_params,
                weights,
                biases,
                regularization_term)

            print('---------------------------------')
            print('Learning rate: ' + str(eta) + ', amount of regularization term: ' + str(regularization_term))
            accuracy_on_validation_set = ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights,
                                                                           best_biases,
                                                                           [exponential_means, exponential_variances])
            accuracies.append(accuracy_on_validation_set)
            print('Accuracy performance on the validation set: ', accuracy_on_validation_set)

        e_min = np.log(0.09)
        e_max = np.log(0.11)

        for _ in range(10):
            eta_term = np.random.rand(1, 1).flatten()[0]
            e = e_min + (e_max - e_min) * eta_term
            eta = np.exp(e)
            etas.append(eta)

            lambdas.append(regularization_term)

            GD_params = [100, eta, 10]

            weights, biases = initialize_weights([[50, 3072], [30, 50], [10, 30]])

            best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(
                X_training_1,
                Y_training_1,
                X_training_2,
                Y_training_2,
                y_training_2,
                GD_params,
                weights,
                biases,
                regularization_term)

            print('---------------------------------')
            print('Learning rate: ' + str(eta) + ', amount of regularization term: ' + str(regularization_term))
            accuracy_on_validation_set = ComputeAccuracyBatchNormalization(X_training_2, y_training_2, best_weights,
                                                                           best_biases,
                                                                           [exponential_means, exponential_variances])
            accuracies.append(accuracy_on_validation_set)
            print('Accuracy performance on the validation set: ', accuracy_on_validation_set)

        sort_them_all = sorted(zip(accuracies, etas, lambdas))

        best_accuracies = [x for x, _, _ in sort_them_all]
        best_etas = [y for _, y, _ in sort_them_all]
        best_lambdas = [z for _, _, z in sort_them_all]

        print('---------------------------------')
        print('BEST PERFORMANCE: ', str(best_accuracies[-1]))
        print('Best eta: ', best_etas[-1])
        print('Best lambda: ', best_lambdas[-1])

        print('---------------------------------')
        print('SECOND BEST PERFORMANCE: ', str(best_accuracies[-2]))
        print('Second best eta: ', best_etas[-2])
        print('Second best lambda: ', best_lambdas[-2])

        print('---------------------------------')
        print('THIRD BEST PERFORMANCE: ', str(best_accuracies[-3]))
        print('Third best eta: ', best_etas[-3])
        print('Third best lambda: ', best_lambdas[-3])

    # compare_with_numerical_gradients()
    sanity()
    # random_search()
    # coarse_search()


if __name__ =='__main__':

    X_training_1, Y_training_1, y_training_1 = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_training_2, Y_training_2, y_training_2 = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, _, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    # exercise_1()
    # exercise_2()
    exercise_3()

    print('Finished!')