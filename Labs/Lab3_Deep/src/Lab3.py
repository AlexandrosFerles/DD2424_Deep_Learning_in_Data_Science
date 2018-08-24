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

    return np.maximum(x, 0)

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

def add_momentum(weights, grad_weights, momentum_weights, biases, grad_biases, momentum_biases, eta, momentum_term):
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

def BatchNormalize(s, mean_s, var_s, epsilon=1e-6):
    """
    Normalizes the scores of a batch based on their mean and variance.

    :param s: Scores evaluated as output of a layer of the network.
    :param mean_s: Mean of the scores.
    :param var_s: Variance of the scores.
    :param epsilon: A small number that is present to ensure that no division by zero will be performed.

    :return: The normalized scores,
    """

    temp = np.diag(np.power((var_s + epsilon), -0.5 ))
    diff = s- mean_s

    return temp * diff

def ForwardPassBatchNormalization(X, weights, biases, exponentials= None):
    """
    Evalueates the forward pass result of the classifier network using batch normalization.

    :param X: Input data.
    :param weights: Weight arrays of the k-layer network.
    :param biases: Bias vectors of the k-layer network.

    :return: Softmax probabilities (predictions) of the true labels of the data.
    """

    s = np.dot(weights[0], X) + biases[0]

    intermediate_outputs = [s]
    intermediate_activations = [ReLU(s)]

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
    # normalized_score = (s - mean_s) / np.sqrt(var_s)

    batch_normalization_outputs = [normalized_score]
    batch_normalization_activations = [ReLU(normalized_score)]

    for index in range(1, len(weights) - 1):
        s = np.dot(weights[index], batch_normalization_activations[-1]) + biases[index]

        intermediate_outputs.append(s)
        intermediate_activations.append(ReLU(s))

        if exponentials is None:
            mean_s = s.mean(axis=1).reshape(s.shape[0], 1)
            var_s = s.var(axis=1).reshape(s.shape[0], 1)

            means.append(mean_s)
            variances.append(var_s)

        else:

            mean_s = exponential_means[index]
            var_s = exponential_variances[index]

        normalized_score = BatchNormalize(s, mean_s, var_s)
        # normalized_score = (s - mean_s) / np.sqrt(var_s)

        batch_normalization_outputs.append(normalized_score)
        batch_normalization_activations.append(ReLU(normalized_score))

    s = np.dot(weights[-1], batch_normalization_activations[-1]) + biases[-1]
    p = softmax(s, axis=0)

    if exponentials is not None:
        return p
    else:
        return p, batch_normalization_activations, batch_normalization_outputs, means, variances, intermediate_activations, intermediate_outputs

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

    if exponentials is not None:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)
    else:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)[0]

    cross_entropy_loss = -np.log(np.diag(np.dot(Y.T, p))).sum() / float(X.shape[1])

    weight_sum = 0
    for weight in weights:

        weight_sum += np.power(weight, 2).sum()

    return cross_entropy_loss + regularization_term * weight_sum

def BatchNormBackPass(g, s, mean_s, var_s, epsilon=1e-5):

    # First part of the gradient:
    V_b = np.power(var_s+ epsilon, -0.5)
    part_1 = g * np.diag(V_b)

    # Second part pf the gradient
    diff = s - mean_s
    grad_J_vb = -0.5 * np.sum(g * np.power(var_s+ epsilon, -1.5) * diff, axis=1)
    grad_J_vb = np.expand_dims(grad_J_vb, axis=1)
    part_2 = (2/float(s.shape[1])) * grad_J_vb * diff

    # Third part of the gradient
    grad_J_mb = -np.sum(g* V_b, axis=1)
    grad_J_mb = np.expand_dims(grad_J_mb, axis=1)
    part_3 = grad_J_mb / float(s.shape[1])

    return part_1 + part_2 + part_3

def BackwardPassBatchNormalization(X, Y, weights, biases, p, intermediate_outputs, intermediate_activations, means, variances, regularization_term):

    # Back-propagate output layer at first

    weight_updates = []
    bias_updates = []

    g = p - Y

    bias_updates.append(g.sum(axis=1).reshape(biases[-1].shape))
    weight_updates.append(np.dot(g, intermediate_activations[-1].T))

    g = np.dot(g.T , weights[-1])
    ind = 1 * (intermediate_activations[-1] > 0)
    g = g.T  * ind

    for i in reversed(range(len(weights) -1)):
    # Back-propagate the gradient vector g to the layer before

        g = BatchNormBackPass(g, intermediate_outputs[i], means[i], variances[i])

        if i == 0:
            weight_updates.append(np.dot(g, X.T))
            bias_updates.append(np.sum(g, axis=1).reshape(biases[i].shape))
            break
        else:
            weight_updates.append(np.dot(g, intermediate_activations[i-1].T))

        bias_updates.append(np.sum(g, axis=1).reshape(biases[i].shape))

        g = np.dot(g.T , weights[i])
        ind = 1 * (intermediate_outputs[i-1] > 0)
        g = g.T * ind

    for elem in weight_updates:
        elem /= X.shape[1]

    for elem in bias_updates:
        elem /= X.shape[1]

    # Reverse the updates to match the order of the layers
    weight_updates = list(reversed(weight_updates)).copy()
    bias_updates = list(reversed(bias_updates)).copy()

    for index in range(len(weight_updates)):
        weight_updates[index] += 2*regularization_term * weight_updates[index]

    return weight_updates, bias_updates

def ExponentialMovingAverage(means, exponential_means, variances, exponential_variances, a=0.99):

    for index, elem in enumerate(exponential_means):

        exponential_means[index] = a * elem + (1-a) * means[index]
        exponential_variances[index] = a * exponential_variances[index] + (1-a) * variances[index]

    return exponential_means, exponential_variances

def MiniBatchGDBatchNormalization(X, Y, X_validation, Y_validation, y_validation, GDparams, weights, biases,
                                    regularization_term, momentum_term=0.9):
    """
    Performs mini batch-gradient descent computations with batch normalization.

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

    for epoch in tqdm(range(epoches)):
        # for epoch in range(epoches):

        if epoch ==1:
            print(f'End batch of previous layer was: {end}')

        for batch in range(1, int(X.shape[1] / number_of_mini_batches)):
            start = (batch - 1) * number_of_mini_batches
            end = min(batch * number_of_mini_batches + int(X.shape[1] / number_of_mini_batches), X.shape[1] )

            p, batch_norm_activations, batch_norm_outputs, means, variances, intermediate_activations, intermediate_outputs = ForwardPassBatchNormalization(X[:, start:end], weights, biases)

            grad_weights, grad_biases = BackwardPassBatchNormalization(X[:, start:end], Y[:, start:end], weights, biases, p, intermediate_outputs, intermediate_activations, means, variances, regularization_term)

            weights, biases, momentum_weights, momentum_biases = add_momentum(weights, grad_weights, momentum_weights, biases, grad_biases, momentum_biases, eta, momentum_term)

            if epoch == 0 and start == 0:
                exponential_means = means.copy()
                exponential_variances = variances.copy()
            else:
                exponential_means, exponential_variances = ExponentialMovingAverage(means, exponential_means, variances, exponential_variances)

        epoch_cost = ComputeCostBatchNormalization(X, Y, weights, biases, regularization_term, exponentials=[exponential_means, exponential_variances])
        if epoch_cost > 3 * original_training_cost:
            break
        val_epoch_cost = ComputeCostBatchNormalization(X_validation, Y_validation, weights, biases, regularization_term, exponentials=[exponential_means, exponential_variances])

        cost.append(epoch_cost)
        val_cost.append(val_epoch_cost)

        validation_set_accuracy = ComputeAccuracyBatchNormalization(X_validation, y_validation, weights, biases, exponentials=[exponential_means, exponential_variances])

        if validation_set_accuracy > best_validation_set_accuracy:

            best_weights = weights
            best_biases = biases
            best_validation_set_accuracy = validation_set_accuracy

        # Decay the learning rate
        eta *= 0.95

    print(f'Best validation set accuracy: {best_validation_set_accuracy}')
    return best_weights, best_biases, cost, val_cost, exponential_means, exponential_variances

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

    mean = np.mean(X_training_1)
    X_training_1 -= mean
    X_training_2 -= mean
    X_test -= mean

    # weights, biases = initialize_weights([[50, 3072], [30, 50], [10,30]])

    GD_params = [100, 0.0171384811847413, 20]

    weights, biases = initialize_weights([[50, 3072], [30, 50], [10,30]])

    weights, biases, training_cost, validation_cost, exponential_means, exponential_variances = MiniBatchGDBatchNormalization(X_training_1,
                                                                                                Y_training_1,
                                                                                                X_training_2,
                                                                                                Y_training_2,
                                                                                                y_training_2,
                                                                                                GD_params,
                                                                                                weights, biases,
                                                                                                regularization_term=0.0001)

    for i in range(len(training_cost)):

        print(f'Cost at training epoch {i+1} is {training_cost[i]}')

    test_set_accuracy = ComputeAccuracyBatchNormalization(X_test, y_test, weights, biases, exponentials = [exponential_means, exponential_variances])

    print(f'Test set accuracy performance: {test_set_accuracy}')





if __name__ =='__main__':

    # exercise_1()
    # exercise_2()
    exercise_3()

    print('Finished!')