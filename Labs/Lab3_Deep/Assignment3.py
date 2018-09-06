import numpy as np
import matplotlib.pyplot as plt
import pickle


def BatchNormBackPass(g, s, mean_s, var_s, epsilon=1e-10):

    # First part of the gradient:
    V_b = (var_s+ epsilon) ** (-0.5)
    part_1 = g * V_b

    # Second part pf the gradient
    diff = s - mean_s
    grad_J_vb = -0.5 * np.sum(g * (var_s+epsilon) ** (-1.5) * diff, axis=1)
    grad_J_vb = np.expand_dims(grad_J_vb, axis=1)
    part_2 = (2/float(s.shape[1])) * grad_J_vb * diff

    # Third part of the gradient
    grad_J_mb = -np.sum(g * V_b, axis=1)
    grad_J_mb = np.expand_dims(grad_J_mb, axis=1)
    part_3 = grad_J_mb / float(s.shape[1])

    return part_1 + part_2 + part_3

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
    # batch_normalization_activations = [ReLU(normalized_score)]
    batch_normalization_activations = [np.maximum(0, normalized_score)]


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
        # batch_normalization_activations.append(ReLU(normalized_score))
        batch_normalization_activations = [np.maximum(0, normalized_score)]

    s = np.dot(weights[-1], batch_normalization_activations[-1]) + biases[-1]

    # p = softmax(s, axis=0)
    p = np.exp(s) / np.sum(np.exp(s), axis=0)

    if exponentials is not None:
        return p
    else:
        return p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances

def ComputeCostBatchNormalization(X, Y, weights, biases, regularization_term, exponentials=None):
    """
    Computes the cross-entropy loss on a batch of data.

    :param X: Input data
    :param y: Labels of the ground truth
    :param weights: Weights arrays of the k layers
    :param biases: Bias vectors of the k layers
    :param regularization_term: Amount of regularization applied.
    :param exponentials: (Optional) Contains the exponential means and variances computed, they are used in call after training.

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

def BackwardPassBatchNormalization(X, Y, weights, biases, p, bn_outputs, bn_activations, intermediate_outputs, means, variances, regularization_term):

    # Back-propagate output layer at first

    g = p - Y

    bias_updates = [g.sum(axis=1).reshape(biases[-1].shape)]
    weight_updates = [np.dot(g, bn_activations[-1].T)]

    g = np.dot(g.T, weights[-1])
    ind = 1 * (bn_outputs[-1] > 0)
    g = g.T * ind

    for i in reversed(range(len(weights) -1)):
    # Back-propagate the gradient vector g to the layer before

        g = BatchNormBackPass(g, intermediate_outputs[i], means[i], variances[i], 1e-10)

        if i == 0:
            weight_updates.append(np.dot(g, X.T))
            bias_updates.append(np.sum(g, axis=1).reshape(biases[i].shape))
            break
        else:
            weight_updates.append(np.dot(g, bn_activations[i-1].T))
            bias_updates.append(np.sum(g, axis=1).reshape(biases[i].shape))

        g = np.dot(g.T, weights[i])
        ind = 1 * (bn_outputs[i-1] > 0)
        g = g.T * ind


    for elem in weight_updates:
        elem /= X.shape[1]

    for elem in bias_updates:
        elem /= X.shape[1]

    # Reverse the updates to match the order of the layers
    weight_updates = list(reversed(weight_updates)).copy()
    bias_updates = list(reversed(bias_updates)).copy()

    for index in range(len(weight_updates)):
        weight_updates[index] += 2*regularization_term * weights[index]

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

        for i in range(b.shape[0]):
            b_try = np.copy(b)
            b_try[i, 0] -= h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c1 = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases, regularization_term=0)
            # c1 = BatchComputeCost(X, Y, weights, temp_biases, reg=0)
            b_try = np.copy(b)
            b_try[i, 0] += h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c2 = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases, regularization_term=0)
            # c2 = BatchComputeCost(X, Y, weights, temp_biases, reg=0)


            grad_b[i, 0] = (c2 - c1) / (2 * h)

        grad_biases.append(grad_b)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i, j] -= h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c1 = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases, regularization_term=0)
                # c1 = BatchComputeCost(X, Y, temp_weights, biases, reg=0)

                W_try = np.copy(W)
                W_try[i, j] += h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c2 = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases, regularization_term=0)
                # c2 = BatchComputeCost(X, Y, temp_weights, biases, reg=0)

                grad_W[i, j] = (c2 - c1) / (2 * h)

        grad_weights.append(grad_W)

    return grad_weights, grad_biases

def batch_norm_back_pass(g, s, mu, var):
    s_mu = s - mu

    grad_var = - 1 / 2 * (g * (var ** (-3 / 2)) * s_mu).sum(axis=1)
    grad_mu = - (g * (var ** (-1 / 2))).sum(axis=1)

    grad_var = np.expand_dims(grad_var, 1)
    grad_mu = np.expand_dims(grad_mu, 1)
    grad_s = g * (var ** (-1 / 2)) + (2 / s.shape[1]) * grad_var * s_mu + grad_mu / s.shape[1]
    return grad_s


''' FERLES '''


def LoadBatch(filename):

    with open(filename, 'rb') as fo:
        D = pickle.load(fo, encoding='bytes')

    X = D[b'data'].T / 255
    Y = np.zeros((10, X.shape[1]))
    y = D[b'labels']

    for i in range(len(y)):
        Y[y[i], i] = 1

    return X, Y, y

def initialize(dim, nodes, layers):

    # np.random.seed(879)
    np.random.seed(627)


    if layers == 1:

        W = [np.random.normal(loc=0, scale=0.01, size=(nodes[-1], dim))]
        b = [np.random.normal(loc=0, scale=0.01, size=(nodes[-1], 1))]

    if layers > 1:

        W = [np.random.normal(loc=0, scale=0.01, size=(nodes[0], dim))]
        b = [np.random.normal(loc=0, scale=0.01, size=(nodes[0], 1))]

        for i in range(1, layers):

            W.append(np.random.normal(loc=0, scale=0.01, size=(nodes[i], nodes[i - 1])))
            b.append(np.random.normal(loc=0, scale=0.01, size=(nodes[i], 1)))

    return W, b

def BatchNormalise(s, mean, var, eps=1e-10):

    Vb = var + eps

    A = np.sqrt(Vb)

    B = s - mean

    return B / A

def BatchNormBackPass1(X, g, s, mean, var):

    # Vb = np.diagflat(v + 1e-10)

    # A = np.dot(g, fractional_matrix_power(Vb, -1/2))
    #
    # B = (2 / X.shape[1]) * (-1/2 * np.sum(np.dot(np.dot(g, fractional_matrix_power(Vb, -3/2)), np.diag(s - k))))
    #
    # C = (1 / X.shape[1]) * (- np.sum(np.dot(g, fractional_matrix_power(Vb, -1/2))))

    # Vb = np.diagflat(var + 1e-10)

    Vb = (var + 1e-10)

    A = g * (Vb ** (-1/2))

    B = (-1 / X.shape[1]) * (np.sum((g * (Vb ** (-3/2))) * np.diag(s - mean), axis=1))

    C = (-1 / X.shape[1]) * (np.sum(A, axis=1))

    return A + B + C

def EvaluateClassifier(x, W, b):

    S = []
    H = [np.copy(x)]

    for i in range(len(W) - 1):

        s = np.dot(W[i], H[i]) + b[i]
        H.append(np.maximum(0, s))
        S.append(s)

    s = np.dot(W[-1], H[-1]) + b[-1]
    P = np.exp(s) / np.sum(np.exp(s), axis=0)

    return P, S, H

def forward_pass(x, W, b, EMA=None):

    S = []
    Mean = []
    Var = []
    H = [np.copy(x)]
    S_bn = []

    if EMA is None:

        for i in range(len(W) - 1):

            s = np.dot(W[i], H[i]) + b[i]
            S.append(s)

            mean = np.mean(s, axis=1).reshape(s.shape[0], 1)
            Mean.append(mean)
            var = np.var(s, axis=1).reshape(s.shape[0], 1)
            Var.append(var)

            s_bn = BatchNormalise(s, mean, var)
            S_bn.append(s_bn)

            H.append(np.maximum(0, s_bn))

        s = np.dot(W[-1], H[-1]) + b[-1]
        P = np.exp(s) / np.sum(np.exp(s), axis=0)

        return P, S, S_bn, H, Mean, Var

    else:

        Mean = np.copy(EMA[0])
        Var = np.copy(EMA[1])

        for i in range(len(W) - 1):

            s = np.dot(W[i], H[i]) + b[i]
            S.append(s)

            s_bn = BatchNormalise(s, Mean[i], Var[i])
            S_bn.append(s_bn)

            H.append(np.maximum(0, s_bn))

        s = np.dot(W[-1], H[-1]) + b[-1]
        P = np.exp(s) / np.sum(np.exp(s), axis=0)

        # return P, S, S_bn, H, Mean, Var
        return P

def ComputeCost(X, Y, W, b, reg):

    P, S, H = EvaluateClassifier(X, W, b)

    lcross = 0
    temp = 0

    for input in range(X.shape[1]):

        lcross -= np.log(np.dot(np.transpose(Y[:, input]), P[:, input]))

    for i in range(len(W)):

        temp += np.sum(np.power(W[i], 2))

    J = (1. / float(X.shape[1])) * lcross + reg * temp

    return J

def BatchComputeCost(X, Y, W, b, reg, EMA=None):

    P = forward_pass(X, W, b, EMA)[0]

    lcross = 0
    temp = 0

    for input in range(X.shape[1]):

        lcross -= np.log(np.dot(np.transpose(Y[:, input]), P[:, input]))

    for i in range(len(W)):

        temp += np.sum(np.power(W[i], 2))

    J = (1. / float(X.shape[1])) * lcross + reg * temp

    return J

def ComputeAccuracy(X, y, W, b):

    P, S, H = EvaluateClassifier(X, W, b)

    count = 0

    for k in range(X.shape[1]):

        k_opt = np.argmax(P[:, k])

        if k_opt == y[k]:

            count += 1

    return count / X.shape[1]

def BatchComputeAccuracy(X, y, W, b, EMA=None):

    P = forward_pass(X, W, b, EMA)[0]

    count = 0

    for k in range(X.shape[1]):

        k_opt = np.argmax(P[:, k])

        if k_opt == y[k]:

            count += 1

    return count / X.shape[1]

def ComputeGradsNumSlow(X, Y, W, b, reg, h):

    grad_W = []
    grad_b = []

    for k in range(len(W)):

        grad_W.append(np.zeros((W[k].shape[0], W[k].shape[1])))
        grad_b.append(np.zeros((b[k].shape[0], b[k].shape[1])))

    for k in range(len(b)):

        b_new = np.copy(b)

        for i in range(len(b[k])):

            b_try = np.copy(b[k])
            b_try[i, 0] -= h
            b_new[k] = b_try
            c1 = ComputeCost(X, Y, W, b_new, reg)

            b_try = np.copy(b[k])
            b_try[i, 0] += h
            b_new[k] = b_try
            c2 = ComputeCost(X, Y, W, b_new, reg)

            grad_b[k][i, 0] = (c2 - c1) / (2 * h)


    for k in range(len(W)):

        for i in range(W[k].shape[0]):
            for j in range(W[k].shape[1]):

                W_new = np.copy(W)
                W_try = np.copy(W[k])
                W_try[i, j] -= h
                W_new[k] = W_try
                c1 = ComputeCost(X, Y, W_new, b, reg)

                W_try = np.copy(W[k])
                W_try[i, j] += h
                W_new[k] = W_try
                c2 = ComputeCost(X, Y, W_new, b, reg)

                grad_W[k][i, j] = (c2 - c1) / (2 * h)

    return grad_W, grad_b

def BatchComputeGradsNumSlow(X, Y, W, b, reg, h):

    grad_W = []
    grad_b = []

    for k in range(len(W)):

        grad_W.append(np.zeros((W[k].shape[0], W[k].shape[1])))
        grad_b.append(np.zeros((b[k].shape[0], b[k].shape[1])))

    for k in range(len(b)):

        b_new = np.copy(b)

        for i in range(len(b[k])):

            b_try = np.copy(b[k])
            b_try[i, 0] -= h
            b_new[k] = b_try
            c1 = BatchComputeCost(X, Y, W, b_new, reg)

            b_try = np.copy(b[k])
            b_try[i, 0] += h
            b_new[k] = b_try
            c2 = BatchComputeCost(X, Y, W, b_new, reg)

            grad_b[k][i, 0] = (c2 - c1) / (2 * h)

    for k in range(len(W)):

        for i in range(W[k].shape[0]):
            for j in range(W[k].shape[1]):

                W_new = np.copy(W)
                W_try = np.copy(W[k])
                W_try[i, j] -= h
                W_new[k] = W_try
                c1 = BatchComputeCost(X, Y, W_new, b, reg)

                W_try = np.copy(W[k])
                W_try[i, j] += h
                W_new[k] = W_try
                c2 = BatchComputeCost(X, Y, W_new, b, reg)

                grad_W[k][i, j] = (c2 - c1) / (2 * h)

    return grad_W, grad_b

def ComputeGradients(X, Y, P, S, H, W, b, reg):

    grad_LW = []
    grad_Lb = []

    for i in range(len(W)):

        grad_Lb.append(np.zeros((b[i].shape[0], b[i].shape[1])))
        grad_LW.append(np.zeros((W[i].shape[0], W[i].shape[1])))

    for input in range(X.shape[1]):

        g = -np.transpose((Y[:, input] - P[:, input]))

        for k in range(len(W) - 1, -1, -1):

            grad_Lb[k] += np.transpose(g).reshape(b[k].shape[0], 1)

            grad_LW[k] += np.dot(np.transpose(g).reshape(b[k].shape[0], 1), np.reshape(H[k][:, input], (1, H[k].shape[0])))

            if k >= 1:

                g = np.dot(np.transpose(g).reshape(1, W[k].shape[0]), W[k])
                dh = np.diagflat(1 * (S[k - 1][:, input] > 0))
                g = np.dot(g, dh)

    for i in range(len(grad_LW)):

        grad_LW[i] /= X.shape[1]
        grad_Lb[i] /= X.shape[1]
        grad_LW[i] += 2 * reg * W[i]

    return grad_LW, grad_Lb

def backward_pass(X, Y, P, S, S_bn, H, W, b, mean, var, reg):

    grad_LW = []
    grad_Lb = []

    for i in range(len(W)):

        grad_Lb.append(np.zeros((b[i].shape[0], b[i].shape[1])))
        grad_LW.append(np.zeros((W[i].shape[0], W[i].shape[1])))

    g = -np.transpose(Y - P)

    grad_Lb[-1] = np.sum(np.transpose(g), axis=1).reshape(b[-1].shape)
    grad_LW[-1] = np.dot(np.transpose(g), np.transpose(H[-1]))

    g = np.dot(g, W[-1])
    dh = 1 * (S_bn[-1] > 0)
    g = np.multiply(np.transpose(g), dh)

    for k in range(len(W) - 2, -1, -1):

        # g = BatchNormBackPass(X, g, S, mean[k], var[k])
        g = batch_norm_back_pass(g, S[k], mean[k], var[k])

        grad_Lb[k] = np.sum(g, axis=1).reshape(b[k].shape)
        grad_LW[k] = np.dot(g, np.transpose(H[k]))

        if k >= 1:

            g = np.dot(np.transpose(g), W[k])
            dh = 1 * (S_bn[k - 1] > 0)
            g = np.multiply(np.transpose(g), dh)

    for i in range(len(grad_LW)):

        grad_LW[i] /= X.shape[1]
        grad_Lb[i] /= X.shape[1]
        grad_LW[i] += 2 * reg * W[i]

    return grad_LW, grad_Lb

def check(ga, gn, eps):

    for grad in range(len(ga)):

        nom = np.linalg.norm(ga[grad] - gn[grad])

        denom = np.amax(np.linalg.norm(ga[grad]) + np.linalg.norm(gn[grad]))

        if eps > denom: RE = nom / eps

        else: RE = nom / denom

        if RE <= eps: print("Correct", RE)

        else: print("Fail", RE)

def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, W, b, reg):

    train_cost = [ComputeCost(X, Y, W, b, reg)]
    val_cost = [ComputeCost(X, Y, W, b, reg)]

    for epoch in range(GDparams[2]):

        for batch in range(1, int(X.shape[1] / GDparams[0])):

            batch_start = (batch - 1) * GDparams[0] + 1
            batch_end = batch * GDparams[0] + 1
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]

            P, S, H = EvaluateClassifier(Xbatch, W, b)

            grad_LW, grad_Lb = ComputeGradients(Xbatch, Ybatch, P, S, H, W, b, reg=0)

            for i in range(len(W)):

                W[i] -= GDparams[1] * grad_LW[i]
                b[i] -= GDparams[1] * grad_Lb[i]

        train_cost.append(ComputeCost(X, Y, W, b, reg))
        print(" Training loss at epoch", epoch + 1, "is:", train_cost[-1])
        val_cost.append(ComputeCost(X_val, Y_val, W, b, reg))
        print(" Val loss at epoch", epoch + 1, "is:", val_cost[-1], "\n")

    plt.title('cost function per epoch')
    plt.plot(train_cost, 'g', label='train cost')
    plt.plot(val_cost, 'r', label='val cost')
    plt.legend(loc='upper right')
    plt.savefig("Cost loss, lambda: "+str(reg)+", eta:"+str(GDparams[1])+".png")
    plt.show()
    plt.clf()

    return W, b

def momentum(X, Y, y, X_val, Y_val, y_val, GDparams, W, b, reg, rho):

    train_cost = [ComputeCost(X, Y, W, b, reg)]
    val_cost = [ComputeCost(X_val, Y_val, W, b, reg)]

    v_W = []
    v_b = []

    for v in range(len(W)):

        v_W.append(np.zeros((W[v].shape[0], W[v].shape[1])))
        v_b.append(np.zeros((b[v].shape[0], b[v].shape[1])))

    for epoch in range(GDparams[2]):

        if epoch > 0:

            GDparams[1] *= 0.95  # eta decay

            if train_cost[-1] > 3 * train_cost[0]: break

        for batch in range(1, int(X.shape[1] / GDparams[0])):
            batch_start = (batch - 1) * GDparams[0] + 1
            batch_end = batch * GDparams[0] + 1
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]

            P, S, H = EvaluateClassifier(Xbatch, W, b)

            grad_LW, grad_Lb = ComputeGradients(Xbatch, Ybatch, P, S, H, W, b, reg=0)

            for i in range(len(v_b)):

                v_W[i] = rho * v_W[i] + GDparams[1] * grad_LW[i]
                v_b[i] = rho * v_b[i] + GDparams[1] * grad_Lb[i]

            for i in range(len(W)):

                W[i] -= v_W[i]
                b[i] -= v_b[i]

        train_cost.append(ComputeCost(X, Y, W, b, reg))
        print(" Training loss at epoch", epoch + 1, "is:", train_cost[-1])
        val_cost.append(ComputeCost(X_val, Y_val, W, b, reg))
        print(" Val loss at epoch", epoch + 1, "is:", val_cost[-1], "\n")

    plt.title('cost function per epoch')
    plt.plot(train_cost, 'g', label='train cost')
    plt.plot(val_cost, 'r', label='val cost')
    plt.legend(loc='upper right')
    plt.savefig("rs_reg_" + str(reg) + "_eta_" + str(GDparams[1]) + ".png")
    plt.show()
    plt.clf()

    return W, b

def Batch_Normalisation(X, Y, X_val, Y_val, GDparams, W, b, reg, rho):

    train_cost = [BatchComputeCost(X, Y, W, b, reg)]
    val_cost = [BatchComputeCost(X, Y, W, b, reg)]
    EMA, v_W, v_b = [], [], []

    for v in range(len(W)):

        v_W.append(np.zeros((W[v].shape[0], W[v].shape[1])))
        v_b.append(np.zeros((b[v].shape[0], b[v].shape[1])))

    for epoch in range(GDparams[2]):

        if epoch > 0:

            GDparams[1] *= 0.95  # eta decay

            if train_cost[-1] > 3 * train_cost[0]: break

        for batch in range(1, int(X.shape[1] / GDparams[0])):

            batch_start = (batch - 1) * GDparams[0] + 1
            batch_end = batch * GDparams[0] + 1
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]

            if (epoch == 0) and (batch == 1):

                P, S, S_bn, H, mean, var = forward_pass(Xbatch, W, b)

                EMA.append(mean)
                EMA.append(var)

                for l in range(len(W) - 1):

                    EMA[0][l] = 0.99 * EMA[0][l] + (1 - 0.99) * mean[l]
                    EMA[1][l] = 0.99 * EMA[1][l] + (1 - 0.99) * var[l]

                grad_LW, grad_Lb = backward_pass(Xbatch, Ybatch, P, S, S_bn, H, W, b, mean, var, reg)

            else:

                P, S, S_bn, H, mean, var = forward_pass(Xbatch, W, b)

                for l in range(len(W) - 1):

                    EMA[0][l] = 0.99 * EMA[0][l] + (1 - 0.99) * mean[l]
                    EMA[1][l] = 0.99 * EMA[1][l] + (1 - 0.99) * var[l]

                grad_LW, grad_Lb = backward_pass(Xbatch, Ybatch, P, S, S_bn, H, W, b, mean, var, reg)

            for i in range(len(v_b)):

                v_W[i] = rho * v_W[i] + GDparams[1] * grad_LW[i]
                v_b[i] = rho * v_b[i] + GDparams[1] * grad_Lb[i]

            for i in range(len(W)):

                W[i] -= v_W[i]
                b[i] -= v_b[i]

        train_cost.append(BatchComputeCost(X, Y, W, b, reg))
        print(" Training loss at epoch", epoch + 1, "is:", train_cost[-1])
        val_cost.append(BatchComputeCost(X_val, Y_val, W, b, reg))
        print(" Val loss at epoch", epoch + 1, "is:", val_cost[-1], "\n")

    plt.title('cost function per epoch')
    plt.plot(train_cost, 'g', label='train cost')
    plt.plot(val_cost, 'r', label='val cost')
    plt.legend(loc='upper right')
    plt.savefig("Cost loss, lambda: "+str(reg)+", eta:"+str(GDparams[1])+".png")
    plt.show()
    plt.clf()

    return W, b

def grad_test(test):

    X_train, Y_train_hot, y_train = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
    mean_X_train = np.mean(X_train, axis=0)
    X_train -= mean_X_train

    if test == 'check1':

        print("CHECK 1", "\n")

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

        # P, S, H = EvaluateClassifier(X_train, W, b)

        P, S, H, ss, mean, var = forward_pass(X_train, W, b)


        gLW, gLb = ComputeGradients(X_train, Y_train_hot, P, S, H, W, b, reg=0)

        grad_W, grad_b = ComputeGradsNumSlow(X_train, Y_train_hot, W, b, 0, 1e-5)

        check(gLW, grad_W, 1e-6)
        check(gLb, grad_b, 1e-6)
        print("\n")

    elif test == 'check2':

        print("CHECK 2", "\n")

        X_train = X_train[:, 0:1]
        Y_train_hot = Y_train_hot[:, 0:1]

        W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)


        P, S, H = EvaluateClassifier(X_train, W, b)

        gLW, gLb = ComputeGradients(X_train, Y_train_hot, P, S, H, W, b, reg=0)

        grad_W, grad_b = ComputeGradsNumSlow(X_train, Y_train_hot, W, b, 0, 1e-5)

        check(gLW, grad_W, 1e-6)
        check(gLb, grad_b, 1e-6)
        print("\n")

    elif test == 'check3':

        print("CHECK 3", "\n")

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

        P, S, H = EvaluateClassifier(X_train, W, b)

        gLW, gLb = ComputeGradients(X_train, Y_train_hot, P, S, H, W, b, reg=0.1)

        grad_W, grad_b = ComputeGradsNumSlow(X_train, Y_train_hot, W, b, 0.1, 1e-5)

        check(gLW, grad_W, 1e-6)
        check(gLb, grad_b, 1e-6)
        print("\n")

    elif test == 'check4':

        print("CHECK 4", "\n")

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

        P, S, H = EvaluateClassifier(X_train, W, b)

        gLW, gLb = ComputeGradients(X_train, Y_train_hot, P, S, H, W, b, reg=1)

        grad_W, grad_b = ComputeGradsNumSlow(X_train, Y_train_hot, W, b, 1, 1e-5)

        check(gLW, grad_W, 1e-6)
        check(gLb, grad_b, 1e-6)
        print("\n")

def batch_grad_test(test):

    X_train, Y_train_hot, y_train = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
    mean_X_train = np.mean(X_train, axis=0)
    X_train -= mean_X_train

    if test == 'check1':

        print("CHECK 1", "\n")

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

        P, S, S_bn, H, mean, var = forward_pass(X_train, W, b)

        gLW, gLb = backward_pass(X_train, Y_train_hot, P, S, S_bn, H, W, b, mean, var, reg=0)

        grad_W, grad_b = BatchComputeGradsNumSlow(X_train, Y_train_hot, W, b, reg=0, h=1e-5)

        check(gLW, grad_W, 1e-6)
        check(gLb, grad_b, 1e-6)
        print("\n")

    elif test == 'check3':

        print("CHECK 3", "\n")

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

        P, S, S_bn, H, mean, var = forward_pass(X_train, W, b)

        gLW, gLb = backward_pass(X_train, Y_train_hot, P, S, S_bn, H, W, b, mean, var, reg=0.1)

        grad_W, grad_b = BatchComputeGradsNumSlow(X_train, Y_train_hot, W, b, 0.1, 1e-5)

        check(gLW, grad_W, 1e-6)
        check(gLb, grad_b, 1e-6)
        print("\n")

    elif test == 'check4':

        print("CHECK 4", "\n")

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
        # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

        P, S, S_bn, H, mean, var = forward_pass(X_train, W, b)

        gLW, gLb = backward_pass(X_train, Y_train_hot, P, S, S_bn, H, W, b, mean, var, reg=1)

        grad_W, grad_b = BatchComputeGradsNumSlow(X_train, Y_train_hot, W, b, 1, 1e-5)

        check(gLW, grad_W, 1e-6)
        check(gLb, grad_b, 1e-6)
        print("\n")

def sanity1():

    X_train, Y_train_hot, y_train = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('Datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('Datasets/cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
    # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
    # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

    X_train = X_train[:, :1000]
    Y_train_hot = Y_train_hot[:, :1000]
    y_train = y_train[:1000]

    X_val = X_val[:, :1000]
    Y_val_hot = Y_val_hot[:, :1000]
    y_val = y_val[:1000]

    X_test = X_test[:, :1000]
    y_test = y_test[:1000]

    W, b = MiniBatchGD(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, 0.08, 200], W, b, reg=0)

    print("Accuracy on the train set:", 100 * ComputeAccuracy(X_train, y_train, W, b), "%")
    print("Accuracy on the val set:", 100 * ComputeAccuracy(X_val, y_val, W, b), "%")
    print("Accuracy on the test set:", 100 * ComputeAccuracy(X_test, y_test, W, b), "%")

def sanity2():

    X_train, Y_train_hot, y_train = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('Datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('Datasets/cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    X_train = X_train[:, :1000]
    Y_train_hot = Y_train_hot[:, :1000]
    y_train = y_train[:1000]

    X_val = X_val[:, :1000]
    Y_val_hot = Y_val_hot[:, :1000]
    y_val = y_val[:1000]

    X_test = X_test[:, :1000]
    y_test = y_test[:1000]

    W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
    # W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
    # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

    W, b = momentum(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, 0.08, 200], W, b, reg=0, rho=0.9)

    print("Accuracy on the train set:", 100 * ComputeAccuracy(X_train, y_train, W, b), "%")
    print("Accuracy on the val set:", 100 * ComputeAccuracy(X_val, y_val, W, b), "%")
    print("Accuracy on the test set:", 100 * ComputeAccuracy(X_test, y_test, W, b), "%")

def sanity():

    X_train, Y_train_hot, y_train = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('Datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('Datasets/cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    X_train = X_train[:, :1000]
    Y_train_hot = Y_train_hot[:, :1000]
    y_train = y_train[:1000]

    X_val = X_val[:, :1000]
    Y_val_hot = Y_val_hot[:, :1000]
    y_val = y_val[:1000]

    X_test = X_test[:, :1000]
    y_test = y_test[:1000]

    # W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
    W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
    # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

    W, b = Batch_Normalisation(X_train, Y_train_hot, X_val, Y_val_hot, [100, 0.08, 200], W, b, reg=0, rho=0.9)

    print("Accuracy on the train set:", 100 * BatchComputeAccuracy(X_train, y_train, W, b), "%")
    print("Accuracy on the val set:", 100 * BatchComputeAccuracy(X_val, y_val, W, b), "%")
    print("Accuracy on the test set:", 100 * BatchComputeAccuracy(X_test, y_test, W, b), "%")

def test():

    X_train, Y_train_hot, y_train = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('Datasets/cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('Datasets/cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    # W, b = initialize(X_train.shape[0], nodes=[50, Y_train_hot.shape[0]], layers=2)
    W, b = initialize(X_train.shape[0], nodes=[50, 30, Y_train_hot.shape[0]], layers=3)
    # W, b = initialize(X_train.shape[0], nodes=[50, 30, 30, Y_train_hot.shape[0]], layers=4)

    W, b = Batch_Normalisation(X_train, Y_train_hot, X_val, Y_val_hot, [100, 0.1, 5], W, b, reg=0, rho=0.9)

    print("Accuracy on the train set:", 100 * BatchComputeAccuracy(X_train, y_train, W, b), "%")
    print("Accuracy on the val set:", 100 * BatchComputeAccuracy(X_val, y_val, W, b), "%")
    print("Accuracy on the test set:", 100 * BatchComputeAccuracy(X_test, y_test, W, b), "%")


def main():

    ''' Gradient descent check '''

    # grad_test('check1')
    # grad_test('check2')
    # grad_test('check3')
    # grad_test('check4')

    ''' Sanity checks for classical vanilla mini-batch and momentum version of the algorithm '''

    # sanity1()
    # sanity2()

    ''' Gradient descent check with batch normalisation '''

    # batch_grad_test('check1')
    # batch_grad_test('check3')
    # batch_grad_test('check4')

    ''' Sanity check for Batch Normalisation '''

    # sanity()

    test()

if __name__ == "__main__":

    main()



