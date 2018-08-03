import numpy as np
import matplotlib.pyplot as plt
import pickle


def LoadBatch(filename):

    with open(filename, 'rb') as fo:
        D = pickle.load(fo, encoding='bytes')

    X = D[b'data'].T / 255
    Y = np.zeros((10, X.shape[1]))
    y = D[b'labels']

    for i in range(len(y)):
        Y[y[i], i] = 1

    return X, Y, y

def initialize(K, d, m):

    np.random.seed(600)

    W1 = np.random.normal(loc=0, scale=0.001, size=(m, d))
    W2 = np.random.normal(loc=0, scale=0.001, size=(K, m))

    b1 = np.random.normal(loc=0, scale=0.001, size=(m, 1))
    b2 = np.random.normal(loc=0, scale=0.001, size=(K, 1))

    return W1, W2, b1, b2

def EvaluateClassifier(x, W1, W2, b1, b2):

    S1 = np.dot(W1, x) + b1
    h = np.maximum(0, S1)
    S = np.dot(W2, h) + b2
    P = np.exp(S) / np.sum(np.exp(S), axis=0)

    return P, S1, h

def ComputeCost(X, Y, W1, W2, b1, b2, reg):

    P, S1, h = EvaluateClassifier(X, W1, W2, b1, b2)

    lcross = 0

    for input in range(X.shape[1]):

        lcross -= np.log(np.dot(np.transpose(Y[:, input]), P[:, input]))

    J = (1. / float(X.shape[1])) * lcross + reg * (np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2)))

    return J

def ComputeAccuracy(X, y, W1, W2, b1, b2):

    P, S1, h = EvaluateClassifier(X, W1, W2, b1, b2)

    count = 0

    for k in range(X.shape[1]):

        k_opt = np.argmax(P[:, k])

        if k_opt == y[k]:

            count += 1

    return count / X.shape[1]

def ComputeGradsNumSlow(X, Y, W1, W2, b1, b2, reg, h):

    grad_W1 = np.zeros((W1.shape[0], W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0], W2.shape[1]))

    grad_b1 = np.zeros((b1.shape[0], b1.shape[1]))
    grad_b2 = np.zeros((b2.shape[0], b2.shape[1]))


    for i in range(len(b1)):

        b1_try = np.copy(b1)
        b1_try[i, 0] -= h
        c1 = ComputeCost(X, Y, W1, W2, b1_try, b2, reg)
        b1_try = np.copy(b1)
        b1_try[i, 0] += h
        c2 = ComputeCost(X, Y, W1, W2, b1_try, b2, reg)
        grad_b1[i, 0] = (c2 - c1) / (2*h)

    for i in range(len(b2)):

        b2_try = np.copy(b2)
        b2_try[i, 0] -= h
        c1 = ComputeCost(X, Y, W1, W2, b1, b2_try, reg)
        b2_try = np.copy(b2)
        b2_try[i, 0] += h
        c2 = ComputeCost(X, Y, W1, W2, b1, b2_try, reg)
        grad_b2[i, 0] = (c2 - c1) / (2*h)


    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):

            W1_try = np.copy(W1)
            W1_try[i, j] -= h
            c1 = ComputeCost(X, Y, W1_try, W2, b1, b2, reg)

            W1_try = np.copy(W1)
            W1_try[i, j] += h
            c2 = ComputeCost(X, Y, W1_try, W2, b1, b2, reg)

            grad_W1[i, j] = (c2 - c1) / (2*h)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):

            W2_try = np.copy(W2)
            W2_try[i, j] -= h
            c1 = ComputeCost(X, Y, W1, W2_try, b1, b2, reg)

            W2_try = np.copy(W2)
            W2_try[i, j] += h
            c2 = ComputeCost(X, Y, W1, W2_try, b1, b2, reg)

            grad_W2[i, j] = (c2 - c1) / (2 * h)


    return grad_W1, grad_W2, grad_b1, grad_b2

def ComputeGradients(X, Y, P, S1, h, W1, W2, b1, b2, reg):

    grad_LW1 = np.zeros((W1.shape[0], W1.shape[1]))
    grad_LW2 = np.zeros((W2.shape[0], W2.shape[1]))

    grad_Lb1 = np.zeros((b1.shape[0], b1.shape[1]))
    grad_Lb2 = np.zeros((b2.shape[0], b2.shape[1]))

    for input in range(X.shape[1]):

        g = -np.transpose((Y[:, input] - P[:, input]))

        grad_LW2 += np.dot(np.transpose(g).reshape(10, 1), np.reshape(h[:, input], (1, h.shape[0])))
        grad_Lb2 += np.transpose(g).reshape(10, 1)

        g = np.dot(np.transpose(g).reshape(1, 10), W2)
        dh = np.diagflat(1 * (S1[:, input] > 0))
        g = np.dot(g, dh)

        grad_Lb1 += np.transpose(g)
        grad_LW1 += np.dot(np.transpose(g), np.transpose(X[:, input]).reshape(1, X.shape[0]))

    grad_LW1 /= X.shape[1]
    grad_Lb1 /= X.shape[1]
    grad_LW2 /= X.shape[1]
    grad_Lb2 /= X.shape[1]

    return grad_LW1 + 2 * reg * W1, grad_LW2 + 2 * reg * W2, grad_Lb1, grad_Lb2

def check(ga, gn, eps):

    nom = np.linalg.norm(ga - gn)

    denom = np.amax(np.linalg.norm(ga) + np.linalg.norm(gn))

    if eps > denom: RE = nom / eps

    else: RE = nom / denom

    if RE <= eps: print("Correct", RE)

    else: print("Fail", RE)

def MiniBatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, W1, W2, b1, b2, reg):

    train_cost = []
    val_cost = []

    for epoch in range(GDparams[2]):

        for batch in range(1, int(X.shape[1] / GDparams[0])):

            batch_start = (batch - 1) * GDparams[0] + 1
            batch_end = batch * GDparams[0] + 1
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]

            P, S1, h = EvaluateClassifier(Xbatch, W1, W2, b1, b2)

            gLW1, gLW2, gLb1, gLb2 = ComputeGradients(Xbatch, Ybatch, P, S1, h, W1, W2, b1, b2, reg)

            W1 -= GDparams[1] * gLW1
            W2 -= GDparams[1] * gLW2
            b1 -= GDparams[1] * gLb1
            b2 -= GDparams[1] * gLb2

        train_cost.append(ComputeCost(X, Y, W1, W2, b1, b2, reg))
        print(" Training loss at epoch", epoch + 1, "is:", train_cost[-1])
        val_cost.append(ComputeCost(X_val, Y_val, W1, W2, b1, b2, reg))
        print(" Val loss at epoch", epoch + 1, "is:", val_cost[-1], "\n")

    plt.title('cost function per epoch')
    plt.plot(train_cost, 'g', label='train cost')
    plt.plot(val_cost, 'r', label='val cost')
    plt.legend(loc='upper right')
    plt.savefig("Cost loss, lambda: "+str(reg)+", eta:"+str(GDparams[1])+".png")
    # plt.show()
    plt.clf()

    return W1, W2, b1, b2

def momentum(X, Y, y, X_val, Y_val, y_val, GDparams, W1, W2, b1, b2, reg, rho):

    train_cost = [ComputeCost(X, Y, W1, W2, b1, b2, reg)]
    val_cost = [ComputeCost(X_val, Y_val, W1, W2, b1, b2, reg)]

    v1 = np.zeros((W1.shape[0], W1.shape[1]))
    v2 = np.zeros((W2.shape[0], W2.shape[1]))
    v3 = np.zeros((b1.shape[0], b1.shape[1]))
    v4 = np.zeros((b2.shape[0], b2.shape[1]))

    for epoch in range(GDparams[2]):

        if epoch > 0:

            GDparams[1] *= 0.95  # eta decay

            if train_cost[-1] > 3 * train_cost[0]: break

        for batch in range(1, int(X.shape[1] / GDparams[0])):
            batch_start = (batch - 1) * GDparams[0] + 1
            batch_end = batch * GDparams[0] + 1
            Xbatch = X[:, batch_start:batch_end]
            Ybatch = Y[:, batch_start:batch_end]

            P, S1, h = EvaluateClassifier(Xbatch, W1, W2, b1, b2)

            gLW1, gLW2, gLb1, gLb2 = ComputeGradients(Xbatch, Ybatch, P, S1, h, W1, W2, b1, b2, reg)

            v1 = rho * v1 + GDparams[1] * gLW1
            v2 = rho * v2 + GDparams[1] * gLW2
            v3 = rho * v3 + GDparams[1] * gLb1
            v4 = rho * v4 + GDparams[1] * gLb2

            W1 -= v1
            W2 -= v2
            b1 -= v3
            b2 -= v4

        train_cost.append(ComputeCost(X, Y, W1, W2, b1, b2, reg))
        print(" Training loss at epoch", epoch + 1, "is:", train_cost[-1])
        val_cost.append(ComputeCost(X_val, Y_val, W1, W2, b1, b2, reg))
        print(" Val loss at epoch", epoch + 1, "is:", val_cost[-1], "\n")

    plt.title('cost function per epoch')
    plt.plot(train_cost, 'g', label='train cost')
    plt.plot(val_cost, 'r', label='val cost')
    plt.legend(loc='upper right')
    plt.savefig("rs_reg_" + str(reg) + "_eta_" + str(GDparams[1]) + ".png")
    # plt.show()
    plt.clf()

    return W1, W2, b1, b2

def grad_test(test):

    X_train, Y_train_hot, y_train = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    mean_X_train = np.mean(X_train, axis=0)
    X_train -= mean_X_train

    if test == 'check1':

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)

        P, S1, h = EvaluateClassifier(X_train, W1, W2, b1, b2)

        gLW1, gLW2, gLb1, gLb2 = ComputeGradients(X_train, Y_train_hot, P, S1, h, W1, W2, b1, b2, reg=0)

        gLsW1, gLsW2, gLsb1, gLsb2 = ComputeGradsNumSlow(X_train, Y_train_hot, W1, W2, b1, b2, 0, 1e-5)

        check(gLW1, gLsW1, 1e-6)
        check(gLW2, gLsW2, 1e-6)
        check(gLb1, gLsb1, 1e-6)
        check(gLb2, gLsb2, 1e-6)

    elif test == 'check2':

        X_train = X_train[:, 0:1]
        Y_train_hot = Y_train_hot[:, 0:1]

        W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)

        P, S1, h = EvaluateClassifier(X_train, W1, W2, b1, b2)

        gLW1, gLW2, gLb1, gLb2 = ComputeGradients(X_train, Y_train_hot, P, S1, h, W1, W2, b1, b2, reg=0)

        gLsW1, gLsW2, gLsb1, gLsb2 = ComputeGradsNumSlow(X_train, Y_train_hot, W1, W2, b1, b2, 0, 1e-5)

        check(gLW1, gLsW1, 1e-6)
        check(gLW2, gLsW2, 1e-6)
        check(gLb1, gLsb1, 1e-6)
        check(gLb2, gLsb2, 1e-6)

    elif test == 'check3':

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)

        P, S1, h = EvaluateClassifier(X_train, W1, W2, b1, b2)

        gLW1, gLW2, gLb1, gLb2 = ComputeGradients(X_train, Y_train_hot, P, S1, h, W1, W2, b1, b2, reg=0.1)

        gLsW1, gLsW2, gLsb1, gLsb2 = ComputeGradsNumSlow(X_train, Y_train_hot, W1, W2, b1, b2, 0.1, 1e-5)

        check(gLW1, gLsW1, 1e-6)
        check(gLW2, gLsW2, 1e-6)
        check(gLb1, gLsb1, 1e-6)
        check(gLb2, gLsb2, 1e-6)

    elif test == 'check4':

        X_train = X_train[0:100, 0:50]
        Y_train_hot = Y_train_hot[:, 0:50]

        W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)

        P, S1, h = EvaluateClassifier(X_train, W1, W2, b1, b2)

        gLW1, gLW2, gLb1, gLb2 = ComputeGradients(X_train, Y_train_hot, P, S1, h, W1, W2, b1, b2, reg=1)

        gLsW1, gLsW2, gLsb1, gLsb2 = ComputeGradsNumSlow(X_train, Y_train_hot, W1, W2, b1, b2, 1, 1e-5)

        check(gLW1, gLsW1, 1e-6)
        check(gLW2, gLsW2, 1e-6)
        check(gLb1, gLsb1, 1e-6)
        check(gLb2, gLsb2, 1e-6)

def sanity1():

    X_train, Y_train_hot, y_train = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)

    X_train = X_train[:, :1000]
    Y_train_hot = Y_train_hot[:, :1000]
    y_train = y_train[:1000]

    X_val = X_val[:, :1000]
    Y_val_hot = Y_val_hot[:, :1000]
    y_val = y_val[:1000]

    X_test = X_test[:, :1000]
    y_test = y_test[:1000]

    W1, W2, b1, b2 = MiniBatchGD(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, 0.08, 200], W1, W2, b1, b2, reg=0)

    print("Accuracy on the train set:", 100 * ComputeAccuracy(X_train, y_train, W1, W2, b1, b2), "%")
    print("Accuracy on the val set:", 100 * ComputeAccuracy(X_val, y_val, W1, W2, b1, b2), "%")
    print("Accuracy on the test set:", 100 * ComputeAccuracy(X_test, y_test, W1, W2, b1, b2), "%")

def sanity2():

    X_train, Y_train_hot, y_train = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

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

    W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)

    W1, W2, b1, b2 = momentum(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, 0.08, 200], W1, W2, b1, b2, reg=0, rho=0.9)

    print("Accuracy on the train set:", 100 * ComputeAccuracy(X_train, y_train, W1, W2, b1, b2), "%")
    print("Accuracy on the val set:", 100 * ComputeAccuracy(X_val, y_val, W1, W2, b1, b2), "%")
    print("Accuracy on the test set:", 100 * ComputeAccuracy(X_test, y_test, W1, W2, b1, b2), "%")

def quick_search():

    X_train, Y_train_hot, y_train = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    for eta in np.arange(start=0.001, stop=0.5, step=0.001):

        print("Current eta =", eta)

        W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)
        W1, W2, b1, b2 = momentum(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, eta, 5], W1, W2, b1, b2, reg=1e-6, rho=0.9)

def coarse_search():

    X_train, Y_train_hot, y_train = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    e_min = np.log(0.01)
    e_max = np.log(0.09)

    file = open("coarse_search.txt", "w")

    for pair in range(14):

        np.random.seed()
        e = e_min + (e_max - e_min) * np.random.rand(1, 1)[0][0]
        eta = np.exp(e)

        for reg in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0]:

            print("Current eta =", eta)
            print("Current lambda =", reg, "\n")

            W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)
            W1, W2, b1, b2 = momentum(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, eta, 10], W1, W2, b1, b2, reg=reg, rho=0.9)

            acc = ComputeAccuracy(X_val, y_val, W1, W2, b1, b2)
            print("lambda =", reg, "eta =", eta, "Accuracy =", 100 * acc, "%", "\n")

            file.write("lambda = "+str(reg)+" eta = "+str(eta)+" Accuracy = "+str(100 * acc)+"%"+"\n")

    file.close()

def fine_search():

    X_train, Y_train_hot, y_train = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X_val, Y_val_hot, y_val = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X_test, Y_test_hot, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))

    X_train -= mean_X_train
    X_val -= mean_X_train
    X_test -= mean_X_train

    e_min = np.log(0.01)
    e_max = np.log(0.04)

    file = open("fine_search.txt", "w")

    for pair in range(16):

        np.random.seed()
        e = e_min + (e_max - e_min) * np.random.rand(1, 1)[0][0]
        eta = np.exp(e)

        for reg in [0, 1e-6, 1e-5, 1e-4, 1e-3]:

            print("Current eta =", eta)
            print("Current lambda =", reg, "\n")

            W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)
            W1, W2, b1, b2 = momentum(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, eta, 15], W1, W2,
                                      b1, b2, reg=reg, rho=0.9)

            acc = ComputeAccuracy(X_val, y_val, W1, W2, b1, b2)
            print("lambda =", reg, "eta =", eta, "Accuracy =", 100 * acc, "%", "\n")

            file.write("lambda = " + str(reg) + " eta = " + str(eta) + " Accuracy = " + str(100 * acc) + "%" + "\n")

    file.close()

def final_evaluation():

    X1, Y1, y1 = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    X2, Y2, y2 = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    X3, Y3, y3 = LoadBatch('../../cifar-10-batches-py/data_batch_3')
    X4, Y4, y4 = LoadBatch('../../cifar-10-batches-py/data_batch_4')
    X5, Y5, y5 = LoadBatch('../../cifar-10-batches-py/data_batch_5')
    X_test, Y_test_hot, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')

    X_val = X5[:, 9000:]
    Y_val_hot = Y5[:, 9000:]
    y_val = y5[9000:]

    X = np.hstack((X1, X2, X3, X4, X5))
    Y = np.hstack((Y1, Y2, Y3, Y4, Y5))
    y = np.hstack((y1, y2, y3, y4, y5))

    X_train = X[:, 0:49000]
    Y_train_hot = Y[:, 0:49000]
    y_train = y[0:49000]

    tmp_mean_X_train = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    mean_X_train = np.dot(tmp_mean_X_train, np.ones((1, X_train.shape[1])))
    mean_X_val = np.dot(tmp_mean_X_train, np.ones((1, X_val.shape[1])))
    mean_X_test = np.dot(tmp_mean_X_train, np.ones((1, X_test.shape[1])))


    X_train -= mean_X_train
    X_val -= mean_X_val
    X_test -= mean_X_test

    W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)
    W1, W2, b1, b2 = momentum(X_train, Y_train_hot, y_train, X_val, Y_val_hot, y_val, [100, 0.0181983222118, 40], W1, W2, b1, b2, reg=0.001, rho=0.9)

    print("Accuracy on the train set:", 100 * ComputeAccuracy(X_train, y_train, W1, W2, b1, b2), "%")
    print("Accuracy on the val set:", 100 * ComputeAccuracy(X_val, y_val, W1, W2, b1, b2), "%")
    print("Accuracy on the test set:", 100 * ComputeAccuracy(X_test, y_test, W1, W2, b1, b2), "%")


def main():

    ''' Gradient descent check '''

    # grad_test('check1')
    # grad_test('check2')
    # grad_test('check3')
    # grad_test('check4')

    ''' Sanity checks for classical vanilla mini-batch and momentum version of the algorithm '''

    sanity1()
    # sanity2()

    ''' Bounds for reasonable values of the learning rate '''

    # quick_search()

    ''' Performs coarse-search '''

    # coarse_search()

    ''' Performs fine-search '''

    # fine_search()

    ''' Final evaluation given the best settings '''

    # final_evaluation()

    # X_train, Y_train_hot, y_train = LoadBatch('../../cifar-10-batches-py/data_batch_1')
    # X_val, Y_val_hot, y_val = LoadBatch('../../cifar-10-batches-py/data_batch_2')
    # X_test, Y_test_hot, y_test = LoadBatch('../../cifar-10-batches-py/test_batch')
    #
    # W1, W2, b1, b2 = initialize(Y_train_hot.shape[0], X_train.shape[0], m=50)
    #
    # P, S1, h = EvaluateClassifier(X_train[:, 0:100], W1, W2, b1, b2)

    debug=0


if __name__ == "__main__":

    main()



