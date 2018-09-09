import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical as make_class_categorical
import math
# -------------------- LOAD DATA ------------------
def LoadBatch(filename):

    # borrowed from https://www.cs.toronto.edu/~kriz/cifar.html
    def unpickle(file):
        import pickle
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

    garbage = ix(dictionary,1)
    y = dictionary[garbage]
    Y = np.transpose(make_class_categorical(y, 10))
    garbage = ix(dictionary,2)
    X = np.transpose(dictionary[garbage]) / 255


    return X, Y, y

# ------------------ INITIALIZE WEIGHTS -----------------------

def initialize_weights(d, K, variance, mode):


    np.random.seed(400)

    if mode =="simple xavier":
        variance = 1.0/float(d)
    elif mode == "xavier":
        variance = 2.0 / float(K+d)

    weight_matrix = np.random.normal(0,variance, (K,d))
    bias = np.random.normal(0 ,variance, (K,1))

    return weight_matrix, bias


# ------------------------ SOFTMAX IMPLEMENTATION -------------

def softmax(p):
# softmax for debugging purposes

    res = np.copy(p)

    for i in range(res.shape[0]):

        for j in range(res.shape[1]):

            res[i,j] = np.exp(p[i,j])/ (float)(np.sum(np.exp(p[:,j])))

    return res

def alt_softmax(X, theta=1.0, axis=None):

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
            
# ----------------------- ASSIGNMENT INSTRUCTION FUNCTIONS -----------------
def EvaluateClassifier(X, W, b):

    s = np.dot(W,X) + b
    # P = softmax(s)
    alt_P = alt_softmax(s, axis =0)
    return alt_P

def SVM_loss(s, label):
    loss = 0.0
    for i in range(len(s)):
        if i != label:
            loss += np.maximum(0, s[i] - s[label] + 1)
    return loss

def ComputeCost(X, Y, y, W, b, regularization_term, loss_mode):

    P = EvaluateClassifier(X,W,b)

    if loss_mode=="cross=entropy":

        cross_entropy_loss = 0
        for i in range(0, X.shape[1]):
            cross_entropy_loss-= np.log(np.dot(Y[:,i].T,P[:,i]))
        cross_entropy_loss/=float(X.shape[1])

        weight_sum = np.power(W,2).sum()

        return cross_entropy_loss + regularization_term*weight_sum

    else:

        svm_loss = 0
        for i in range(0, len(y)):
            x = np.expand_dims(X[:,i], axis = 1)
            s = np.dot(W, x) + b
            svm_loss += SVM_loss(s, y[i])
            
        svm_loss /= float(X.shape[1])
        weight_sum = np.power(W, 2).sum()

        return svm_loss + 2*regularization_term * weight_sum

def ComputeAccuracy(X, y, W, b):
# slightly modified and done through the labels y!

    P = EvaluateClassifier(X,W,b)

    miss_classified_data=0
    for i in range(X.shape[1]):

        predicted_class_index = np.argmax(P[:,i])

        if (predicted_class_index!=y[i]):
            miss_classified_data+=1

    return 1 - miss_classified_data/float(X.shape[1])

def ComputeAccuracySVM(X, y, W):
# slightly modified and done through the labels y!

    P = np.dot(W,X)

    miss_classified_data=0
    for i in range(X.shape[1]):

        predicted_class_index = np.argmax(P[:,i])

        if (predicted_class_index!=y[i]):
            miss_classified_data+=1

    return 1 - miss_classified_data/float(X.shape[1])


def ComputeGradients(X, Y, y, P, W, b, regularization_term, mode):

    gradW = np.zeros((W.shape[0],W.shape[1]))
    gradb = np.zeros((W.shape[0], 1))

    # ---------------CROSS ENTROPY LOSS---------------

    for i in range(X.shape[1]):

        g = -(Y[:,i] - P[:,i]).T
        gradb += g.T.reshape(g.shape[0], 1)
        gradW += np.dot(g.T.reshape(g.shape[0], 1), X[:, i].T.reshape(1, X.shape[0]))

    # normalize
    gradW/=X.shape[1]
    gradW+= 2*regularization_term*W
    gradb/=X.shape[1]

    return gradW, gradb

# ---------------SVM LOSS AND COST---------------
def ComputeGradientsSVM(X, y, W, b, regularization_term):

    gradW = np.zeros((W.shape[0], W.shape[1]))
    gradb = np.zeros((W.shape[0], 1))

    for i in range(X.shape[1]):
        x = np.expand_dims(X[:, i], axis=1)
        s = (np.dot(W, x) + b).clip(0)
        g = (s[y[i]] - s - 1 < 0) * 1
        g[y[i]] = - (np.sum(g) - 1)
        gradb += g
        gradW += np.dot(g, x.T)
        gradW /= len(y)
        gradb /= len(y)

    J_grad_W = gradW + regularization_term * W
    J_grad_b = gradb
    return J_grad_W, J_grad_b

# --------------------------- EXTRA FUNCTIONS ------------------------------
# compute accuracy when prediction is based on a majority vote
def ComputeAccuracyEnsemble(X, y, classifiers):

    miss_classified_data = 0
    solution = dict()
    classifier_suggestion = dict()
    classifier_performance = dict()

    for i in range(X.shape[1]):

        for classifier_index in range(len(classifiers)):

            classifier = classifiers[classifier_index]
            predicted_class_index = np.argmax(classifier[:, i])
            classifier_suggestion[classifier_index] = classifier_suggestion.get(classifier_index, 0) + predicted_class_index
            solution[predicted_class_index]=solution.get(predicted_class_index, 0) +1

        predicted_class_index=-1
        count=-1

        best = 0
        
        for elem in solution:
            
            if solution[elem]>count:
                predicted_class_index= elem
                count= solution[predicted_class_index]
                best = 1

            elif solution[elem]==count:
                best+=1


        # more than one classes are equally likely to happen:
        if best>1:

            winner = -1
            for classifier_index in range(len(classifiers)):

                if classifier_suggestion[classifier_index]>winner:

                    predicted_class_index = classifier_suggestion[classifier_index]

        if (predicted_class_index!=y[i]):
            miss_classified_data+=1

        # rate the performance of all classifiers for future reference
        for classifier_index in range(len(classifiers)):

            if classifier_suggestion[classifier_index] == y[i]:
                classifier_performance[classifier_index] = classifier_performance.get(classifier_index, 0) +1
            else:
                classifier_performance[classifier_index] = classifier_performance.get(classifier_index, 0) - 1

        solution.clear()
        classifier_suggestion.clear()

    return 1 - miss_classified_data/float(X.shape[1])

# ---------------------- MINI BATCH ---------------------------
def MiniBatchGD( X, Y, validation_X, validation_Y, y, y_validation, GDparams, W, b, regularization_term,loss_mode, with_early_stopping, with_patience, patience=12):

    number_of_mini_batches = GDparams[0]
    eta = GDparams[1]
    epoches=GDparams[2]

    cost=[]
    val_cost=[]

    if loss_mode=="cross-entropy":

        early_stopping = 10e8

        # uncomment for patience in early stopping
        early_stopping_cnt=0

        save_W = np.copy(W)
        save_b= np.copy(b)

        temp_W = np.copy(W)
        temp_b = np.copy(b)

        for epoch in range(epoches):

            for batch in range(1,int(X.shape[1]/number_of_mini_batches)):

                start = (batch-1)*number_of_mini_batches+1
                end = batch*number_of_mini_batches+1

                P = EvaluateClassifier(X[:, start:end], W, b )

                gradW, gradb = ComputeGradients(X=X[:, start:end], Y=Y[:, start:end],y=y[start:end], P=P, W=W, b=b, regularization_term=regularization_term, mode="Cross-entropy")

                # Compute similarity between analytic and numerical computation
                # gradW_num, gradb_num = ComputeGradsNumSlow(X=X[:, start:end], Y=Y[:, start:end], y=y, W=W, b=b, regularization_term=regularization_term)
                # check_similarity(gradb, gradW, gradb_num, gradW_num)

                temp_W = np.copy(W)
                temp_b = np.copy(b)

                W-=eta*gradW
                b-=eta*gradb

            epoch_cost = ComputeCost(X=X, Y=Y, y=y, W=W, b=b, regularization_term=regularization_term, loss_mode=loss_mode)
            val_epoch_cost = ComputeCost(X=validation_X, Y=validation_Y,y=y_validation, W=W, b=b, regularization_term=regularization_term, loss_mode=loss_mode)

            # ---------EARLY STOPPING----------
            if with_early_stopping:
                if with_patience:
                    if early_stopping - val_epoch_cost > 1e-6:
                        early_stopping_cnt= 0
                        early_stopping= val_epoch_cost
                    else:
                        if early_stopping_cnt == patience:
                            print("Early stopping after "+str(epoch)+" epochs")
                            return save_W, save_b
                        else:
                            if early_stopping_cnt == 0:
                                save_W= temp_W
                                save_b= temp_b
                            early_stopping_cnt+=1
                else:
                    if early_stopping - val_epoch_cost > 1e-6:
                        early_stopping= val_epoch_cost
                    else:
                        print("Early stopping after " + str(epoch) + " epochs")
                        return temp_W, temp_b

            cost.append(epoch_cost)
            val_cost.append(val_epoch_cost)

        # uncomment to visualize
        visualize_costs(cost, val_cost, regularization_term, eta)

        return W, b

    elif loss_mode=="SVM":

        for epoch in range(epoches):

            for batch in range(1, int(X.shape[1] / number_of_mini_batches)):

                start = (batch - 1) * number_of_mini_batches + 1
                end = batch * number_of_mini_batches + 1

                gradW, gradb = ComputeGradientsSVM(X=X[:, start:end],y=y[start:end], W=W, b=b, regularization_term=regularization_term)

                W -= eta * gradW
                b -= eta * b

            print(epoch)

            training_loss = ComputeCost(X=validation_X, Y=validation_Y,y=y_validation, W=W, b=b, regularization_term=regularization_term, loss_mode=loss_mode)
            val_epoch_cost = ComputeCost(X=X, Y=Y,y=y_validation, W=W, b=b, regularization_term=regularization_term, loss_mode=loss_mode)

            cost.append(training_loss)
            val_cost.append(val_epoch_cost)

        # uncomment to visualize
        visualize_costs(cost, val_cost, regularization_term, eta)

        return W, b

# ------------------------- VISUALIZING FUNCTIONS -------------------
def visualize_costs(cost, val_cost, regularization_term, eta):
    plt.title('Cost function loss per epoch, $\lambda$=' + str(regularization_term) + ', $\eta$=' + str(eta))
    plt.plot(cost, 'g', label='Training set ')
    plt.plot(val_cost, 'r', label='Validation set')
    plt.legend(loc='upper right')
    plt.savefig("../images/Cost loss, lambda: " + str(regularization_term) + ", eta:" + str(eta) + ".png")
    plt.show()
    plt.clf()

def visualize_image(image):
    image = image.reshape(3, 32, 32)
    image = (image - image.min()) / (image.max() - image.min())
    image = image.T
    image = np.rot90(image, k=3)
    plt.imshow(image)
    plt.show()

def weights_vis(W, GDparams, regularization_term):
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
    plt.savefig("../images/Weight visualization, $\lambda=$" + str(regularization_term)+ ", $\eta=$" + str(GDparams[1]) + ", $epochs=$:" + str(GDparams[2]) + ".png", bbox_inches='tight')
    plt.show()
    plt.clf()

# ------------------------------------ EXPERIMENTS -------------------------------------------------------------------------------------------------------------------------------------
# experiments with improvements in the defaults settings
# Improvement no.1: Training for a larger number of epochs, and using a validation set for early stopping

# Improvement no.2: Using all the available datasets for training, and keeping only the last 1000 data from the second training batch (original training set) as the new validation set

# Improvement no.3: Training 5 different classifiers (one from each dataset) and predicting the class of each data using the majority vote. During the "voting" process, each classifier
# is rated according to its votes up to this point. When 2 or more classes share equal votes, the best classifier so far decides for the predicted class index.

# Improvement no.4:  Fine tuning of parameters $\eta$ and $\lambda$ to define some "optimal" values that perform better than the best performance so far ($\eta=0.01$ and $\lambda=0$)%

# Improvement no.5: Xavier initialization on the weights, setting the variance of their zero-mean normal distribution to $\frac{1}{d}$,
# since if we transpose the matrices $X$, $W$, the dimensionality of each image is number of inputs to the first layer of the weights

# Improvement no.6: Decaying the learning rate to its half value every 10 epochs

# Improvements no.7- : Using combinations of the above
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dropping learnig rate, borrowed from: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# Tuning of the parameters of eta, l to find the optinal combo
# For 40 epochs, the best seems to be: lambda=0.01, eta=0.006 with accuracy 0.3892:
def fine_tuning(weight_initialization, with_early_stopping):

    X_training_1, yaining_1, yaining_1 = LoadBatch('../Datasets/data_batch_1')
    X_training_2, yaining_2, yaining_2= LoadBatch('../Datasets/data_batch_2')
    X_testing, _, y_testing = LoadBatch('../Datasets/test_batch')

    weight_matrix, bias = initialize_weights(d=3072, K=10, variance=0.01, mode=weight_initialization)

    best_acc = 0

    for eta in np.arange(0.001,0.015,0.001):

        for reg in [0, 0.0001, 0.001, 0.01]:
        # for reg in [0, 0.0001, 0.01, 0.1, 1,10, 100]:
            W, b = MiniBatchGD(
                X=X_training_1,
                Y=yaining_1,
                validation_X=X_training_2,
                validation_Y=yaining_2,
                y=yaining_1,
                y_validation=yaining_2,
                GDparams=[100, eta, 40],
                W=weight_matrix,
                b=bias,
                regularization_term=reg,
                loss_mode="cross-entropy",
                with_early_stopping=with_early_stopping)

            acc = ComputeAccuracy(X_testing, y_testing, W, b)
            print("------------------------------------------------------\n")
            if acc > best_acc:
                print("!!!!!!!!!!!!!!!!!!!BEST SO FAR !!!!!!!!!!!!!!!!!!!!!")
                best_acc = acc
            print("Lambda: " + str(reg))
            print("Eta: " + str(eta))
            print("Accuracy:" + str(acc))
            print("------------------------------------------------------\n")
            
def ensemble_learners():

    X_training_1, yaining_1, yaining_1 = LoadBatch('../Datasets/data_batch_1')
    X_training_2, yaining_2, yaining_2 = LoadBatch('../Datasets/data_batch_2')
    X_training_3, yaining_3, yaining_3 = LoadBatch('../Datasets/data_batch_3')
    X_training_4, yaining_4, yaining_4 = LoadBatch('../Datasets/data_batch_4')
    X_training_5, yaining_5, yaining_5 = LoadBatch('../Datasets/data_batch_5')


    X_validation = np.copy(X_training_2[:, 9000:])
    Y_validation = np.copy(yaining_2[:, 9000:])
    y_validation = np.copy(yaining_2[:, 9000:])

    X_training_2 = np.copy(X_training_2[:, :9000])
    yaining_2 = np.copy(yaining_2[:, :9000])


    X_testing, _, y_testing = LoadBatch('../Datasets/test_batch')

    classifiers = []

    # training model on dataset_1
    weight_matrix, bias = initialize_weights(d=3072, K=10,variance=0.01, mode="normal")

    W, b = MiniBatchGD(
        X=X_training_1,
        Y=yaining_1,
        validation_X=X_validation,
        validation_Y=Y_validation,
        y=yaining_1,
        y_validation=y_validation,
        GDparams=[100, 0.06, 100],
        W=weight_matrix,
        b=bias,
        regularization_term=0,
        loss_mode="cross-entropy",
        with_early_stopping=True)

    classifiers.append(EvaluateClassifier(X_testing,W,b))

    # training model on dataset_2
    weight_matrix, bias = initialize_weights(d=3072, K=10, variance=0.01, mode="normal")

    W, b = MiniBatchGD(
        X=X_training_2,
        Y=yaining_2,
        validation_X=X_validation,
        validation_Y=Y_validation,
        y=yaining_2,
        y_validation=y_validation,
        GDparams=[100, 0.01, 40],
        W=weight_matrix,
        b=bias,
        regularization_term=0,
        loss_mode="cross-entropy",
        with_early_stopping=False)

    classifiers.append(EvaluateClassifier(X_testing, W, b))

    # training model on dataset 3

    W, b = MiniBatchGD(
        X=X_training_3,
        Y=yaining_3,
        validation_X=X_validation,
        validation_Y=Y_validation,
        y=yaining_3,
        y_validation=y_validation,
        GDparams=[100, 0.01, 40],
        W=weight_matrix,
        b=bias,
        regularization_term=0,
        loss_mode="cross-entropy",
        with_early_stopping=False)

    classifiers.append(EvaluateClassifier(X_testing,W,b))

    # training model on dataset 4

    W, b = MiniBatchGD(
        X=X_training_4,
        Y=yaining_4,
        validation_X=X_validation,
        validation_Y=Y_validation,
        y=yaining_4,
        y_validation=y_validation,
        GDparams=[100, 0.01, 40],
        W=weight_matrix,
        b=bias,
        regularization_term=0,
        loss_mode="cross-entropy",
        with_early_stopping=False)

    classifiers.append(EvaluateClassifier(X_testing, W, b))

    # training model on dataset 5

    W, b = MiniBatchGD(
        X=X_training_5,
        Y=yaining_5,
        validation_X=X_validation,
        validation_Y=Y_validation,
        y=yaining_5,
        y_validation=y_validation,
        GDparams=[100, 0.01, 40],
        W=weight_matrix,
        b=bias,
        regularization_term=0,
        loss_mode="cross-entropy",
        with_early_stopping=False)

    classifiers.append(EvaluateClassifier(X_testing, W, b))

    print("---------------------------------------------")
    print("Bagging:")
    print("Accuracy: "+str(round(ComputeAccuracyEnsemble(X_testing, y_testing, classifiers)*100,2))+"%")
    print("---------------------------------------------")

#experiments that need to be run for the compulsory part of the assignment
def simple_experiments():

    X_training_1, yaining_1, yaining = LoadBatch('../Datasets/data_batch_1')
    X_training_2, yaining_2, y_validation = LoadBatch('../Datasets/data_batch_2')
    X_testing, _ , y_testing = LoadBatch('../Datasets/test_batch')

    weight_matrix, bias = initialize_weights(d=3072, K=10, mode="normal")
    for list in [[0,0.1], [0,0.01], [0.1, 0.01], [1,0.01]]:

        reg=list[0]
        eta=list[1]

        W, b= MiniBatchGD(
            X=X_training_1,
            Y=yaining_1,
            validation_X= X_training_2,
            validation_Y= yaining_2,
            y=yaining,
            y_validation=y_validation,
            GDparams=[100,eta,100],
            W=weight_matrix,
            b=bias,
            regularization_term=0.01,
            loss_mode="cross-entropy",
            with_early_stopping=True,
            with_patience=True)
            # regularization_term=reg)

        # uncomment to visualize the weights
        # weights_vis(W, GDparams=[100,eta,40], regularization_term=reg)

        print("------------------------------------------------------\n")
        print("Lambda: "+str(reg))
        print("Eta: "+str(eta))
        print("Accuracy:"+str(round(ComputeAccuracy(X_testing, y_testing, W, b)*100,2))+"%")
        print("------------------------------------------------------\n")

def extended_experiments(loss_mode):


    X_training_1, yaining_1, yaining_1 = LoadBatch('../Datasets/data_batch_1')
    X_training_3, yaining_3, yaining_3 = LoadBatch('../Datasets/data_batch_3')
    X_training_4, yaining_4, yaining_4 = LoadBatch('../Datasets/data_batch_4')
    X_training_5, yaining_5, yaining_5 = LoadBatch('../Datasets/data_batch_5')

    X_training = np.concatenate((X_training_1, X_training_3), axis=1)
    X_training = np.copy(np.concatenate((X_training, X_training_4), axis=1))
    X_training = np.copy(np.concatenate((X_training, X_training_5), axis=1))

    yaining = np.concatenate((yaining_1, yaining_3), axis=1)
    yaining = np.copy(np.concatenate((yaining, yaining_4), axis=1))
    yaining = np.copy(np.concatenate((yaining, yaining_5), axis=1))

    yaining = np.concatenate((yaining_1, yaining_3), axis=1)
    yaining = np.copy(np.concatenate((yaining, yaining_4), axis=1))
    yaining = np.copy(np.concatenate((yaining, yaining_5), axis=1))

    X_validation, Y_validation, y_validation = LoadBatch('../Datasets/data_batch_2')

    X_training = np.concatenate((X_training, X_validation[:, :9000]), axis=1)
    yaining = np.concatenate((yaining, Y_validation[:, :9000]), axis=1)
    yaining = np.concatenate((yaining, y_validation[:, :9000]), axis=1)

    X_validation = np.copy(X_validation[:, 9000:])
    Y_validation = np.copy(Y_validation[:, 9000:])
    y_validation = np.copy(y_validation[:, 9000:])
    X_testing, Y_testing, y_testing = LoadBatch('../Datasets/test_batch')

    weight_matrix, bias = initialize_weights(d=3072, K=10,variance=0.01, mode="normal")
    W, b = MiniBatchGD( X=X_training, Y=yaining, validation_X=X_validation, validation_Y=Y_validation, y=yaining, y_validation=y_validation,
        GDparams=[100,0.01,200],
        W=weight_matrix, b=bias,
        regularization_term=0,
        loss_mode="cross-entropy",
        with_early_stopping=True)

    print("Accuracy:" + str(round(ComputeAccuracy(X_testing, y_testing, W, b) * 100, 2)) + "%")

def single_experiment_SVM(eta, regularization_term, weight_mode):
    X_training_1, yaining_1, yaining = LoadBatch('../Datasets/data_batch_1')
    X_training_2, yaining_2, y_validation = LoadBatch('../Datasets/data_batch_2')
    X_testing, _, y_testing = LoadBatch('../Datasets/test_batch')

    weight_matrix, bias = initialize_weights(d=3072, K=10, variance=0.01, mode=weight_mode)
    # weight_matrix= np.copy(np.concatenate((weight_matrix, bias), axis=1))
    # pseudo= np.ones(shape=(1,X_training_1.shape[1]))
    # X_training_1= np.concatenate((X_training_1, pseudo), axis=0)
    # X_training_2= np.concatenate((X_training_2, pseudo), axis=0)
    # pseudo_test= np.ones(shape=(1,X_testing.shape[1]))
    # X_testing= np.concatenate((X_testing, pseudo_test), axis=0)

    W, _ = MiniBatchGD(
        X=X_training_1,
        Y=yaining_1,
        validation_X=X_training_2,
        validation_Y=yaining_2,
        y=yaining,
        y_validation=y_validation,
        GDparams=[100, eta, 40],
        W=weight_matrix,
        b=bias,
        regularization_term=regularization_term,
        loss_mode="SVM",
        with_early_stopping=False,
        with_patience=False)

    # uncomment to visualize the weights
    # weights_vis(W, GDparams=[100,eta,40], regularization_term=reg)

    print("------------------------------------------------------\n")
    print("Lambda: " + str(regularization_term))
    print("Eta: " + str(eta))
    print("Accuracy:" + str(round(ComputeAccuracySVM(X_testing, y_testing, W) * 100, 2)) + "%")
    print("------------------------------------------------------\n")

def single_experiment(eta, regularization_term, weight_mode):

    X_training_1, yaining_1, yaining = LoadBatch('../Datasets/data_batch_1')
    X_training_2, yaining_2, y_validation = LoadBatch('../Datasets/data_batch_2')
    X_testing, _ , y_testing = LoadBatch('../Datasets/test_batch')

    weight_matrix, bias = initialize_weights(d= 3072, K= 10, variance= 0.01, mode=weight_mode)

    W, b= MiniBatchGD(
        X=X_training_1,
        Y=yaining_1,
        validation_X= X_training_2,
        validation_Y= yaining_2,
        y=yaining,
        y_validation=y_validation,
        GDparams=[100,eta,40],
        W=weight_matrix,
        b=bias,
        regularization_term=regularization_term,
        loss_mode="SVM",
        with_early_stopping=False,
        with_patience=False)

    # uncomment to visualize the weights
    # weights_vis(W, GDparams=[100,eta,40], regularization_term=reg)

    print("------------------------------------------------------\n")
    print("Lambda: " + str(regularization_term))
    print("Eta: " + str(eta))
    print("Accuracy:" + str(round(ComputeAccuracy(X_testing, y_testing, W, b) * 100, 2)) + "%")
    print("------------------------------------------------------\n")

# ---------------------- RUN!!!! ----------------------

def main(mode):

    if mode == "single":
        eta= 0.001
        regularization_term=1
        print("Single experiment with eta:"+str(eta)+" lambda:"+str(regularization_term))
        single_experiment(eta=eta, regularization_term=regularization_term, weight_mode="normal")

    elif mode == "singleSVM":
        eta= 0.001
        regularization_term=0.1
        print("Single experiment with eta:"+str(eta)+" lambda:"+str(regularization_term))
        single_experiment_SVM(eta=eta, regularization_term=regularization_term, weight_mode="normal")


    elif mode == "simple":

        simple_experiments()

    elif mode == "extended":

        extended_experiments(loss_mode="Cross-entropy")


if __name__ == '__main__':

    # Uncomment for a single-experiment
    # main(mode="single")

    # Uncomment for a single SVM experiment
    main(mode="singleSVM")

    # Uncomment for the compulsory-part experiment
    # main(mode="simple")

    # Uncomment for some of the bonus-part experiments
    # main(mode="extended")

    # Uncomment for fine tuning on lambda and eta
    # fine_tuning(weight_initialization="normal", with_early_stopping=False)

    # Uncomment for combining several simple classifiers
    # ensemble_learners()