import numpy as np
from matplotlib import pyplot as plt
from keras.utils import to_categorical as make_class_categorical

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

def initialize_weights(d, K, variance):


    np.random.seed(400)

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

def ComputeCost(X, Y, W, b, regularization_term):

    P = EvaluateClassifier(X,W,b)

    cross_entropy_loss = 0
    for i in range(0, X.shape[1]):
        cross_entropy_loss-= np.log(np.dot(Y[:,i].T,P[:,i]))
    cross_entropy_loss/=float(X.shape[1])

    weight_sum = np.power(W,2).sum()

    return cross_entropy_loss + regularization_term*weight_sum

def ComputeAccuracy(X, y, W, b):
# slightly modified and done through the labels y!

    P = EvaluateClassifier(X,W,b)

    miss_classified_data=0
    for i in range(X.shape[1]):

        predicted_class_index = np.argmax(P[:,i])

        if (predicted_class_index!=y[i]):
            miss_classified_data+=1

    return 1 - miss_classified_data/float(X.shape[1])

def ComputeGradients(X, Y, P, W, b, regularization_term):

    gradW = np.zeros((W.shape[0],W.shape[1]))
    gradb = np.zeros((W.shape[0], 1))

    for i in range(X.shape[1]):

        g = -(Y[:,i] - P[:,i]).T
        gradb += g.T.reshape(g.shape[0],1)
        gradW += np.dot(g.T.reshape(g.shape[0],1), X[:,i].T.reshape(1,X.shape[0]))

    # normalize
    gradW/=X.shape[1]
    gradW+= 2*regularization_term*W
    gradb/=X.shape[1]


    return gradW, gradb


# --------------------------- NUMERICAL GRADIENTS -------------------------------
def ComputeGradsNum(X, Y, W, b, regularization_term, h=10e-6):

    gradW = np.zeros((W.shape[0], W.shape[1]))
    gradb = np.zeros((W.shape[0], 1))

    c = ComputeCost(X,Y,W,b,regularization_term)

    for i in range(0,b.shape[0]):

        b_try = np.copy(b)
        b_try[i,0] += h
        c2=ComputeCost(X,Y,W,b_try,regularization_term)
        gradb[i,0]=(c2-c)/h

    for i in range(0, W.shape[0]):
        for j in range(0, W.shape[1]):

            W_try=np.copy(W)
            W_try[i,j]+=h
            c2 = ComputeCost(X, Y, W_try, b_try, regularization_term)
            gradW[i, j] = (c2 - c) / h

    return gradW, gradb

def ComputeGradsNumSlow(X, Y, W, b, regularization_term, h=10e-6):

    no = W.shape[1]
    d = X.shape[1]

    gradW = np.zeros((W.shape[0], W.shape[1]))
    gradb = np.zeros((W.shape[0], 1))

    for i in range(b.shape[0]):

        b_try = np.copy(b)
        b_try[i,0] -= h
        c1 = ComputeCost(X,Y,W,b_try, regularization_term)
        b_try = np.copy(b)
        b_try[i,0] += h
        c2 = ComputeCost(X,Y,W,b_try, regularization_term)
        gradb[i,0] = (c2-c1)/(2*h)


    for i in range(0, W.shape[0]):
        for j in range(0, W.shape[1]):

            W_try=np.copy(W)
            W_try[i,j]-=h
            c1 = ComputeCost(X, Y, W_try, b_try, regularization_term)


            W_try=np.copy(W)
            W_try[i,j]+=h
            c2 = ComputeCost(X, Y, W_try, b_try, regularization_term)
            gradW[i, j] = (c2 - c1) / (2*h)

    return gradW, gradb

# --------------------------------- GRADIENTS SIMILARITY CHECK -------------------------
def check_similarity(gradb, gradW, gradb_num, gradW_num):

    # This function is based on the assumption that lambda is 0
    # And is only provided for test purposes
    W_abs = np.abs(gradW - gradW_num)
    b_abs = np.abs(gradb - gradb_num)
    W_nominator = np.average(W_abs)
    b_nominator = np.average(b_abs)


    gradW_abs = np.absolute(gradW)
    gradW_num_abs = np.absolute(gradW_num)

    gradb_abs = np.absolute(gradb)
    gradb_num_abs = np.absolute(gradb)

    sum_W = gradW_abs + gradW_num_abs
    sum_b = gradb_abs + gradb_num_abs
    check_W = W_nominator / np.amax(sum_W)
    check_b = b_nominator / np.amax(sum_b)


    if check_W < 10e-6 and check_b < 1e-6:
        print( "Success!!")
        print("Average error on weights=", check_W)
        print("Average error on bias=", check_b)
    else:
        print("Failure")
        print("Average error on weights=", check_W)
        print("Average error on bias=", check_b)

# ---------------------- MINI BATCH ---------------------------
def MiniBatchGD( X, Y, validation_X, validation_Y, GDparams, W, b , regularization_term):

    number_of_mini_batches = GDparams[0]
    eta = GDparams[1]
    epoches=GDparams[2]

    cost=[]
    val_cost=[]

    for epoch in range(epoches):

        for batch in range(1,int(X.shape[1]/number_of_mini_batches)):

            start = (batch-1)*number_of_mini_batches+1
            end = batch*number_of_mini_batches+1

            P = EvaluateClassifier(X[:, start:end], W, b )

            gradW, gradb = ComputeGradients(X=X[:, start:end], Y=Y[:, start:end], P=P, W=W, b=b, regularization_term=regularization_term)

            # Compute similarity between analytic and numerical computation
            # gradW_num, gradb_num = ComputeGradsNumSlow(X=X[:, start:end], Y=Y[:, start:end], W=W, b=b, regularization_term=regularization_term)
            # check_similarity(gradb, gradW, gradb_num, gradW_num)
            # continue

            W-=eta*gradW
            b-=eta*gradb

        epoch_cost = ComputeCost(X=X, Y=Y, W=W, b=b, regularization_term=regularization_term)
        val_epoch_cost = ComputeCost(X=validation_X, Y=validation_Y, W=W, b=b, regularization_term=regularization_term)

        cost.append(epoch_cost)
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
    # plt.savefig("../images/Cost loss, lambda: " + str(regularization_term) + ", eta:" + str(eta) + ".png")
    plt.show()
    plt.clf()

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
    # plt.savefig("../images/Weight visualization, $\lambda=$" + str(regularization_term)+ ", $\eta=$" + str(GDparams[1]) + ", $epochs=$:" + str(GDparams[2]) + ".png", bbox_inches='tight')
    plt.show()
    plt.clf()

# ------------------------------------ EXPERIMENTS -----------------------------

def single_experiment(eta, regularization_term):

    X_training_1, Y_training_1, _ = LoadBatch('../Datasets/data_batch_1')
    X_training_2, Y_training_2, _ = LoadBatch('../Datasets/data_batch_2')
    X_testing, _ , y_testing = LoadBatch('../Datasets/test_batch')

    weight_matrix, bias = initialize_weights(d=3072, K=10, variance=0.01)

    W, b= MiniBatchGD(
        X=X_training_1,
        Y=Y_training_1,
        validation_X= X_training_2,
        validation_Y= Y_training_2,
        GDparams=[100,eta,1],
        W=weight_matrix,
        b=bias,
        regularization_term=regularization_term)

    # uncomment to visualize the weights
    # weights_vis(W, GDparams=[100,eta,40], regularization_term=reg)

    print("------------------------------------------------------\n")
    print("Lambda: " + str(regularization_term))
    print("Eta: " + str(eta))
    print("Accuracy:" + str(round(ComputeAccuracy(X_testing, y_testing, W, b) * 100, 2)) + "%")
    print("------------------------------------------------------\n")


#experiments that need to be run for the compulsory part of the assignment
def simple_experiments():

    X_training_1, Y_training_1, _ = LoadBatch('../Datasets/data_batch_1')
    X_training_2, Y_training_2, _ = LoadBatch('../Datasets/data_batch_2')
    X_testing, _ , y_testing = LoadBatch('../Datasets/test_batch')

    weight_matrix, bias = initialize_weights(d=3072, K=10, variance=0.01)
    for list in [[0,0.1], [0,0.01], [0.1, 0.01], [1,0.01]]:

        reg=list[0]
        eta=list[1]

        W, b= MiniBatchGD(
            X=X_training_1,
            Y=Y_training_1,
            validation_X= X_training_2,
            validation_Y= Y_training_2,
            GDparams=[100,eta,40],
            W=weight_matrix,
            b=bias,
            regularization_term=reg)

        # uncomment to visualize the weights
        weights_vis(W, GDparams=[100,eta,40], regularization_term=reg)

        print("------------------------------------------------------\n")
        print("Lambda: "+str(reg))
        print("Eta: "+str(eta))
        print("Accuracy:"+str(round(ComputeAccuracy(X_testing, y_testing, W, b)*100,2))+"%")
        print("------------------------------------------------------\n")


def main(mode):

    if mode == "simple":

        simple_experiments()

    elif mode == "single":

        single_experiment(eta=0.01, regularization_term=0 )

# ---------------------- RUN!!!! ----------------------
if __name__ == '__main__':

    # main(mode="simple")
    main(mode="single")
