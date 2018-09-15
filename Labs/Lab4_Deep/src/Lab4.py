import numpy as np
import matplotlib.pyplot as plt

def Load_Text_Data(file_path='../goblet_book.txt'):
    """
    Reads the input data.

    :param file_path: (Optional) Position of the txt file in the local system.

    :return: book_data: all input characters and unique_characters: unique single characters of the input data.
    """
    book_data = open(file_path, 'r').read()
    unique_characters = list(sorted(set(book_data)))

    return book_data, unique_characters

def Char_to_Ind(chars_list, unique_chars):
    """
    Maps the original characters to integers.

    :param chars_list: The list of characters to be encoded into integers.
    :param unique_chars: The set of unique characters available.

    :return: A list of integers that correspond to the characters.
    """

    encoded_characters = []

    for char in chars_list:
        for index, letter in enumerate(unique_chars):

            if char == letter:

                encoded_characters.append(index)

    return encoded_characters

def Ind_to_Char(one_hot_representation, unique_chars_list):
    """
    Maps a list of integers to their corresponding characters,

    :param one_hot_representation: A list of one_hot representations to be transformed to their correspoding characters.
    :param unique_chars_list: The list of unique characters.

    :return: The actual character sequence.
    """

    actual_character_sequence = []

    for i in range(one_hot_representation.shape[1]):

        letter_pos = np.where(one_hot_representation[:,i] == 1.0)[0][0]
        actual_character_sequence.append(unique_chars_list[letter_pos])

    return actual_character_sequence


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

def create_one_hot_endoding(x, K):
    """
    Creates the one hot encoding representation of an array.


    :param x: The array that we wish to map in an one-hot representation.
    :param K: The number of distinct classes.

    :return: One hot representation of this number.
    """

    x_encoded = np.zeros((K, len(x)))
    for index, elem in enumerate(x):

        x_encoded[elem, index] = 1.0

    return x_encoded

class RNN:
    """
    Recurrent Neural Network object
    """

    def __init__(self, m, K, eta, seq_length, std, epsilon=1e-10):
        """
        Initial setting of the RNN.

        :param m: Dimensionality of the hidden state.
        :param K: Number of unique classes to identify.
        :param eta: The learning rate of the training process.
        :param seq_length: The length of the input sequence.
        :param std: the variance of the normal distribution that initializes the weight matrices.
        :param epsilon: epsilon parameter of ada-grad.
        """

        self.m = m
        self.K = K
        self.eta = eta
        self.seq_length = seq_length
        self.std = std
        self.epsilon = epsilon

    def init_weights(self):
        """
        Initializes the weights and bias matrices
        """

        np.random.seed(400)

        U = np.random.normal(0, self.std, size=(self.m, self.K))
        W = np.random.normal(0, self.std, size=(self.m, self.m))
        V = np.random.normal(0, self.std, size=(self.K, self.m))

        b = np.zeros((self.m, 1))
        c = np.zeros((self.K, 1))

        return [W, U, b, V, c]

    def synthesize_sequence(self, h0, x0, weight_parameters, text_length):
        """
        Synthesizes a sequence of characters under the RNN values.

        :param self: The RNN.
        :param h0: Hidden state at time 0.
        :param x0: First (dummy) input vector of the RNN.
        :param weight_parameters: The weighst and biases of the RNN, which are:
            :param W: Hidden-to-Hidden weight matrix.
            :param U: Input-to-Hidden weight matrix.
            :param b: Bias vector of the hidden layer.
            :param V: Hidden-to-Output weight matrix.
            :param c: Bias vector of the output layer.
        :param text_length: The length of the text you wish to generate

        :return: Synthesized text through.
        """

        W, U, b, V, c = weight_parameters
        Y = np.zeros(shape=(x0.shape[0], text_length))

        alpha = np.dot(W, h0) + np.dot(U, np.expand_dims(x0[:,0], axis=1)) + b
        h = np.tanh(alpha)
        o = np.dot(V, h) + c
        p = softmax(o)

        # Compute the cumulative sum of p and draw a random sample from [0,1)
        cumulative_sum = np.cumsum(p)
        draw_number = np.random.sample()

        # Find the element that corresponds to this random sample
        pos = np.where(cumulative_sum > draw_number)[0][0]

        # Create one-hot representation of the found position
        Y[pos, 0] = 1.0

        h0 = np.copy(h)
        x0 = np.expand_dims(np.copy(Y[:,0]), axis=1)

        for index in range(1, text_length):

            alpha = np.dot(W, h0) + np.dot(U, x0) + b
            h = np.tanh(alpha)
            o = np.dot(V, h) + c
            p = softmax(o)

            # Compute the cumulative sum of p and draw a random sample from [0,1)
            cumulative_sum = np.cumsum(p)
            draw_number = np.random.sample()

            # Find the element that corresponds to this random sample
            pos = np.where(cumulative_sum > draw_number)[0][0]

            # Create one-hot representation of the found position
            Y[pos, index] = 1.0

            h0 = np.copy(h)
            x0 = np.expand_dims(np.copy(Y[:, index]), axis=1)

        return Y

    def ComputeLoss(self, input_sequence, Y, weight_parameters):
        """
        Computes the cross-entropy loss of the RNN.

        :param input_sequence: The input sequence.
        :param weight_parameters: Weights and matrices of the RNN, which in particularly are:
            :param W: Hidden-to-Hidden weight matrix.
            :param U: Input-to-Hidden weight matrix.
            :param b: Bias vector of the hidden layer.
            :param V: Hidden-to-Output weight matrix.
            :param c: Bias vector of the output layer.

        :return: Cross entropy loss (divergence between the predictions and the true output)
        """

        p = self.ForwardPass(input_sequence, weight_parameters)[2]

        cross_entropy_loss = -np.log(np.diag(np.dot(Y.T, p))).sum()

        return cross_entropy_loss

    def ForwardPass(self, input_sequence, weight_parameters, h0 = None):
        """
        Evaluates the predictions that the RNN does in an input character sequence.

        :param input_sequence: The one-hot representation of the input sequence.
        :param weight_parameters: The weights and biases of the network, which are:
            :param W: Hidden-to-Hidden weight matrix.
            :param U: Input-to-Hidden weight matrix.
            :param b: Bias vector of the hidden layer.
            :param V: Hidden-to-Output weight matrix.
            :param c: Bias vector of the output layer.
        :param h0: Initial hidden state value.

        :return: The predicted character sequence based on the input one.
        """

        W, U, b, V, c = weight_parameters

        p = np.zeros(input_sequence.shape)
        if h0 is None:
            h_list = [np.zeros((self.m, 1))]
        else:
            h_list = [h0]

        a_list = []

        for index in range(0, input_sequence.shape[1]):

            alpha = np.dot(W, h_list[-1]) + np.dot(U, np.expand_dims(input_sequence[:,index], axis=1)) + b
            a_list.append(alpha)
            h = np.tanh(alpha)
            h_list.append(h)
            o = np.dot(V, h) + c
            p[:, index] = softmax(o).reshape(p.shape[0],)

        return a_list, h_list[1:], p

    def BackwardPass(self, x, Y, p, W, V, a, h, with_clipping=True):
        """
        Computes the gradient updates of the network's weight and bias matrices based on the divergence between the
        prediction and the true output.

        :param x: Input data to the network.
        :param p: Predictions of the network.
        :param Y: One-hot representation of the correct sequence.
        :param W: Hidden-to-Hidden weight matrix.
        :param V: Hidden-to-Output weight matrix.
        :param a: Hidden states before non-linearity.
        :param h: Hidden states of the network at each time step.
        :param with_clipping: (Optional) Set to False for not clipping in he gradients

        :return:  Gradient updates.
        """

        # grad_W, grad_U, grad_b, grad_V, grad_c = \
        #     np.zeros(W.shape), np.zeros((W.shape[0], x.shape[1])), np.zeros((W.shape[0], 1)), np.zeros(V), np.zeros((x.shape[0], 1))

        # Computing the gradients for the last time step

        grad_c = np.expand_dims((p[:, x.shape[1]- 1] - Y[:, x.shape[1]- 1]).T, axis=1)
        grad_V = np.dot(grad_c, h[-1].T)

        grad_h = np.dot(V.T, grad_c)

        grad_b = np.expand_dims(np.dot(grad_h.reshape((grad_h.shape[0],)), np.diag(1 - np.tanh(a[-1].reshape((a[-1].shape[0],))) ** 2)), axis=1)

        grad_W = np.dot(grad_b, h[-2].T)
        grad_U = np.dot(grad_b, np.expand_dims(x[:,-1], axis=0))

        grad_a = grad_b

        for time_step in reversed(range(x.shape[1]- 1)):

            grad_o = np.expand_dims((p[:, time_step] - Y[:, time_step]).T, axis=1)
            grad_V += np.dot(grad_o, np.transpose(h[time_step]))
            grad_c += grad_o

            grad_h = np.dot(V.T, grad_o) + np.dot(grad_a.T, W).T

            grad_a = np.expand_dims(np.dot(grad_h.reshape((grad_h.shape[0],)), np.diag(1 - np.tanh(a[time_step].reshape((a[time_step].shape[0],))) ** 2)), axis=1)

            grad_W += np.dot(grad_a, h[time_step-1].T)
            grad_U += np.dot(grad_a, np.expand_dims(x[:,time_step], axis=1).T)

            grad_b += grad_a

        gradients = [grad_W, grad_U, grad_b, grad_V, grad_c]

        if with_clipping:

            for index, elem in enumerate(gradients):

                gradients[index] = np.maximum(-5, np.minimum(elem, 5))

        return gradients

    def initialize_ada_grad(self, weight_parameters):
        """
        Initializes the ada_grads of the weights parameters.

        :param weight_parameters: The weights and biases of the RNN

        :return: Initialized ada-grad parameters.
        """

        ada_grads = []

        for elem in weight_parameters:

            ada_grads.append(np.zeros(shape=elem.shape))

        return ada_grads

    def ada_grad_update(self, weight_parameters, ada_grads, gradients, eta):
        """
        Conducts one update step of the ada-grants and weights parameters based on the currently estimated gradient updates.

        :param weight_parameters: Weights and biases of the RNN network.
        :param ada_grads: Ada grad parameters.
        :param gradients: Gradient updates of a training step.
        :param eta: Learning rate of the training process.

        :return: Updated weight and ada_grad parameters.
        """

        # Update ada-grads
        for index, elem in enumerate(ada_grads):

            elem += gradients[index] ** 2

        for index, weight_elem in enumerate(weight_parameters):

            weight_elem -= eta * gradients[index] / np.sqrt(ada_grads[index] + self.epsilon)

        return weight_parameters, ada_grads


    def fit(self, X, Y, epoches, unique_characters):
        """
        Comnducts the training pprocess of the RNN nad estimates the model.

        :param X: Input data (one-hot representation).
        :param Y: Treu labels (one-hot representation).
        :param epoches: Number of training epochs.
        :param unique_characters: The unique characters that can be generated from the training process.

        :return: The trained model.
        """

        weight_parameters = self.init_weights()
        gradient_object = Gradients(self)

        # Number of distinct training sequences per epoch
        training_sequences_per_epoch = X.shape[1] - self.seq_length

        ada_grads = self.initialize_ada_grad(weight_parameters)

        for epoch in range(epoches):

            hprev = np.zeros(shape=(self.m, 1))

            for e in range(training_sequences_per_epoch):

                current_update_step = epoch * training_sequences_per_epoch + e

                x = X[:, e:e + self.seq_length]
                y = Y[:, e + 1:e + self.seq_length + 1]

                gradient_updates, hprev = gradient_object.ComputeGradients(x, y, weight_parameters, hprev)

                weight_parameters, ada_grads = self.ada_grad_update(weight_parameters, ada_grads, gradient_updates, eta=self.eta)

                if epoch ==0 and e ==0 :
                    smooth_loss_evolution = [self.ComputeLoss(x, y, weight_parameters)]
                    minimum_loss = smooth_loss_evolution[0]
                else:
                    current_loss = 0.999 * smooth_loss_evolution[-1] + 0.001 * self.ComputeLoss(x, y, weight_parameters)
                    smooth_loss_evolution.append(current_loss)
                    if current_loss < minimum_loss:
                        best_weights = weight_parameters

                if len(smooth_loss_evolution) % 100 == 0 and len(smooth_loss_evolution) > 0:
                    print('---------------------------------------------------------')
                    print(f'Smooth loss at update step no.{current_update_step}: {smooth_loss_evolution[-1]}')

                    # Also generate synthesized text if 500 updates steps have been conducted
                    if len(smooth_loss_evolution) % 500 == 0 and len(smooth_loss_evolution) > 0:

                        synthesized_text = Ind_to_Char(self.synthesize_sequence(h0=hprev, x0=X, weight_parameters=weight_parameters, text_length=200), unique_characters)
                        print('---------------------------------------------------------')
                        print(f'Synthesized text of update step no.{current_update_step}')
                        print(''.join(synthesized_text))

        return best_weights, smooth_loss_evolution

class Gradients:

    def __init__(self, RNN):
        self.RNN = RNN

    def ComputeGradients(self, X, Y, weight_parameters, hprev, with_clipping=True):
        """
        Computes the analytical gradient updates of the network.
        :param X: Input sequence.
        :param Y: True output
        :param weight_parameters: Weights and bias matrices of the network.
        :param hprev: Initial hidden state to be used in the forward and backwward pass of the RNN.
        :param with_clipping: (Optional) Set ot False if you don't wish to apply clipping in the gradients.

        :return: Gradients updates.
        """

        a_list, h_list, p = self.RNN.ForwardPass(X, weight_parameters, h0=hprev)
        gradients = self.RNN.BackwardPass(X, Y, p, weight_parameters[0], weight_parameters[3], a_list, h_list, with_clipping)

        return gradients, h_list[0]

    def ComputeGradsNumSlow(self, X, Y, weight_parameters, h=1e-4):

        from tqdm import tqdm
        all_grads_num = []

        for index, elem in enumerate(weight_parameters):

            grad_elem = np.zeros(elem.shape)
            # h_prev = np.zeros((W.shape[1], 1))

            for i in tqdm(range(elem.shape[0])):
                for j in range(elem.shape[1]):

                    elem_try = np.copy(elem)
                    elem_try[i, j] -= h
                    all_weights_try = weight_parameters.copy()
                    all_weights_try[index] = elem_try
                    c1 = self.RNN.ComputeLoss(X, Y, weight_parameters=all_weights_try)

                    elem_try = np.copy(elem)
                    elem_try[i, j] += h
                    all_weights_try = weight_parameters.copy()
                    all_weights_try[index] = elem_try
                    c2 = self.RNN.ComputeLoss(X, Y, weight_parameters=all_weights_try)

                    grad_elem[i, j] = (c2-c1) / (2*h)

            all_grads_num.append(grad_elem)

        return all_grads_num

    def check_similarity(self, X, Y, weight_parameters, with_cliping = False):
        """
        Computes and compares the analytical and numerical gradients.

        :param X: Input sequence.
        :param Y: True output
        :param weight_parameters: Weights and bias matrices of the network.
        :param with_cliping: (Optional) Set to True to apply clipping in the gradients

        :return: None.
        """

        analytical_gradients = self.ComputeGradients(X, Y, weight_parameters)
        numerical_gradients = self.ComputeGradsNumSlow(X, Y, weight_parameters)

        for weight_index in range(len(analytical_gradients)):
            print('-----------------')
            print(f'Weight parameter no. {weight_index+1}:')

            weight_abs = np.abs(analytical_gradients[weight_index] - numerical_gradients[weight_index])

            weight_nominator = np.average(weight_abs)

            grad_weight_abs = np.absolute(analytical_gradients[weight_index])
            grad_weight_num_abs = np.absolute(numerical_gradients[weight_index])

            sum_weight = grad_weight_abs + grad_weight_num_abs

            print(f'Deviation between analytical and numerical gradients: {weight_nominator / np.amax(sum_weight)}')

def main():


    def syntesize_text():

        book_data, unique_characters = Load_Text_Data()

        rnn = RNN(m=100, K=len(unique_characters), eta=0.1, seq_length=25, std=0.1)

        weight_parameters = rnn.init_weights()

        input_sequence = book_data[:rnn.seq_length]

        integer_encoding = Char_to_Ind(input_sequence, unique_characters)
        input_sequence_one_hot = create_one_hot_endoding(integer_encoding, len(unique_characters))

        test = rnn.synthesize_sequence(h0=np.zeros((rnn.m, 1)), x0=input_sequence_one_hot, weight_parameters=weight_parameters)
        test2 = Ind_to_Char(test, unique_characters)
        print(''.join(test2))

    def compare__with_numericals():
        book_data, unique_characters = Load_Text_Data()

        rnn_object = RNN(m=5, K=len(unique_characters), eta=0.1, seq_length=25, std=0.01)
        gradient_object = Gradients(rnn_object)

        weight_parameters = rnn_object.init_weights()

        input_sequence = book_data[:rnn_object.seq_length]
        integer_encoding = Char_to_Ind(input_sequence, unique_characters)
        input_sequence_one_hot = create_one_hot_endoding(integer_encoding, len(unique_characters))

        output_sequence = book_data[1:1 + rnn_object.seq_length]
        output_encoding = Char_to_Ind(output_sequence, unique_characters)
        output_sequence_one_hot = create_one_hot_endoding(output_encoding, len(unique_characters))

        gradient_object.check_similarity(input_sequence_one_hot, output_sequence_one_hot, weight_parameters)

    def train_with_ada_grad():

        book_data, unique_characters = Load_Text_Data()

        rnn = RNN(m=100, K=len(unique_characters), eta=0.1, seq_length=25, std=0.1)

        # Create one-hot data
        integer_encoding = Char_to_Ind(book_data, unique_characters)
        input_sequence_one_hot = create_one_hot_endoding(integer_encoding, len(unique_characters))
        output_sequence_one_hot = create_one_hot_endoding(integer_encoding, len(unique_characters))

        weight_parameters = rnn.fit(X=input_sequence_one_hot, Y=output_sequence_one_hot, epoches=5, unique_characters=unique_characters)

    # syntesize_text()
    # compare__with_numericals()
    train_with_ada_grad()

    print('Finished!')

if __name__=='__main__':
    main()