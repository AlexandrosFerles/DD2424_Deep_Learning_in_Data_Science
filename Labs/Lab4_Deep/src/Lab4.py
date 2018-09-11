import numpy as np
import matplotlib.pyplot as plt

def main():

    print('Finished!')

def Load_Text_Data(file_path='../goblet_book.txt'):
    """
    Reads the input data.

    :param file_path: (Optional) Position of the txt file in the local system.

    :return: book_data: all input characters and unique_characters: unique single characters of the input data.
    """
    book_data = open(file_path, 'r').read()
    unique_characters = list(set(book_data))

    return book_data, unique_characters

def Char_to_Ind(unique_chars):
    """
    Maps the original characters to integers.

    :param unique_chars: The set of unique characters available.

    :return: A list of integers taht correspond to the characters.
    """

    return [index for index, _ in enumerate(unique_chars)]

def Ind_to_Char(integer_list, unique_chars_list):
    """
    Maps a list of integers to thei corresponding characters,

    :param integer_list: A list of integer representation of a character sequence.
    :param unique_chars_list: The list of unique characters.

    :return: The actual character sequence.
    """

    return [unique_chars_list[int(elem)] for elem in integer_list]

class RNN:
    """
    Recurrent Neural Network object
    """

    def __init__(self, m=100, K=10, eta=0.1, seq_length=25, std=0.01):
        """
        Initial setting of the RNN.

        :param m: Dimensionality of the hidden state.
        :param K: Number of unique classes to identify.
        :param eta: The learning rate of the training process.
        :param seq_length: The length of the input sequence.
        :param std: the variance of the normal distribution that initializes the weight matrices.
        """

        self.m = m
        self.K = K
        self.eta = eta
        self.seq_length = seq_length
        self.std = std

    def init_weights(self):
        """
        Initializes the weights and bias matrices
        """

        U = np.random.randn(self.m, self.K) * self.std
        W = np.random.randn(self.m, self.m) * self.std
        V = np.random.randn(self.K, self.m) * self.std

        b = np.zeros(self.m, 1)
        c = np.zeros(self.m, 1)

    def synthesize_sequence(self, h0, x0, seq_length, W, U, b, V, c):
        """
        Synthesizes a sequence of characters under the RNN values.

        :param self: The RNN.
        :param h0: Hidden state at time 0.
        :param x0: First (dummy) input vector of the RNN.
        :param seq_length: Length of the sequence that we wish to generate.
        :param W: Hidden-to-Hidden weight matrix.
        :param U: Input-to-Hidden weight matrix.
        :param b: Bias vector of the hidden layer.
        :param V: Hidden-to-Output weight matrix.
        :param c: Bias vector of the output layer.


        :return: Synthesized text through feed-forward process of the RNN.
        """

        alpha = np.dot(self., h0) + np.dot(U, x0) + b
        h = np.tanh(alpha)
        o = np.dot(V, h) + c
        p = softmax(p)

        return p






if __name__=='__main__':
    book_data, unique_characters = Load_Text_Data()

    integers = Char_to_Ind(unique_characters)
    lst = [2, 3, 5]
    chars = Ind_to_Char(lst, unique_characters)

    main()
