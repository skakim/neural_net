import numpy
import random
import itertools


def random_weight(range_value):
    """
    :param range_value: float, limits of random.uniform
    :return: float, number between (-range_value,range_value), excluding 0.0
    """
    value = 0.0
    while (value == 0.0):  # not really beautiful, but works
        value = random.uniform(-range_value, range_value)  # TODO: ask Bruno what would be a good range of values
    return value


def generate_NN(input_size, output_size, hidden_layers_sizes, neurons_type='sigmoid'):
    """
    :param input_size: int, size of an input
    :param output_size: int, size of an output
    :param hidden_layers_sizes: list of int, size of each hidden layer
    :param neurons_type: 'bias', 'sigmoid' or 'ReLU'
    :return: neurons dict and connections dict
    """
    neurons = {}
    connections = {}
    input_layer = [Neuron('bias')]
    input_layer += [Neuron(neurons_type) for _ in range(1, input_size + 1)]
    neurons[0] = input_layer
    layer_no = 1
    for size in hidden_layers_sizes:
        layer = [Neuron('bias')]
        layer += [Neuron(neurons_type) for _ in range(1, size + 1)]
        neurons[layer_no] = layer
        connections[layer_no - 1] = {comb: random_weight(5) for comb in
                                     itertools.product(range(len(neurons[layer_no - 1])), range(1, size + 1))}
        layer_no += 1
    output_layer = [Neuron(neurons_type) for _ in range(1, output_size + 1)]  # output layer doesn't have a bias neuron
    neurons[layer_no] = output_layer
    connections[layer_no - 1] = {comb: random_weight(5) for comb in
                                 itertools.product(range(len(neurons[layer_no - 1])), range(1, output_size + 1))}

    return neurons, connections


class NeuralNet(object):
    """
    neuron(l,i) = neurons[l][i] [by definition, if i=0, it is a BIAS neuron]
    weight(l,i,j) = connections[l][(i,j)]
    """

    def __init__(self, input_size, output_size, hidden_layers_sizes, neurons_type='sigmoid', alpha=0.0001,
                 regularization=False):
        """
        :param input_size: int, size of an input
        :param output_size: int, size of an output
        :param hidden_layers_sizes: list of int, size of each hidden layer
        :param neurons_type: 'bias', 'sigmoid' or 'ReLU'
        :param alpha: float, alpha value
        :param regularization: bool, True if want to use Regularization, False if not
        """
        self.neurons, self.connections = generate_NN(input_size, output_size, hidden_layers_sizes, neurons_type)
        self.alpha = alpha
        self.regularization = regularization

    def train(self, input, expected_output):  # TODO
        """
        :param input: list of float, ONE input
        :param expected_output: float, the expected output for this input
        the idea is this first see the predict output of the NN, then aplicates back_propagation
        """
        pass

    def predict(self, input):  # TODO
        """
        :param input: list of float, ONE input
        :return: float, the predicted value
        """
        pass

    def back_propagation(self, expected_output, output):  # TODO
        """
        :param expected_output: float, the expected value it should have predicted
        :param output: float, the value the NN predicted
        remember to use self.regularization to see if need to use regularization or not
        probably will need to create smaller auxiliary function
        """
        pass


class Neuron(object):
    def __init__(self, type):
        """
        :param type: 'bias' (always return 1.0), 'sigmoid' (1/1+exp(-x)) or 'ReLU' (max(0,x))
        (no need to calc delta for bias neurons)
        """
        self.type = type.upper()

    def activation(self,
                   input_values=()):  # if BIAS neuron, won't have input_values (input_values = weights * activations)
        """
        :param input_values: list of float, each value correspond to weight(i) * activation(i) (already calculated by the caller)
        :return: float, the activation output value
        """
        if self.type == 'BIAS':
            return 1.0
        else:
            if input_values == [] or input_values == ():
                raise ValueError('Input for sigmoid and ReLu can\'t be empty')

            x = sum(input_values)
            if self.type == 'SIGMOID':
                return 1.0 / (1.0 + numpy.exp(-x))
            elif self.type == 'RELU':
                return max(0.0, x)

    @staticmethod
    def delta(weights: list, deltas: list, activation: float):
        """
        WILL need input values (the weights, deltas and activation) (slide 56 aula 11)
        :return: the delta value (to calc the gradients to att the weights)
        """
        return sum([weights[i] * deltas[i] for i in range(0, len(weights))]) * activation * (1 - activation)
