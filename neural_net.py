import numpy as np
import itertools
from copy import deepcopy
from collections import OrderedDict


def random_weight():
    """
    :return: float, number generated by np.random.normal, excluding 0.0
    """
    #parameters of np.random.normal (suggested by Bruno to be small values close to 0.0)
    average = 0.0
    stddev = 0.15

    value = 0.0
    while (value == 0.0):  # not really beautiful, but works
        value = np.random.normal(average, stddev)
    return value


class NeuralNet(object):
    """
    neuron(l,i) = neurons[l][i] [by definition, if i=0, it is a BIAS neuron]
    weight(l,i,j) = connections[l][(i,j)]
    """

    def __init__(self, input_size, output_size, hidden_layers_sizes, neurons_type='sigmoid', alpha=0.0001,
                 lamb=0.0):
        """
        :param input_size: int, size of an input
        :param output_size: int, size of an output
        :param hidden_layers_sizes: list of int, size of each hidden layer
        :param neurons_type: 'bias', 'sigmoid' or 'ReLU'
        :param alpha: float, alpha value
        :param lamb: float, lambda value (0.0 = no regularization)
        """
        self.neurons, self.connections = self._generate_NN(input_size, output_size, hidden_layers_sizes, neurons_type)
        self.alpha = alpha
        self.lamb = lamb
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.neuron_type = neurons_type

    def predict(self, nn_input: list) -> (list, list):
        """
        :param nn_input: list of float (EXCLUDING CONSTANT FEATURE)
        :return: list of float, the predicted values and all neuron activations
        """

        if len(nn_input) != self.input_size:
            raise ValueError("This neural network requires a list of %d elements, but %d were given" %
                             (self.input_size, len(nn_input)))

        activations = deepcopy(nn_input)
        activations = [1] + activations

        all_activations = []

        for layer_no, layer in enumerate(self.neurons):
            new_activations = []
            for neuron_no, neuron in enumerate(layer):
                activation_input = [a * w for a, w in zip(activations, self.connections[layer_no][neuron_no])]
                new_activations.append(neuron.activation(activation_input))
            all_activations.append(new_activations)
            activations = new_activations

        return activations, all_activations

    def back_propagation(self, nn_input: list, excpected_list: list):  # TODO
        """
        :list error: the error of the NN, calculated outside
        probably will need to create smaller auxiliary function
        """

        prediction_list, activations = self.predict(nn_input)

        all_deltas = []
        current_deltas = [prediction - excpected for prediction, excpected in zip(prediction_list, excpected_list)]
        all_deltas.append(current_deltas)

        for layer_no, layer in reversed(list(enumerate(self.neurons))):
            new_deltas = []
            for neuron_no, neuron in layer:
                delta = sum([d * w[neuron_no] for d, w in zip(current_deltas, self.connections[layer_no])])\
                             * neuron.last_activation * (1 - neuron.last_activation)
                new_deltas.append(delta)
            all_deltas.append(new_deltas)
            current_deltas = new_deltas
        pass

    @staticmethod
    def _generate_NN(input_size, output_size, hidden_layers_sizes, neurons_type='sigmoid'):
        # TODO transform this in a private method of NeuralNet
        """
        :param input_size: int, size of an input
        :param output_size: int, size of an output
        :param hidden_layers_sizes: list of int, size of each hidden layer
        :param neurons_type: 'bias', 'sigmoid' or 'ReLU'
        :return: neurons dict and connections dict
        """
        neurons = []
        connections =[]
        input_layer = [Neuron('bias')]
        input_layer += [Neuron(neurons_type) for _ in range(1, input_size + 1)]
        neurons.append(input_layer)

        for hidden_layer_size in hidden_layers_sizes:
            hidden_layer = [Neuron('bias')]
            hidden_layer += [Neuron(neurons_type) for _ in range(0, hidden_layer_size)]
            neurons.append(hidden_layer)

        neurons.append([Neuron(neurons_type) for _ in range(0, output_size)])

        #Iterates all layers except first because it is the input layer.
        for layer_no, layer in enumerate(neurons[1:], start=1):
            neurons_connections = []
            for _ in layer:
                a_neuron_connection = [random_weight() for _ in neurons[layer_no-1]]
                neurons_connections.append(a_neuron_connection)
            connections.append(neurons_connections)

        return neurons[1:], connections


class Neuron(object):
    def __init__(self, type):
        """
        :param type: 'bias' (always return 1.0), 'sigmoid' (1/1+exp(-x)) or 'ReLU' (max(0,x))
        (no need to calc delta for bias neurons)
        """
        self.type = type.upper()
        self.last_activation = 0

    def activation(self, input_values=()):
        # if BIAS neuron, won't have input_values (input_values = weights * activations)
        """
        :param input_values: list of float, each value correspond to weight(i) * activation(i) (already calculated by the caller)
        :return: float, the activation output value
        """
        if self.type == 'BIAS':
            self.last_activation = 1
            return 1.0
        else:
            if input_values == [] or input_values == ():
                raise ValueError('Input for sigmoid and ReLu can\'t be empty')

            x = sum(input_values)
            if self.type == 'SIGMOID':
                self.last_activation = 1.0 / (1.0 + np.exp(-x))
                return self.last_activation
            elif self.type == 'RELU':
                self.last_activation = max(0.0, x)
                return self.last_activation

    @staticmethod
    def delta(weights: list, deltas: list, activation: float):
        """
        WILL need input values (the weights, deltas and activation) (slide 56 aula 11)
        :return: the delta value (to calc the gradients to att the weights)
        """
        return sum([w * d for w, d in zip(weights, deltas)]) * activation * (1 - activation)
