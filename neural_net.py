import numpy as np
import itertools
from copy import deepcopy
from collections import OrderedDict
from math import log
from pprint import pprint
from time import sleep
from random import uniform
from math import isclose

def random_weight():
    """
    :return: float, number generated by np.random.normal, excluding 0.0
    """
    #parameters of np.random.normal (suggested by Bruno to be small values close to 0.0)
    average = 0.0
    stddev = 0.15

    value = 0.0
    while value == 0.0:  # not really beautiful, but works
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
        self.neurons, self.connections = self._generate_nn(input_size, output_size, hidden_layers_sizes, neurons_type)
        self.connections = np.array(self.connections)
        self.alpha = alpha
        self.lamb = lamb
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.neuron_type = neurons_type
        self.old_gradient_list = []
        self.old_connections = deepcopy(self.connections)
        self.last_prediction = []
        self.last_expected = []

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
            temp = []
            for neuron_no, neuron in enumerate(layer):
                activation_input = [a * w for a, w in zip(activations, self.connections[layer_no][neuron_no])]
                temp.append(activation_input)
                new_activations.append(neuron.activation(activation_input))

            all_activations.append(new_activations)
            activations = new_activations

        return activations, all_activations

    def back_propagation(self, nn_input: list, expected_list: list):
        """
        :list error: the error of the NN, calculated outside
        remember to use self.regularization to see if need to use regularization or not
        probably will need to create smaller auxiliary function
        """

        prediction_list, activations = self.predict(nn_input)
        self.last_prediction = prediction_list
        self.last_expected = expected_list

        error = self.calculate_error(prediction_list, expected_list)

        error += self.regression_calc(self.connections,self.lamb)

        reverse_delta_list = self._reverse_delta_list(prediction_list, expected_list)

        gradient_list = self._gradient_list(reverse_delta_list, nn_input)

        self.old_gradient_list = gradient_list

        self._update_connections(gradient_list, error)

        return error

    @staticmethod
    def regression_calc(connection_list, lamb):
        size_counter = 0
        connection_sum = 0
        for layer in connection_list:
            for neuron in layer[1:]:
                for connection in neuron:
                    connection_sum += connection
                    size_counter += 1
        return (lamb / (2 * size_counter)) * connection_sum

    def _update_connections(self, gradient_list, error):
        for layer_no, _ in enumerate(self.connections):
            self.connections[layer_no] =\
                np.subtract(self.connections[layer_no], np.array(gradient_list[layer_no]) * self.alpha * error)

            
    def _gradient_list(self, reverse_delta_list, nn_input):

        delta_list = reverse_delta_list[::-1]
        gradient_list = []

        first_layer = [1] + nn_input

        temp_list = []
        for neuron_no, connections in enumerate(self.connections[0]):
            temp_list.append([first_layer[connection_no] * delta_list[0][neuron_no] for connection_no, a_connection in enumerate(connections)])
        gradient_list.append(temp_list)

        for layer_no, connections_layer in list(enumerate(self.connections))[1:]:
            temp_layer = []
            for neuron_no, connections in enumerate(connections_layer):
                temp_connection_list = []
                for connection_no, a_connection in enumerate(connections):
                    temp_connection_list.append((self.neurons[layer_no-1][connection_no].last_activation *
                                                delta_list[layer_no][neuron_no]) +
                                                (self.lamb * self.connections[layer_no][neuron_no][connection_no]))

                temp_layer.append(temp_connection_list)
            gradient_list.append(temp_layer)

        return np.array(gradient_list)

    def _reverse_delta_list(self, prediction_list, expected_list):

        all_deltas = []
        current_deltas = [prediction - expected for prediction, expected in zip(prediction_list, expected_list)]
        all_deltas.append(current_deltas)

        for layer_no, layer in reversed(list(enumerate(self.neurons[:-1]))):
            test_deltas = np.dot(np.transpose(self.connections[layer_no+1]), current_deltas)
            activation = np.array([neuron.last_activation for neuron in layer])

            new_deltas = test_deltas*activation*(1-activation)

            all_deltas.append(new_deltas)
            current_deltas = new_deltas

        return all_deltas

    @staticmethod
    def _generate_nn(input_size, output_size, hidden_layers_sizes, neurons_type='sigmoid'):
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

        # Iterates all layers except first because it is the input layer.
        for layer_no, layer in enumerate(neurons[1:], start=1):
            neurons_connections = []
            for _ in layer:
                a_neuron_connection = [random_weight() for _ in neurons[layer_no-1]]
                neurons_connections.append(a_neuron_connection)
            connections.append(neurons_connections)

        return neurons[1:], connections

    @staticmethod
    def calculate_error(predicted: list, expected: list) -> float:
        try:
            return sum([(-y * (log(f))) - ((1 - y)*(log(1 - f))) for y, f in zip(expected, predicted)])
        except:
            print(predicted, expected)
            raise

    def gradient_verification(self, delta_layer_no, delta_neuron_no, delta_no, nn_input, expected_list, epsilon=0.001):
        """
        Inefficient method used for testing the gradient after a backpropagation call
        :param delta_layer_no: 
        :param delta_neuron_no: 
        :param delta_no: 
        :param nn_input: 
        :param expected_list: 
        :param epsilon: 
        :return: 
        """
        gradients = deepcopy(self.old_gradient_list)
        current_connections = deepcopy(self.connections)
        old_connection_with_lower_delta = deepcopy(self.old_connections)
        old_connection_with_lower_delta[delta_layer_no][delta_neuron_no][delta_no] -= epsilon

        old_connection_with_higher_delta = deepcopy(self.old_connections)
        old_connection_with_higher_delta[delta_layer_no][delta_neuron_no][delta_no] += epsilon

        self.connections = self.old_connections
        prediction_current_connection, _ = self.predict(nn_input)

        self.connections = old_connection_with_lower_delta
        prediction_lower_delta, _ = self.predict(nn_input)

        self.connections = old_connection_with_higher_delta
        prediction_higher_delta, _ = self.predict(nn_input)

        self.connections = current_connections

        # error_current = self.calculate_error(prediction_current_connection, expected_list)
        error_lower = self.calculate_error(prediction_lower_delta, expected_list)
        error_higher = self.calculate_error(prediction_higher_delta, expected_list)

        return gradients[delta_layer_no][delta_neuron_no][delta_no], (error_higher - error_lower)/(2*epsilon)


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


def verify(neural_net: NeuralNet):
    nn_input = [uniform(0, 1) for _ in range(0,neural_net.input_size)]
    nn_output = [uniform(0, 1) for _ in range(0,neural_net.output_size)]


    neural_net.back_propagation(nn_input, nn_output)

    print('Validating delta via numeric aproximation with error margin of 1%')
    for connectons_layer_no, connections_layer in enumerate(neural_net.connections):
        for neuron_connections_no, neuron_connections in enumerate(connections_layer):
            for connection_no, connection in enumerate(neuron_connections):
                result = neural_net.gradient_verification(connectons_layer_no, neuron_connections_no, connection_no,
                                                          nn_input, nn_output)
                print("Connection(%d,%d,%d)\nBackpropagation Value: %.10f\nNumeric Aprox Value:   %.10f\nWithin error "
                     "margin? %s\n"%
                     (connectons_layer_no, neuron_connections_no, connection_no, result[0], result[1],
                      str(isclose(*result, rel_tol=0.01))))


    pass