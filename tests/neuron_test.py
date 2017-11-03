import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import unittest
import neural_net
from numpy import exp


class TestNeuronMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_activation_sigmoid(self):

        def fun(x):
            return 1.0 / (1.0 + exp(-x))

        a_neuron = neural_net.Neuron('sigmoid')
        self.assertEqual(a_neuron.activation([1]), fun(1))
        self.assertEqual(a_neuron.activation([1, 2, 3]), fun(6))

        with self.assertRaises(ValueError) as context:
            a_neuron.activation([])

    def test_activation_Relu(self):

        def fun(x):
            return max(0.0, x)

        a_neuron = neural_net.Neuron('ReLU')
        self.assertEqual(a_neuron.activation([1]), fun(1))
        self.assertEqual(a_neuron.activation([1, 2, 3]), fun(6))
        self.assertEqual(a_neuron.activation([-2]), 0)

    def test_activation_Bias(self):

        def fun(x):
            return max(0.0, x)

        a_neuron = neural_net.Neuron('bias')
        self.assertEqual(a_neuron.activation([]), 1)
        self.assertEqual(a_neuron.activation([1, 2, 3]), 1)
        self.assertEqual(a_neuron.activation([-2]), 1)

    def test_activation_bias(self):
        pass

if __name__ == '__main__':
    unittest.main()