import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import unittest
import neural_net
from math import isclose
from numpy import exp
from pprint import pprint


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        pass

    def test_build(self):

        #TODO REMOVE THIS RETURN
        return

        network = neural_net.NeuralNet(3, 1, [2, 4])

        # Checks if layers sizes are correct
        self.assertEqual(len(network.neurons[0]), 2+1)
        self.assertEqual(len(network.neurons[1]), 4+1)

        # Checks if connections sizes are correct
        self.assertEqual(len(network.connections[0]), 2+1)
        self.assertEqual(len(network.connections[1]), 4+1)
        self.assertEqual(len(network.connections[2]), 1)

        # Check connections from Hidden Layer 1
        self.assertEqual(len(network.connections[0][0]), 3 + 1)
        self.assertEqual(len(network.connections[0][1]), 3 + 1)
        self.assertEqual(len(network.connections[0][2]), 3 + 1)

        # Check connections from Hidden Layer 2
        self.assertEqual(len(network.connections[1][0]), 2 + 1)
        self.assertEqual(len(network.connections[1][1]), 2 + 1)
        self.assertEqual(len(network.connections[1][2]), 2 + 1)
        self.assertEqual(len(network.connections[1][3]), 2 + 1)

        # Check connections from output layer
        self.assertEqual(len(network.connections[2][0]), 4 + 1)

        pass

    def test_gradient(self):
        network = neural_net.NeuralNet(3, 1, [2, 4])
        network.back_propagation([1, 2, 3], [1])

        for layer_no, layer in enumerate(network.old_gradient_list):
            for neuron_no, gradients in enumerate(layer):
                for gradient_no, _ in enumerate(gradients):
                    self.assertTrue(isclose(*network.gradient_verification(layer_no,neuron_no,gradient_no,[1,2,3],[1]),
                                            rel_tol=0.01))

if __name__ == '__main__':
    unittest.main()
