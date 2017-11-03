import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import unittest
import neural_net
from numpy import exp
from pprint import pprint


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        pass

    def test_build(self):
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

    def test_activation(self):
        # TODO do math to compare predictions
        pass

if __name__ == '__main__':
    unittest.main()
