import numpy
import random
import itertools

def random_weight(range_value):
    value = 0.0
    while(value == 0.0): #not really beautiful, but works
        value = random.uniform(-range_value,range_value) #TODO: ask Bruno what would be a good range of values
    return value

def generate_NN(input_size,output_size,hidden_layers_sizes,neurons_type="sigmoid"):
    neurons = {}
    connections = {}
    input_layer = [Neuron("bias")]
    input_layer += [Neuron(neurons_type) for _ in range(1,input_size+1)]
    neurons[0] = input_layer
    layer_no = 1
    for size in hidden_layers_sizes:
        layer = [Neuron("bias")]
        layer += [Neuron(neurons_type) for _ in range(1,size+1)]
        neurons[layer_no] = layer
        connections[layer_no-1] = {comb:random_weight(5) for comb in
                                   itertools.product(range(len(neurons[layer_no-1])),range(1,size+1))}
        layer_no += 1
    output_layer = [Neuron(neurons_type) for _ in range(1,output_size+1)] #output layer doesn't have a bias neuron
    neurons[layer_no] = output_layer
    connections[layer_no - 1] = {comb: random_weight(5) for comb in
                                 itertools.product(range(len(neurons[layer_no - 1])), range(1, output_size + 1))}

    return neurons,connections

'''
neuron(l,i) = neurons[l][i] [by definition, if i=0, it is a BIAS neuron]
weight(l,i,j) = connections[l][(i,j)]
'''
class NeuralNet():
    def __init__(self,input_size,output_size,hidden_layers_sizes,alpha=0.0001,regularization=False):
        self.neurons, self.connections = generate_NN(input_size,output_size,hidden_layers_sizes)
        self.alpha = alpha
        self.regularization = regularization

'''
type = bias (always return 1.0), sigmoid (1/1+exp(-x)) or ReLU (max(0,x))
(no need to calc delta for bias neurons)
'''
class Neuron():
    def __init__(self,type):
        self.type = type