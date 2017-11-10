from neural_net import NeuralNet
from statistics import mean, stdev
from random import shuffle
import argparse
import csv
from argv_parser import parser
import pickle


def normalize(value, oldmin, oldmax, newmin, newmax):  # will use to put everything between 0 and 1
    newvalue = (((float(value) - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin
    return newvalue


def read_dataset(dataset):
    '''
    :param filename: string, the name of the dataset
    :return: dict of instances {array_of_inputs : expected_output_values}
    '''
    data = {}
    if dataset == 'survival':
        # age: 30-83
        # year: 58-69
        # aux_nodes: 0-52.0
        # survival: 1-2 (output) {transformed into probability}
        min_values = [30.0, 58.0, 0.0, 1.0]
        max_values = [83.0, 69.0, 52.0, 2.0]
        with open("datasets/haberman/haberman.data") as hab_file:
            hab_reader = csv.reader(hab_file)
            for row in hab_reader:
                instance = [normalize(row[i], min_values[i], max_values[i], 0.0, 1.0) for i in range(len(row))]
                input_values = tuple(instance[:-1])
                output_values = instance[-1]
                data[input_values] = output_values

    elif dataset == 'wine':
        # class: 1-2-3 (output) {transformer into probabilities}
        # alcohol: 11.03-14.83
        # malic: 0.74-5.8
        # ash: 1.36-3.23
        # alcal: 10.6-30.0
        # magnes: 70.0-162.0
        # phenols: 0.98-3.88
        # flavan: 0.34-5.08
        # n-flavan: 0.13-0.66
        # prc: 0.41-3.58
        # color: 1.28-13.0
        # hue: 0.48-1.71
        # od: 1.27-4.0
        # proline: 278.0-1680.0
        min_values = [1.0, 11.03, 0.74, 1.36, 10.6, 70.0, 0.98, 0.34, 0.13, 0.14, 1.28, 0.48, 1.27, 278.0]
        max_values = [3.0, 14.83, 5.80, 3.23, 30.0, 162.0, 3.88, 5.08, 0.66, 3.58, 13.0, 1.71, 4.0, 1680.0]
        convertion = {'1': [1.0, 0.0, 0.0], '2': [0.0, 1.0, 0.0], '3': [0.0, 0.0, 1.0]}
        with open("datasets/wine/wine.data") as wine_file:
            wine_reader = csv.reader(wine_file)
            for row in wine_reader:
                instance = [convertion[row[i]] if i == 0 else normalize(row[i], min_values[i], max_values[i], 0.0, 1.0)
                            for i in range(len(row))]
                input_values = tuple(instance[1:])
                output_values = instance[0]
                data[input_values] = output_values

    elif dataset == 'contraceptive':
        # wage: 16.0-49.0
        # weduc: 1.0-4.0
        # heduc: 1.0-4.0
        ##child: 0.0-16.0
        # wrelig: 0.0-1.0
        # wwork: 0.0-1.0
        # hoccup: 1.0-4.0
        # sol: 1.0-4.0
        # mexp: 0.0-1.0
        # method: 1-2-3 (output) {transformer into probabilities}
        min_values = [16.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        max_values = [49.0, 4.0, 4.0, 16.0, 1.0, 1.0, 4.0, 4.0, 1.0, 3.0]
        convertion = {'1': [1.0, 0.0, 0.0], '2': [0.0, 1.0, 0.0], '3': [0.0, 0.0, 1.0]}
        with open("datasets/cmc/cmc.data") as cmc_file:
            cmc_reader = csv.reader(cmc_file)
            for row in cmc_reader:
                instance = [convertion[row[i]] if i == 9 else normalize(row[i], min_values[i], max_values[i], 0.0, 1.0)
                            for i in range(len(row))]
                input_values = tuple(instance[:9])
                output_values = instance[9]
                data[input_values] = output_values

    elif dataset == 'cancer':
        # diagnosis: M-B (output) {transformed into probability}
        # features: calculated previously (too much to list)
        min_values = [0.0, 6.981, 9.71, 43.79, 143.5, 0.05263, 0.01938, 0.0, 0.0, 0.106, 0.04996, 0.1115, 0.3602, 0.757,
                      6.802, 0.001713, 0.002252, 0.0, 0.0, 0.007882, 0.0008948, 7.93, 12.02, 50.41, 185.2, 0.07117,
                      0.02729, 0.0, 0.0, 0.1565, 0.05504]
        max_values = [1.0, 28.11, 39.28, 188.5, 2501.0, 0.1634, 0.3454, 0.4268, 0.2012, 0.304, 0.09744, 2.873, 4.885,
                      21.98, 542.2, 0.03113, 0.1354, 0.396, 0.05279, 0.07895, 0.02984, 36.04, 49.54, 251.2, 4254.0,
                      0.2226, 1.058, 1.252, 0.291, 0.6638, 0.2075]
        convertion = {'M': 0.0, 'B': 1.0}
        with open("datasets/breast-cancer-wisconsin/wdbc.data") as cancer_file:
            cancer_reader = csv.reader(cancer_file)
            for row in cancer_reader:
                instance = [convertion[row[i]]
                            if i == 1
                            else normalize(row[i],
                                           min_values[i - 1],
                                           max_values[i - 1], 0.0, 1.0) for i in range(1, len(row))]
                input_values = tuple(instance[1:])
                output_values = instance[0]
                data[input_values] = output_values

    return data


def holdout(dataset, percentage_test):  # TODO: we need to know how will be the dataset (there are classes? how much?)
    '''
    :param dataset: the full dataset
    :param percentage_test: float, percentage of instances that needs to go to the test partition
    :return: (dict_of_train_instances, dict_of_test_instances)
    '''
    pass


def cross_validation(dataset, percentage_test, iterations, mode,
                     hidden_layers_sizes, neurons_type='sigmoid', alpha=0.0001, lamb=0.0):
    '''
    :param dataset: the full dataset
    :param percentage_test: float, percentage of instances that needs to go to the test partition
    :param iterations: int, number of holdouts to execute
    should call holdout to generate the train and test dicts
    should call train_NN to train the dataset
    should call test_NN to get the performances
    :return: (average_performance, stddev_performance)
    '''
    results = []
    for it in range(1, iterations + 1):
        train_dataset, test_dataset = holdout(dataset, percentage_test)
        input_size = len(train_dataset.keys()[0])
        output_size = len(train_dataset[train_dataset.keys()[0]])
        nn = NeuralNet(input_size, output_size, hidden_layers_sizes, neurons_type, alpha, lamb)
        train_NN(nn, train_dataset, mode)
        performance = test_NN(nn, test_dataset)
        results.append(performance)

    return (mean(results), stdev(results))


def train_NN(NN, train_instances, mode):
    '''
    :param NN: the neural net
    :param train_instances: the train instances dict
    :param mode: "mini_batch" (call back_propagation for each instance) or "batch" (call back_propagation after all instances)
    '''
    number_of_instances = len(train_instances.keys())
    if mode == "batch":
        while True:
            # accumulate the errors
            error_acc = 0.0
            for instance in train_instances.keys():
                output = NN.predict(instance)
                error_acc += (output - train_instances[instance]) / number_of_instances
            # call back_propagation
            NN.back_propagation(error_acc)

            if __stop_conditions:  # TODO: we need to talk about what will be the stop condition(s) (ask Bruno too)
                break

    elif mode == "mini_batch":
        while True:
            for instance in train_instances.keys():
                output = NN.predict(instance)
                error = (output - train_instances[instance])
                # call back_propagation
                NN.back_propagation(error)

            if __stop_conditions:  # TODO: we need to talk about what will be the stop condition(s) (ask Bruno too)
                # the stop condition should consider a full-batch measure, even if it is mini_batch?
                break


def test_NN(NN, test_instances):
    '''
    :param NN: the neural net
    :param test_instances: the test instances dict
    :return: the performance of the NN in test_mode
    only implemented after we know what will be the dataset (predict a category or a number? this is relevant)
    '''
    pass


if __name__ == "__main__":
    mode_parser, mode_argument_parses = parser()

    if str(mode_parser.mode) == 'create_net':
        print(mode_argument_parses.o)
        output_directory = str(mode_argument_parses.o[0])
        print(output_directory)
        input_size = int(mode_argument_parses.i[0])
        output_size = int(mode_argument_parses.output_size[0])
        hidden_layer_sizes = [int(layer_size) for layer_size in mode_argument_parses.l]
        alpha = float(mode_argument_parses.a[0])
        gamma = float(mode_argument_parses.g[0])
        neuron_type = mode_argument_parses.t[0]

        a_neural_net = NeuralNet(input_size, output_size, hidden_layer_sizes, neuron_type, alpha, gamma)

        try:
            with open(output_directory,'wb') as output:
                pickle.dump(a_neural_net,output,pickle.HIGHEST_PROTOCOL)
                print("Neural Network created and saved at %s" % output_directory)
                exit()
        except EnvironmentError:
            print("Can't create file '%s'" % output_directory)
            exit()


