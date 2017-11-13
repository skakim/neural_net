from neural_net import NeuralNet, verify
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
    """
    :param filename: string, the name of the dataset
    :return: dict of instances {array_of_inputs : expected_output_values}
    """
    data = []
    if dataset == 'survival':
        # age: 30-83
        # year: 58-69
        # aux_nodes: 0-52.0
        # survival: 1-2 (output) {transformed into probability}
        min_values = [30.0, 58.0, 0.0, 1.0]
        max_values = [83.0, 69.0, 52.0, 2.0]
        data.append([])
        data.append([])
        with open("datasets/haberman/haberman.data") as hab_file:
            hab_reader = csv.reader(hab_file)
            for row in hab_reader:
                instance = [normalize(row[i], min_values[i], max_values[i], 0.0, 1.0) for i in range(len(row)-1)]
                data[int(row[-1])-1].append(instance)

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
        data.append([])
        data.append([])
        data.append([])
        with open("datasets/wine/wine.data") as wine_file:
            wine_reader = csv.reader(wine_file)
            for row in wine_reader:
                instance = [normalize(row[i], min_values[i], max_values[i], 0.0, 1.0) for i in range(1,len(row))]
                data[int(row[0])-1].append(instance)

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
        data.append([])
        data.append([])
        data.append([])
        with open("datasets/cmc/cmc.data") as cmc_file:
            cmc_reader = csv.reader(cmc_file)
            for row in cmc_reader:
                instance = [normalize(row[i], min_values[i], max_values[i], 0.0, 1.0) for i in range(len(row)-1)]
                data[int(row[-1])-1].append(instance)

    elif dataset == 'cancer':
        # diagnosis: M-B (output) {transformed into probability}
        # features: calculated previously (too much to list)
        min_values = [0.0, 6.981, 9.71, 43.79, 143.5, 0.05263, 0.01938, 0.0, 0.0, 0.106, 0.04996, 0.1115, 0.3602, 0.757,
                      6.802, 0.001713, 0.002252, 0.0, 0.0, 0.007882, 0.0008948, 7.93, 12.02, 50.41, 185.2, 0.07117,
                      0.02729, 0.0, 0.0, 0.1565, 0.05504]
        max_values = [1.0, 28.11, 39.28, 188.5, 2501.0, 0.1634, 0.3454, 0.4268, 0.2012, 0.304, 0.09744, 2.873, 4.885,
                      21.98, 542.2, 0.03113, 0.1354, 0.396, 0.05279, 0.07895, 0.02984, 36.04, 49.54, 251.2, 4254.0,
                      0.2226, 1.058, 1.252, 0.291, 0.6638, 0.2075]
        data.append([])
        data.append([])
        with open("datasets/breast-cancer-wisconsin/wdbc.data") as cancer_file:
            cancer_reader = csv.reader(cancer_file)
            for row in cancer_reader:
                instance = [normalize(row[i], min_values[i - 1], max_values[i - 1], 0.0, 1.0) for i in range(2, len(row))]
                if row[1]=='M':
                    data[0].append(instance)
                else:
                    data[1].append(instance)

    return data


def holdout(dataset, percentage_test):
    """
    :param dataset: the full dataset
    :param percentage_test: float, percentage of instances that needs to go to the test partition
    :return: (dict_of_train_instances, dict_of_test_instances)
    """
    number_classes = len(dataset) #if 2, only one output, if 3, three outputs

    if number_classes == 2:
        group1 = dataset[0]
        group2 = dataset[1]
        shuffle(group1)
        shuffle(group2)

        n_g1 = int(len(group1) * percentage_test)
        n_g2 = int(len(group2) * percentage_test)

        group1_train_dataset = group1[:n_g1]
        group1_test_dataset = group1[n_g1:]
        group2_train_dataset = group2[:n_g2]
        group2_test_dataset = group2[n_g2:]

        train_dataset = {tuple(x):[0.0] for x in group1_train_dataset}
        train_dataset.update({tuple(x):[1.0] for x in group2_train_dataset})

        test_dataset = {tuple(x): [0.0] for x in group1_test_dataset}
        test_dataset.update({tuple(x): [1.0] for x in group2_test_dataset})

    else:
        group1 = dataset[0]
        group2 = dataset[1]
        group3 = dataset[2]
        shuffle(group1)
        shuffle(group2)
        shuffle(group3)

        n_g1 = int(len(group1) * percentage_test)
        n_g2 = int(len(group2) * percentage_test)
        n_g3 = int(len(group2) * percentage_test)

        group1_train_dataset = group1[:n_g1]
        group1_test_dataset = group1[n_g1:]
        group2_train_dataset = group2[:n_g2]
        group2_test_dataset = group2[n_g2:]
        group3_train_dataset = group3[:n_g3]
        group3_test_dataset = group3[n_g3:]

        train_dataset = {tuple(x): [1.0, 0.0, 0.0] for x in group1_train_dataset}
        train_dataset.update({tuple(x): [0.0, 1.0, 0.0] for x in group2_train_dataset})
        train_dataset.update({tuple(x): [0.0, 0.0, 1.0] for x in group3_train_dataset})

        test_dataset = {tuple(x): [1.0, 0.0, 0.0] for x in group1_test_dataset}
        test_dataset.update({tuple(x): [0.0, 1.0, 0.0] for x in group2_test_dataset})
        test_dataset.update({tuple(x): [0.0, 0.0, 1.0] for x in group3_test_dataset})

    return (train_dataset,test_dataset)


def cross_validation(dataset, percentage_test, iterations,
                     hidden_layers_sizes, neurons_type='sigmoid', alpha=0.0001, lamb=0.0):
    """
    :param dataset: the full dataset
    :param percentage_test: float, percentage of instances that needs to go to the test partition
    :param iterations: int, number of holdouts to execute
    should call holdout to generate the train and test dicts
    should call train_NN to train the dataset
    should call test_NN to get the performances
    :return: (average_performance, stddev_performance)
    """
    results = []
    for it in range(1, iterations + 1):
        print("Iteration",it)
        train_dataset, test_dataset = holdout(dataset, percentage_test)
        input_size = len(list(train_dataset.keys())[0])
        output_size = len(list(train_dataset.values())[0])
        nn = NeuralNet(input_size, output_size, hidden_layers_sizes, neurons_type, alpha, lamb)
        train_NN(nn, train_dataset)
        performance = test_NN(nn, test_dataset)
        results.append(performance)

    return (mean(results), stdev(results))


def train_NN(NN, train_instances):
    """
    :param NN: the neural net
    :param train_instances: the train instances dict
    """
    errors = [1.0]
    number_of_instances = len(train_instances.keys())
    i=0
    while True:
        i+=1
        # accumulate the errors
        error_acc = 0.0
        instances = list(train_instances.keys())
        shuffle(instances)
        for instance in instances:
            error_acc += NN.back_propagation(list(instance), train_instances[instance])
        mean_error = error_acc/number_of_instances
        #print(mean_error)
        if i%100 == 0:
            print(i)
        if i > 1000: #(abs(mean_error-errors[-1])/((mean_error+errors[-1]))/2) < 0.0001: #stop condition
            errors.append(mean_error)
            return errors
        errors.append(mean_error)


def test_NN(NN, test_instances):
    """
    :param NN: the neural net
    :param test_instances: the test instances dict
    :return: the performance of the NN in test_mode
    only implemented after we know what will be the dataset (predict a category or a number? this is relevant)
    """
    number_of_instances = len(test_instances.keys())
    acc = 0
    for instance in list(test_instances.keys()):
        expected = test_instances[instance]
        output,_ = NN.predict(list(instance))
        if len(expected) == 1:
            if expected[0] == 0.0 and output[0] < 0.5:
                acc += 1
            if expected[0] == 1.0 and output[0] > 0.5:
                acc += 1
        else:
            if expected.index(max(expected)) == output.index(max(output)):
                acc += 1
    print(float(acc)/float(number_of_instances))
    return float(acc)/float(number_of_instances)

if __name__ == "__main__":
    dataset = read_dataset('wine')
    print(cross_validation(dataset, 0.2, 5, [13,5], neurons_type='sigmoid', alpha=0.01, lamb=0.0))
    """mode_parser, mode_argument_parses = parser()

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

    if str(mode_parser.mode) == 'verify':
        input_nn = str(mode_argument_parses.i[0])
        try:
            with open(input_nn,'rb') as input_file:
                neural_net = pickle.load(input_file)
                verify(neural_net)
        except EnvironmentError:
            print("Can't open file %s" % input_nn)
            exit()"""

