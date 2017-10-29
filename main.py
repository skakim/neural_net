from neural_net import NeuralNet
from statistics import mean, stdev
from random import shuffle

def read_dataset(filename): #TODO: we need to know how will be the dataset
    '''
    :param filename: string, the name of the dataset file
    :return: dict of instances {array_of_inputs : expected_output_values}
    '''
    pass

def holdout(dataset, percentage_test): #TODO: we need to know how will be the dataset (there are classes? how much?)
    '''
    :param dataset: the full dataset
    :param percentage_test: float, percentage of instances that needs to go to the test partition
    :return: (dict_of_train_instances, dict_of_test_instances)
    '''
    pass

def cross_validation(dataset, percentage_test, iterations, mode,
                     hidden_layers_sizes, neurons_type='sigmoid', alpha=0.0001, regularization=False):
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
    for it in range(1,iterations+1):
        train_dataset, test_dataset = holdout(dataset, percentage_test)
        input_size = len(train_dataset.keys()[0])
        output_size = len(train_dataset[train_dataset.keys()[0]])
        nn = NeuralNet(input_size, output_size, hidden_layers_sizes, neurons_type, alpha, regularization)
        train_NN(nn, train_dataset, mode)
        performance = test_NN(nn, test_dataset)
        results.append(performance)

    return (mean(results),stdev(results))

def train_NN(NN, train_instances, mode):
    '''
    :param NN: the neural net
    :param train_instances: the train instances dict
    :param mode: "mini_batch" (call back_propagation for each instance) or "batch" (call back_propagation after all instances)
    '''
    number_of_instances = len(train_instances.keys())
    if mode == "batch":
        while True:
            #accumulate the errors
            error_acc = 0.0
            for instance in train_instances.keys():
                output = NN.predict(instance)
                error_acc += (output - train_instances[instance])/number_of_instances
            #call back_propagation
            NN.back_propagation(error_acc)

            if __stop_conditions: #TODO: we need to talk about what will be the stop condition(s) (ask Bruno too)
                break

    elif mode == "mini_batch":
        while True:
            for instance in train_instances.keys():
                output = NN.predict(instance)
                error = (output - train_instances[instance])
                #call back_propagation
                NN.back_propagation(error)

            if __stop_conditions: #TODO: we need to talk about what will be the stop condition(s) (ask Bruno too)
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
    #TODO: use argparse to read parameters like .txt name? depend how it will be given to us
    dataset = read_dataset("test.txt")
    #TODO: define here the tests we'll do