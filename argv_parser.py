import argparse
import traceback
import sys


def parser() -> (argparse.Namespace, list):
    """
    Parses command line arguments.

    :return: a tuple containing parsed arguments and leftovers
    """
    mode_parser = argparse.ArgumentParser(prog='Neural_Net')
    mode_p = mode_parser.add_subparsers(dest='mode')
    mode_p.required = True
    mode_p.add_parser('create_net',
                      help='creates a neural_network')
    mode_p.add_parser('test', help='test a neural network given a test_set')
    mode_p.add_parser('train', help='trains a neural network given a data_set')
    mode_p.add_parser('auto_train', help='automated training')

    args_first, leftovers = mode_parser.parse_known_args()

    if str(args_first.mode) == 'create_net':
        second_parser = argparse.ArgumentParser(prog='create_net')
        second_parser.add_argument('-o', nargs=1, help='directory of output', type=int, required=True)
        second_parser.add_argument('-input_size', nargs=1, help='input size of neural net', type=int, required=True)
        second_parser.add_argument('-output_size', nargs=1, help='number of output neurons', type=int, required=True)
        second_parser.add_argument('-hidden_layer_sizes', nargs='+', help='sizes of hidden layer ex:\n'
                                   '-hidden_layer_sizes 4 6 8 4', type=int, required=True)
    elif str(args_first.mode) == 'test':
        second_parser = argparse.ArgumentParser(prog='test')
        second_parser.add_argument('-i', nargs=1, help='input neural_net', required=True)
        second_parser.add_argument('-o', nargs=1, help='output results', required=True)
    elif str(args_first.mode) == 'train':
        second_parser = argparse.ArgumentParser(prog='train')
        second_parser.add_argument('-i', nargs=1, help='input neural_net', required=True)
        second_parser.add_argument('-o', nargs=1, help='output trained_net', required=True)
        second_parser.add_argument('-data',  nargs=1, help='input dataset', required=True)
    else:
        second_parser = argparse.ArgumentParser(prog='auto_train')
        second_parser.add_argument('-i', nargs=1, help='input neural_net', required=True)
        second_parser.add_argument('-o', nargs=1, help='output trained_net', required=True)
        second_parser.add_argument('-data',  nargs=1, help='input dataset', required=True)

    args_second, second_leftovers = second_parser.parse_known_args(leftovers)

    return args_first, args_second






if __name__ == '__main__':
    parser()
