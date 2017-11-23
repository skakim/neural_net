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
    mode_p.add_parser('survival',
                      help='Train a NN with survival dataset.')
    mode_p.add_parser('wine', help='Train a NN with wine dataset.')
    mode_p.add_parser('contraceptive', help='Train a NN with contraceptive dataset.')
    mode_p.add_parser('cancer', help='Train a NN with cancer dataset.')
    mode_p.add_parser('verify', help='Verify random NN gradient.')

    args_first, leftovers = mode_parser.parse_known_args()

    return args_first
