import random
import math


def rand_epsilon():
    return random.random() * 0.0001


def activate(value, func):
    if func == 'linear':
        return value
    if func == 'sigmoid':
        return 1 / (1 + math.e ** (-value))


def generate_random_name():
    name = '%03x' % random.getrandbits(16)
    return name[:3]


def print_initial_message(task, verbose_level=0):
    def __print_message(task, verbose_level):
        print('Running uNEAT-T1 (%s)' % task)

        if verbose_level < 1:
            return

        print('micro-(Neuro-Evolution of Augmenting Topology) Framework -- Version T1')
        print('(C) 2018 - Aref Moqadam Mehr - Under MIT Licenced')
        print('For more information visit: http://github.com/arefmq')

        if verbose_level < 2:
            return
        print('''
Usage:
    - train.py : for training a network and store the result in
                 'result.network.json' file.
    - test.py  : for validating the trained network.
''')

    __print_message(task, verbose_level)
    print('-------------------------------------------------------')

