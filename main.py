#!/usr/bin/python
from collections import OrderedDict

from neural_network import *
from dataset import dataset


def main():
    n1 = OrderedDict([
        ('x1', make_neuron()),
        ('x2', make_neuron()),

        ('a1', make_neuron(['x1', 'x2'], [+20, +20], -30, activation='sigmoid')),
        ('a2', make_neuron(['x1', 'x2'], [-20, -20], +10, activation='sigmoid')),

        ('o', make_neuron(['a1', 'a2'], [+20, +20], -10, activation='sigmoid')),

    ])

    n2 = OrderedDict([
        ('x1', make_neuron()),
        ('x2', make_neuron()),

        ('a1', make_neuron(['x1', 'x2'], activation='sigmoid')),
        ('a2', make_neuron(['x1', 'x2'], activation='sigmoid')),

        ('o', make_neuron(['a1', 'a2'], activation='sigmoid')),
    ])

    n3 = OrderedDict([('x1', {'connections': None, 'activation_function': 'linear', 'bias': 9.828672788366738e-05, 'weights': None, 'value': 1}), ('x2', {'connections': None, 'activation_function': 'linear', 'bias': 2.7259131107723678e-05, 'weights': None, 'value': 1}), ('a1', {'connections': ['x1', 'x2'], 'activation_function': 'sigmoid', 'bias': 3.76319154772294, 'weights': [-12.938129613639354, -13.49425982954134], 'value': 1.4285401294873268e-10}), ('a2', {'connections': ['x1', 'x2'], 'activation_function': 'sigmoid', 'bias': 13.644960724170112, 'weights': [-10.421073514779145, -10.5135504240457], 'value': 0.000682092322050054}), ('o', {'connections': ['a1', 'a2'], 'activation_function': 'sigmoid', 'bias': 12.527017250878345, 'weights': [19.05677649753416, -17.942940946087855], 'value': 0.9999963280297361})])

    print('%0.4f' % fitness(n1, dataset))
    print '0 0', binary_classify(n1, [0, 0])
    print '0 1', binary_classify(n1, [0, 1])
    print '1 0', binary_classify(n1, [1, 0])
    print '1 1', binary_classify(n1, [1, 1])
    print '------------------------------------'

    print('%0.4f' % fitness(n2, dataset))
    print '0 0', binary_classify(n2, [0, 0])
    print '0 1', binary_classify(n2, [0, 1])
    print '1 0', binary_classify(n2, [1, 0])
    print '1 1', binary_classify(n2, [1, 1])

    print '------------------------------------'
    print('%0.4f' % fitness(n3, dataset))
    print '0 0', binary_classify(n3, [0, 0])
    print '0 1', binary_classify(n3, [0, 1])
    print '1 0', binary_classify(n3, [1, 0])
    print '1 1', binary_classify(n3, [1, 1])


    # update_network(neurons)
    # print('result = %f (%s)' % (neurons['o']['value'], 'yes' if neurons['o']['value'] > 0.5 else 'no'))


if __name__ == "__main__":
    print('running NEAT-T1')
    main()
