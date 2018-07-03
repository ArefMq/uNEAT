#!/usr/bin/python
from collections import OrderedDict

from neural_network import *
from dataset import dataset


def test_sample(network, sample):
    print('[%d %d] : %r' % (sample[0], sample[1], binary_classify(network, sample)))


def main():
    truth_network = OrderedDict([
        ('x1', make_neuron()),
        ('x2', make_neuron()),

        ('a1', make_neuron(['x1', 'x2'], [+20, +20], -30, activation='sigmoid')),
        ('a2', make_neuron(['x1', 'x2'], [-20, -20], +10, activation='sigmoid')),

        ('o', make_neuron(['a1', 'a2'], [+20, +20], -10, activation='sigmoid')),

    ])

    arbitrary_network = OrderedDict([
        ('x1', make_neuron()),
        ('x2', make_neuron()),

        ('a1', make_neuron(['x1', 'x2'], activation='sigmoid')),
        ('a2', make_neuron(['x1', 'x2'], activation='sigmoid')),

        ('o', make_neuron(['a1', 'a2'], activation='sigmoid')),
    ])

    trained_network = load_network_from_file('result.network.json')

    print('%0.4f' % fitness(truth_network, dataset))
    test_sample(truth_network, [0, 0])
    test_sample(truth_network, [0, 1])
    test_sample(truth_network, [1, 0])
    test_sample(truth_network, [1, 1])
    print '------------------------------------'

    print('%0.4f' % fitness(arbitrary_network, dataset))
    test_sample(arbitrary_network, [0, 0])
    test_sample(arbitrary_network, [0, 1])
    test_sample(arbitrary_network, [1, 0])
    test_sample(arbitrary_network, [1, 1])

    print '------------------------------------'
    print('%0.4f' % fitness(trained_network, dataset))
    test_sample(trained_network, [0, 0])
    test_sample(trained_network, [0, 1])
    test_sample(trained_network, [1, 0])
    test_sample(trained_network, [1, 1])


if __name__ == "__main__":
    print('running uNEAT-T1 (Test)')
    print('micro-(Neuro-Evolution of Augmenting Topology) Framework -- Version T1')
    print('Licenced under MIT')
    main()
