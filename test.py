#!/usr/bin/python
from dataset import dataset
from neural_network import Network
from util import print_initial_message


def test_sample(network, sample):
    print('[%d %d] : %r' % (sample[0], sample[1], network.binary_classify(sample)))


def main():
    net = Network(2, 1)
    net.add_neuron(connections=['in0', 'in1'], weights=[+20, +20], bias=-30, activation='sigmoid')
    net.add_neuron(connections=['in0', 'in1'], weights=[-20, -20], bias=+10, activation='sigmoid')
    net.neurons['out0'].connections = ['h0', 'h1']
    net.neurons['out0'].weights = [+20, +20]
    net.neurons['out0'].neuron_type = 'feed_forward_neuron'
    net.neurons['out0'].bias = -10
    net.neurons['out0'].activation = 'sigmoid'
    net.save_network_to_file('truth.network.json')

    print('Truth network (#%s):' % net.name)
    print('%0.4f' % net.fitness(dataset))
    test_sample(net, [0, 0])
    test_sample(net, [0, 1])
    test_sample(net, [1, 0])
    test_sample(net, [1, 1])
    print '------------------------------------'

    try:
        net = Network()
        net.load_network_from_file('result.network.json')

        print('result network (#%s):' % net.name)
        print('%0.4f' % net.fitness(dataset))
        test_sample(net, [0, 0])
        test_sample(net, [0, 1])
        test_sample(net, [1, 0])
        test_sample(net, [1, 1])
        print '------------------------------------'
    except IOError:
        pass


if __name__ == "__main__":
    print_initial_message('test', verbose_level=1)
    main()
