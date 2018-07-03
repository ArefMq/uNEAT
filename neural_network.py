#!/usr/bin/python
import random
import math
import json
from collections import OrderedDict

neurons = {}
REG_TERM = 0.00001

CONNECTIONS = 'connections'
WEIGHTS = 'weights'
BIAS = 'bias'
ACTIVATION = 'activation_function'
VALUE = 'value'


def rand_epsilon():
    return random.random() * 0.0001


def make_neuron(connections=None, weights=None, bias=None, activation='linear', value=0):
    if bias is None:
        bias = rand_epsilon()

    if connections is not None and weights is None:
        weights = [rand_epsilon() for _ in range(len(connections))]

    return {
        CONNECTIONS: connections,
        WEIGHTS: weights,
        BIAS: bias,
        ACTIVATION: activation,
        VALUE: value
    }


def activate(value, func):
    if func == 'linear':
        return value
    if func == 'sigmoid':
        return 1 / (1 + math.e ** (-value))


def binary_classify(network, values):
    res = predict(network, values)
    return res > 0.5


def predict(network, values):
    network['x1'][VALUE] = values[0]
    network['x2'][VALUE] = values[1]

    # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # print json.dumps(network, indent=2)
    # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    update_network(network)

    return network['o'][VALUE]


def update_network(network):
    for n in network:
        connections = network[n][CONNECTIONS]
        weights = network[n][WEIGHTS]

        if connections is None:
            continue

        if len(connections) != len(weights):
            raise Exception('coefficients mis-match...')

        result = network[n][BIAS]
        for i in range(len(connections)):
            # print('  %s (%0.2f) * %f' % (connections[i], network[connections[i]]['value'], weights[i]))
            result += network[connections[i]][VALUE] * weights[i]
        network[n]['value'] = activate(result, network[n][ACTIVATION])
        # print('%s - %0.2f' % (n, network[n]['value']))


def error(network, dataset):
    total_error = 0
    for data in dataset:
        inputs = data['inputs']
        outputs = data['outputs']

        for i in inputs:
            network[i][VALUE] = inputs[i]

        update_network(network)

        sample_error = 0
        for o in outputs:
            y = outputs[o]
            h = network[o][VALUE]
            if y == 0:
                if h == 1:
                    h = 0.5
                sample_error += math.log(1.0-h)
            elif y == 1:
                if h == 0:
                    h = 0.5
                sample_error += math.log(h)
            else:
                if h == 0 or h == 1:
                    h = 0.5
                sample_error += y * math.log(h) + (1.0-y) * math.log(1.0-h)
        total_error += sample_error
    return total_error / -len(dataset)


def regularity(network, dataset):
    reg = 0
    for l in network:
        if network[l][WEIGHTS] is not None:
            reg += sum([i*i for i in network[l][WEIGHTS]])
    return REG_TERM * reg / (2 * len(dataset))


def fitness(network, dataset):
    return error(network, dataset) + regularity(network, dataset)


def save_network_to_file(network, file):
    res = []
    for k in network.keys():
        res.append((k, network[k]))
    with open(file, 'w') as f:
        json.dump(res, f, indent=2)
    return True


def load_network_from_file(file):
    with open(file, 'r') as f:
        data = json.load(f)
        return OrderedDict(data)
