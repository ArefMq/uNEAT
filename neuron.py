#!/usr/bin/python
import logging

from constants import *
from util import rand_epsilon, activate

neurons = {}
logger = logging.getLogger(__name__)


class Neuron:
    def __init__(self, **kwargs):
        # parameters
        self.connections = kwargs.get(CONNECTIONS, None)
        self.weights = kwargs.get(WEIGHTS, None)
        self.bias = kwargs.get(BIAS, rand_epsilon())
        self.activation = kwargs.get(ACTIVATION, 'linear')

        # dependent parameters
        if self.connections is not None:
            self.neuron_type = FEED_FORWARD_NEURON
            if self.weights is None:
                self.weights = [rand_epsilon() for _ in range(len(self.connections))]
            elif len(self.weights) != len(self.connections):
                raise Exception('connection and weight size does not match.')
        else:
            self.neuron_type = TERMINAL_NEURON

        # variables
        self.enabled = True
        self.value = None

    def update(self, network):
        result = self.bias
        for i in range(len(self.connections)):
            n = network[self.connections[i]]
            result += n.get_value(network) * self.weights[i]
        self.value = activate(result, self.activation)

    def disable(self):
        self.enabled = False

    def set_enabled(self, enabling=True):
        self.enabled = enabling

    def reset(self):
        self.value = None

    def set_value(self, value):
        if self.neuron_type != TERMINAL_NEURON:
            logger.warning('setting a not terminal neuron value.')
        self.value = value

    def get_value(self, network):
        if self.value is None:
            self.update(network)
        return self.value

    def serialize(self):
        return {
            'connections': self.connections,
            'weights': self.weights,
            'bias': self.bias,
            'activation': self.activation
        }

    @staticmethod
    def deserialize(data):
        n = Neuron(**data)
        return n

