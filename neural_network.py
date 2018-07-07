#!/usr/bin/python
import json
import math

from constants import *
from neuron import Neuron
from util import generate_random_name


class Network:
    def __init__(self, num_of_inputs=None, num_of_outputs=None, name=None):
        self.neuron_counter = 0
        self.num_of_outputs = num_of_outputs
        self.neurons = {}
        self.name = name if name is not None else generate_random_name()

        if num_of_outputs is None or num_of_inputs is None:
            return

        for i in range(num_of_inputs):
            self.neurons['in%d' % i] = Neuron(name='in%d' % i)
        for i in range(num_of_outputs):
            self.neurons['out%d' % i] = Neuron(name='out%d' % i)

    def add_neuron(self, **kwargs):
        kwargs['name'] = 'h%d' % self.neuron_counter
        self.neurons['h%d' % self.neuron_counter] = Neuron(**kwargs)
        self.neuron_counter += 1

    def binary_classify(self, input_values):
        res = self.predict(input_values)[0]
        return res > 0.5

    def predict(self, values):
        for n in self.neurons:
            self.neurons[n].reset()

        for i, v in enumerate(values):
            if v is None:
                continue
            self.neurons['in%d' % i].set_value(v)

        res = []
        for i in range(self.num_of_outputs):
            val = self.neurons['out%d' % i].get_value(self.neurons)
            res.append(val)
        return res

    def error(self, dataset):
        total_error = 0
        for data in dataset:
            x = data['inputs']
            y = data['outputs']
            h = self.predict(x)

            sample_error = 0
            for i in range(len(y)):
                if y[i] == 0:
                    if h[i] == 1:
                        h[i] = 0.5
                    sample_error += math.log(1.0 - h[i])
                elif y[i] == 1:
                    if h[i] == 0:
                        h[i] = 0.5
                    sample_error += math.log(h[i])
                else:
                    if h[i] == 0 or h[i] == 1:
                        h[i] = 0.5
                    sample_error += y[i] * math.log(h[i]) + (1.0 - y[i]) * math.log(1.0 - h[i])
            total_error += sample_error
        return total_error / -len(dataset)

    def regularity(self, dataset):
        reg = 0
        for l in self.neurons:
            if self.neurons[l].weights is not None:
                reg += sum([i * i for i in self.neurons[l].weights])
        return REG_TERM * reg / (2 * len(dataset))

    def fitness(self, dataset):
        return self.error(dataset) + self.regularity(dataset)

    def save_network_to_file(self, filename):
        res = {key: value.serialize() for key, value in self.neurons.items()}
        res['__neuron_counter'] = self.neuron_counter
        res['__num_of_outputs'] = self.num_of_outputs
        res['__name'] = self.name
        with open(filename, 'w') as f:
            json.dump(res, f, indent=2)

    def load_network_from_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.neurons = {key: Neuron.deserialize(value) for key, value in data.items() if not key.startswith('__')}
            self.neuron_counter = data['__neuron_counter']
            self.num_of_outputs = data['__num_of_outputs']
            self.name = data['__name']
