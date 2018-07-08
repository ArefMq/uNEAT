import random
from collections import OrderedDict

from neural_network import Network
from dataset import dataset


class Chromosome:
    def __init__(self, genes=None):
        self.Genes = genes if genes is not None else self._random_genes()
        self.Fitness = None
        self.Regularization = None
        self.Error = None
        self.update_fitness()

    @staticmethod
    def _random_weight(scale=20):
        return random.random() * 2.0 * scale - scale

    def _random_genes(self):
        net = Network(2, 1)
        net.add_neuron(connections=['in0', 'in1'], weights=[self._random_weight(), self._random_weight()], bias=self._random_weight(), activation='sigmoid')
        net.add_neuron(connections=['in0', 'in1'], weights=[self._random_weight(), self._random_weight()], bias=self._random_weight(), activation='sigmoid')
        net.neurons['out0'].connections = ['h0', 'h1']
        net.neurons['out0'].weights = [self._random_weight(), self._random_weight()]
        net.neurons['out0'].neuron_type = 'feed_forward_neuron'
        net.neurons['out0'].bias = self._random_weight()
        net.neurons['out0'].activation = 'sigmoid'

        return net

    def update_fitness(self):
        self.Regularization = self.Genes.regularity(dataset)
        self.Error = self.Genes.error(dataset)
        self.Fitness = self.Error + self.Regularization

    @staticmethod
    def cross_over(a, b):
        def __extract_weights(gene):
            return [
                gene.neurons['h0'].weights[0],
                gene.neurons['h0'].weights[1],
                gene.neurons['h0'].bias,
                gene.neurons['h1'].weights[0],
                gene.neurons['h1'].weights[1],
                gene.neurons['h1'].bias,
                gene.neurons['out0'].weights[0],
                gene.neurons['out0'].weights[1],
                gene.neurons['out0'].bias
            ]

        def __set_weights(gene, weights):
            gene.neurons['h0'].weights[0] = weights[0]
            gene.neurons['h0'].weights[1] = weights[1]
            gene.neurons['h0'].bias = weights[2]
            gene.neurons['h1'].weights[0] = weights[3]
            gene.neurons['h1'].weights[1] = weights[4]
            gene.neurons['h1'].bias = weights[5]
            gene.neurons['out0'].weights[0] = weights[6]
            gene.neurons['out0'].weights[1] = weights[7]
            gene.neurons['out0'].bias = weights[8]

            return gene

        former_w = dict()
        former_w['a'] = __extract_weights(a.Genes)
        former_w['b'] = __extract_weights(b.Genes)
        new_w = [(i1+i2)/2 for i1, i2 in zip(former_w['a'], former_w['b'])]

        # for i in range(len(former_w['a'])):
        #     index = random.sample(['a', 'b'], 1)[0]
        #     new_w.append(former_w[index][i])

        result = Chromosome()
        result.Genes = __set_weights(result.Genes, new_w)
        return result

    def mutate(self, allow_only_growing=False):
        def __mutate_weight(layer_index, weight_index):
            if weight_index == -1:
                self.Genes.neurons[layer_index].bias = self._random_weight()
            else:
                self.Genes.neurons[layer_index].weights[weight_index] = self._random_weight()

        index = random.sample(['h0', 'h1', 'out0'], 1)[0]
        w_index = random.sample([-1, 0, 1], 1)[0]
        __mutate_weight(index, w_index)

    def __str__(self):
        if self.Genes is None:
            return 'None'

        print self.Genes.neurons

        res = ''
        for g in self.Genes.neurons:
            gene = self.Genes.neurons[g]
            res += '%s)\t%s(%0.2f)\t%s(%0.2f)\tbias(%0.2f)\t%s\n' % (
                g,
                '* ' if gene.connections is None else gene.connections[0],
                0 if gene.weights is None else gene.weights[0],
                '* ' if gene.connections is None else gene.connections[1],
                0 if gene.weights is None else gene.weights[1],
                0 if gene.bias is None else gene.bias,
                '*' if gene.activation is None else gene.activation
            )
        res += '\n'
        res += 'Fitness:    %0.3f\n' % self.Fitness
        res += 'Error:      %0.3f\n' % self.Error
        res += 'Regularity: %0.3f\n' % self.Regularization
        return res
