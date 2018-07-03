import random
from collections import OrderedDict

from neural_network import *
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
        return OrderedDict([
            ('x1', make_neuron()),
            ('x2', make_neuron()),

            ('a1', make_neuron(['x1', 'x2'], [self._random_weight(), self._random_weight()], self._random_weight(), activation='sigmoid')),
            ('a2', make_neuron(['x1', 'x2'], [self._random_weight(), self._random_weight()], self._random_weight(), activation='sigmoid')),

            ('o', make_neuron(['a1', 'a2'], [self._random_weight(), self._random_weight()], self._random_weight(), activation='sigmoid')),
        ])

    def update_fitness(self):
        self.Regularization = regularity(self.Genes, dataset)
        self.Error = error(self.Genes, dataset)
        self.Fitness = self.Error + self.Regularization

    @staticmethod
    def cross_over(a, b):
        def __extract_weights(gene):
            return [
                gene['a1'][WEIGHTS][0],
                gene['a1'][WEIGHTS][1],
                gene['a1'][BIAS],
                gene['a2'][WEIGHTS][0],
                gene['a2'][WEIGHTS][1],
                gene['a2'][BIAS],
                gene['o'][WEIGHTS][0],
                gene['o'][WEIGHTS][1],
                gene['o'][BIAS],
            ]

        def __set_weights(gene, weights):
            gene['a1'][WEIGHTS][0] = weights[0]
            gene['a1'][WEIGHTS][1] = weights[1]
            gene['a1'][BIAS] = weights[2]
            gene['a2'][WEIGHTS][0] = weights[3]
            gene['a2'][WEIGHTS][1] = weights[4]
            gene['a2'][BIAS] = weights[5]
            gene['o'][WEIGHTS][0] = weights[6]
            gene['o'][WEIGHTS][1] = weights[7]
            gene['o'][BIAS] = weights[8]
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
                self.Genes[layer_index][BIAS] = self._random_weight()
            else:
                self.Genes[layer_index][WEIGHTS][weight_index] = self._random_weight()

        index = random.sample(['a1', 'a2', 'o'], 1)[0]
        w_index = random.sample([-1, 0, 1], 1)[0]
        __mutate_weight(index, w_index)

    def __str__(self):
        if self.Genes is None:
            return 'None'

        print self.Genes

        res = ''
        for g in self.Genes:
            gene = self.Genes[g]
            res += '%s)\t%s(%0.2f)\t%s(%0.2f)\tbias(%0.2f)\t%s\n' % (
                g,
                '* ' if gene[CONNECTIONS] is None else gene[CONNECTIONS][0],
                0 if gene[WEIGHTS] is None else gene[WEIGHTS][0],
                '* ' if gene[CONNECTIONS] is None else gene[CONNECTIONS][1],
                0 if gene[WEIGHTS] is None else gene[WEIGHTS][1],
                0 if gene[BIAS] is None else gene[BIAS],
                '*' if gene[ACTIVATION] is None else gene[ACTIVATION]
            )
        res += '\n'
        res += 'Fitness:    %0.3f\n' % self.Fitness
        res += 'Error:      %0.3f\n' % self.Error
        res += 'Regularity: %0.3f\n' % self.Regularization
        return res
