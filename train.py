#!/usr/bin/python
from dataset import dataset
from genetics import Genetics
from util import print_initial_message


def test_sample(network, sample):
    print('[%d %d] : %r' % (sample[0], sample[1], network.binary_classify(sample)))


if __name__ == '__main__':
    print_initial_message('train', verbose_level=1)
    best = None
    try:
        gn = Genetics(initial_population=300, max_population=9000)
        best = gn.get_best(max_iter=40000, optimal_fitness=0.02)

        best.Genes.save_network_to_file('result.network.json')
    except Exception as ex:
        print('\n\nProgram Terminated. %s' % ex.message)

    if best is not None:
        print('validating::')
        print('Fitness:    %0.4f' % best.Genes.fitness(dataset))
        print('error:      %0.4f' % best.Genes.error(dataset))
        print('regularity: %0.4f' % best.Genes.regularity(dataset))

        print('\n\nSamples:')
        test_sample(best.Genes, [0, 0])
        test_sample(best.Genes, [0, 1])
        test_sample(best.Genes, [1, 0])
        test_sample(best.Genes, [1, 1])
