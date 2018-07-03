from genetics import Genetics
from neural_network import binary_classify, error, regularity, fitness
from dataset import dataset


if __name__ == '__main__':
    print('running uNEAT-T1 (Test)')
    print('micro-(Neuro-Evolution of Augmenting Topology) Framework -- Version T1')
    print('Licenced under MIT')

    gn = Genetics(initial_population=300, max_population=9000)
    best = gn.get_best(max_iter=40000, optimal_fitness=0.02)

    print 'validating::'
    print('Fitness:    %0.4f' % fitness(best.Genes, dataset))
    print('error:      %0.4f' % error(best.Genes, dataset))
    print('regularity: %0.4f' % regularity(best.Genes, dataset))

    print '0 0', binary_classify(best.Genes, [0, 0])
    print '0 1', binary_classify(best.Genes, [0, 1])
    print '1 0', binary_classify(best.Genes, [1, 0])
    print '1 1', binary_classify(best.Genes, [1, 1])
