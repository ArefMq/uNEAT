from genetics import Genetics
from neural_network import binary_classify, error, regularity, fitness
from dataset import dataset


if __name__ == '__main__':
    gn = Genetics(initial_population=300, max_population=9000)

    best = gn.get_best(max_iter=40000, optimal_fitness=0.02)
    print(best)

    print 'validating::'
    n3 = best.Genes
    print('Fitness:    %0.4f' % fitness(n3, dataset))
    print('error:      %0.4f' % error(n3, dataset))
    print('regularity: %0.4f' % regularity(n3, dataset))

    print '0 0', binary_classify(n3, [0, 0])
    print '0 1', binary_classify(n3, [0, 1])
    print '1 0', binary_classify(n3, [1, 0])
    print '1 1', binary_classify(n3, [1, 1])

    # try:
    #     best = gn.get_best(max_iter=100000)
    #     print(best)
    # except Exception as ex:
    #     print('\n\n%s\n' % ex.message)
