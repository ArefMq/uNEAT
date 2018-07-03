from genetics import Genetics
from neural_network import binary_classify, error, regularity, fitness, save_network_to_file
from dataset import dataset


def test_sample(network, sample):
    print('[%d %d] : %r' % (sample[0], sample[1], binary_classify(network, sample)))


if __name__ == '__main__':
    print('running uNEAT-T1 (Test)')
    print('micro-(Neuro-Evolution of Augmenting Topology) Framework -- Version T1')
    print('Licenced under MIT')

    best = None
    try:
        gn = Genetics(initial_population=300, max_population=9000)
        best = gn.get_best(max_iter=40000, optimal_fitness=0.02)
        save_network_to_file(best.Genes, 'result.network.json')
    except Exception as ex:
        print('\n\nProgram Terminated. %s' % ex.message)

    if best is not None:
        print('validating::')
        print('Fitness:    %0.4f' % fitness(best.Genes, dataset))
        print('error:      %0.4f' % error(best.Genes, dataset))
        print('regularity: %0.4f' % regularity(best.Genes, dataset))

        print('\n\nSamples:')
        test_sample(best.Genes, [0, 0])
        test_sample(best.Genes, [0, 1])
        test_sample(best.Genes, [1, 0])
        test_sample(best.Genes, [1, 1])
