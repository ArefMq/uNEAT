import random
import math
from time import sleep

from copy import deepcopy
from chromosome import Chromosome


class Genetics(object):
    def __init__(self, initial_population=10, max_population=1000):
        # parameters
        self.max_population = max_population
        self.initial_population = initial_population
        self.mutation_chance = 0.9
        self.selection_rate = 0.30
        self.genesis_rate = 0.15
        self.cross_over_rate = 4.6

        # variables
        self.population = []
        self.best = None
        self.fitness = None

    def _init_population(self):
        self.population = [
            Chromosome() for _ in range(self.initial_population)
        ]

    def _calculate_fitness(self):
        if not self.population:
            return

        if self.best is None:
            self.best = self.population[0]

        error = 0
        for p in self.population:
            p.update_fitness()
            if p.Fitness < self.best.Fitness:
                self.best = deepcopy(p)
            error += p.Fitness
        self.fitness = error / len(self.population)

    def _select_bests(self):
        cumulative_set = []
        total = 0
        for d in self.population:
            total += 1 / (0.01 + d.Fitness)
            cumulative_set.append((d, total))

        step = total / (len(cumulative_set) * self.selection_rate)
        init_step = random.random() * step

        def select_r(value):
            for d in cumulative_set:
                if value < d[1]:
                    return d[0]

        result = []
        i = init_step
        while i < total:
            r = select_r(i)
            if r is None:
                raise Exception('invalid select_r')
            result.append(r)
            i += step

        self.population = result

    def _genesis(self):
        if random.random() < self.genesis_rate:
            self.population.append(Chromosome())

    def _cross_over(self, positive_progress_only=False):
        if len(self.population) <= 1:
            return

        cross_over_occur = math.ceil(len(self.population) * self.cross_over_rate)

        result = []
        for i in range(int(cross_over_occur)):
            a, b = random.sample(self.population, 2)
            offspring = Chromosome.cross_over(a, b)
            if not positive_progress_only or (offspring.Fitness > max(a.Fitness, b.Fitness)):
                result.append(offspring)
        self.population.extend(result)

    def _mutate(self):
        for i in self.population:
            if random.random() < self.mutation_chance:
                i.mutate(True)

    def get_best(self, max_iter=1000, optimal_fitness=0):
        random.seed()
        self._init_population()

        iteration = 0
        while iteration < max_iter and (not self.best or self.best.Fitness > optimal_fitness):
            self._cross_over(True)
            self._select_bests()
            self._mutate()
            self._genesis()

            self._dynamic()
            self._calculate_fitness()
            self._validation()
            self._display(iteration)
            iteration += 1

        return self.best

    def _dynamic(self):
        if len(self.population) < self.initial_population:
            if self.cross_over_rate < 1:
                self.cross_over_rate += 0.1
            if self.genesis_rate < 1:
                self.genesis_rate += 0.01
            if self.selection_rate < 0.9:
                self.selection_rate += 0.1
        if len(self.population) > self.initial_population:
            if self.cross_over_rate > 0.1:
                self.cross_over_rate -= 0.1
            if self.genesis_rate > 0.01:
                self.genesis_rate -= 0.01
            if self.selection_rate > 0.01:
                self.selection_rate -= 0.01

        # def __mutation_rate(pop):
        #     b = {i.Genes: True for i in pop}
        #     return float(len(pop) - len(b)) / len(pop)
        # self.mutation_chance = __mutation_rate(self.population)

        self.initial_population += 0.01

    def _display(self, iteration):
        print('~~~~~~~~ iteration: %d ~~~~~~~~' % iteration)
        print('population: len(%d%s)   Total Fitness(%0.4f)' % (
            len(self.population),
            '*' if len(self.population) == self.max_population else '',
            self.fitness
        ))

        print('dynamics: cross_over_rate(%0.2f)   genesis_rate(%0.2f)   selection_rate(%0.2f)   mutation_change(%0.2f)' % (
            self.cross_over_rate,
            self.genesis_rate,
            self.selection_rate,
            self.mutation_chance
        ))
        if self.best:
            print('best: %0.4f   ->   %0.4f  +  %0.4f' % (
                self.best.Fitness,
                self.best.Error,
                self.best.Regularization
            ))
        # for p in self.population:
        #     p.update_fitness()
        #     print('    - %0.4f   ->   %0.4f  +  %0.4f' % (
        #         p.Fitness,
        #         p.Error,
        #         p.Regularization
        #     ))
        # sleep(01.1)
        print('--------------------------------')

    def _validation(self):
        if len(self.population) > self.max_population:
            print('clipping population')
            self.population = self.population[:self.max_population]
        if len(self.population) == 0:
            raise Exception('Population Extincted')
