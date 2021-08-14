# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2020/12/10
"""
import random
import numpy as np
from sklearn.base import clone
from deap import algorithms, base, creator, tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from Handler.evaluation import Evaluator
from Handler.loadDS import load_ds_name

import warnings

warnings.filterwarnings('ignore')


class NSGAII(object):
    def __init__(self, X, y, clf, cv, outpath="dataset"):
        self.X, self.y, self.clf, self.cv = X, y, clf, cv
        self.evar = Evaluator(X, y, clf, cv, outpath)
        self.maxEva = max(3000, 100 * int(X.shape[1] ** 0.5))
        print("\n最大评估次数：", self.maxEva)
        self.A = np.arange(X.shape[1])

    def _score_sub(self, ind):
        """
        :param ind: ind must be [0,1,..,0,1] type array.
        :return:
        """
        ind = np.array(ind)
        subset = self.A[np.where(ind == 1)]
        eva = self.evar.evaluate(subset)
        return eva, len(subset)

    def fit(self):
        """
        sourcecode from paper: DEAP: Evolutionary Algorithms Made Easy
        :return:
        """
        # self.evar.evaluate(self.A.tolist())
        n = self.X.shape[1]  # 特征维数
        n_generation = 100
        n_population = int(self.maxEva / n_generation)

        cxpb = 0.9  # 交叉概率
        mutpb = 1 / n  # 变异概率

        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Accuracy max 1.0, number min -1.0
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        tbx = base.Toolbox()
        tbx.register("bit", random.randint, 0, 1)
        tbx.register('individual', tools.initRepeat, creator.Individual, tbx.bit, n)
        tbx.register('population', tools.initRepeat, list, tbx.individual, n=n_population)
        tbx.register("evaluate", self._score_sub)
        tbx.register("mate", tools.cxUniform, indpb=cxpb)
        tbx.register("mutate", tools.mutFlipBit, indpb=mutpb)
        tbx.register("select", tools.selNSGA2)

        population = tbx.population()
        fits = tbx.map(tbx.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit

        for gen in range(n_generation - 1):
            offspring = algorithms.varOr(population, tbx, lambda_=n_population, cxpb=cxpb, mutpb=mutpb)
            fits = tbx.map(tbx.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            population = tbx.select(population + offspring, k=n_population)

        # best = tools.selBest(population, k=1)[0]
        # self.opt_fitness = best.fitness.values[0]
        # self.opt_subset = [idx for (idx, i) in enumerate(best) if i == 1]
        # print(best)


if __name__ == '__main__':
    dataset = 'Yale'
    X, y = load_ds_name(dataset)

    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    slctr = NSGAIISelection(X, y, clf, cv, outpath=dataset)
    slctr.fit()
