# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/10
"""
import numpy as np
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from Handler.evaluation import Evaluator
from Handler.loadDS import load_ds_name

import warnings

warnings.filterwarnings('ignore')


class BWOA(object):
    def __init__(self, X, y, clf, cv, outpath='dataset'):
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
        return 0.99 * (1 - eva) + 0.01 * sum(ind) / len(self.A)

    def fit(self):
        """
        https://github.com/ZongSingHuang/Binary-Whale-Optimization-Algorithm/blob/master/BWOA.py
        https://link.springer.com/article/10.1007/s13042-019-00996-5
        :return:
        """
        # self.evar.evaluate(self.A.tolist())
        num_dim = self.X.shape[1]
        max_iter = 100
        num_particle = int(self.maxEva / max_iter)

        x_min, x_max = 0, 1
        # X = 1 * (np.random.uniform(low=x_min, high=x_max, size=[num_particle, num_dim]) > 0.5)
        X = np.random.randint(0, 2, size=(num_particle, num_dim))
        gBest_X = None
        gBest_score = np.inf
        Lbest = []

        b = 1
        for iter in range(max_iter):
            a = 2 * (1 - iter / max_iter)
            for i in range(num_particle):
                score = self._score_sub(X[i])

                if score <= gBest_score:
                    gBest_X = X[i].copy()  # 后面的操作直接在X上进行更改，所以这里要用copy
                    gBest_score = score
                    Lbest.append(X[i].copy())
                    if len(Lbest) > 3:
                        Lbest = Lbest[-3:]
                    # fake_X = gBest_X.copy()

                if iter > max_iter / 3:
                    idx = int(np.random.randint(low=0, high=len(Lbest), size=1))
                    # fake_X = Lbest[idx].copy()
                    gBest_X = Lbest[idx].copy()

                r1 = np.random.uniform()
                r2 = np.random.uniform()
                A = 2 * a * r1 - a
                C = 2 * r2
                l = np.random.uniform(low=-1, high=1)
                p = np.random.uniform()

                for j in range(num_dim):
                    rd = np.random.uniform()
                    if p < 0.4:
                        D = np.abs(C * gBest_X[j] - X[i, j])
                        TF = np.abs(np.pi / 3 * np.arctan(np.pi / 3 * A * D) + 0.02)
                        if np.abs(A) < 1:
                            if rd < TF:
                                X[i, j] = 1 - X[i, j]
                            else:
                                X[i, j] = gBest_X[j].copy()
                        else:
                            if rd < TF:
                                X[i, j] = 1 - X[np.random.randint(low=0, high=num_particle, size=1), j]
                            else:
                                X[i, j] = X[np.random.randint(low=0, high=num_particle, size=1), j].copy()
                    else:
                        D = np.abs(gBest_X[j] - X[i, j])
                        S = D * np.exp(b * l) * np.cos(2 * np.pi * l)
                        TF = np.abs(np.arctan(S) + 0.09) / 4
                        if rd > 0.92 and TF == 0.09 / 4:
                            X[i, j] = 1 - gBest_X[j]
                        elif rd > TF:
                            X[i, j] = gBest_X[j].copy()
                        else:
                            X[i, j] = X[i, j].copy()


if __name__ == '__main__':
    dataset = 'Zoo'
    X, y = load_ds_name(dataset)

    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    slctr = BWOA(X, y, clf, cv, outpath=dataset)
    slctr.fit()
