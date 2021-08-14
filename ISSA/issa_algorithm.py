# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/14
"""
import numpy as np
from numpy.random import rand
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from Handler.evaluation import Evaluator
from Handler.loadDS import load_ds_name

import warnings

warnings.filterwarnings('ignore')


class ISSA(object):
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
        from paper: # [2020]-"Improved Salp Swarm Algorithm based on opposition based learning and novel local search algorithm for feature selection"
        :return:
        """
        # self.evar.evaluate(self.A.tolist())
        dim = self.X.shape[1]
        max_iter = 100
        N = int(self.maxEva / max_iter)

        lb, ub = 0, 1
        thres = 0.5
        max_local_iter = 10  # maximum iteration for local search

        if np.size(lb) == 1:
            ub = ub * np.ones([1, dim], dtype='float')
            lb = lb * np.ones([1, dim], dtype='float')

        # Follows are the useful functions
        ################################################################
        def init_position(lb, ub, N, dim):
            X = np.zeros([N, dim], dtype='float')
            for i in range(N):
                for d in range(dim):
                    X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
            return X

        def binary_conversion(X, thres, N, dim):
            Xbin = np.zeros([N, dim], dtype='int')
            for i in range(N):
                for d in range(dim):
                    if X[i, d] > thres:
                        Xbin[i, d] = 1
                    else:
                        Xbin[i, d] = 0
            return Xbin

        def boundary(x, lb, ub):
            if x < lb:
                x = lb
            if x > ub:
                x = ub
            return x

        # --- Opposition based learning (7)
        def opposition_based_learning(X, lb, ub, thres, N, dim):
            Xo = np.zeros([N, dim], dtype='float')
            for i in range(N):
                for d in range(dim):
                    Xo[i, d] = lb[0, d] + ub[0, d] - X[i, d]
            return Xo

        ################################################################

        # Initialize position
        X = init_position(lb, ub, N, dim)

        # Pre
        fit = np.zeros([N, 1], dtype='float')
        Xf = np.zeros([1, dim], dtype='float')
        fitF = float('inf')

        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)

        # Fitness
        for i in range(N):
            fit[i, 0] = self._score_sub(Xbin[i])
            if fit[i, 0] < fitF:
                Xf[0, :] = X[i, :]
                fitF = fit[i, 0]

        # --- Opposition based learning
        Xo = opposition_based_learning(X, lb, ub, thres, N, dim)
        # --- Binary conversion
        Xobin = binary_conversion(Xo, thres, N, dim)

        # --- Fitness
        fitO = np.zeros([N, 1], dtype='float')
        for i in range(N):
            fitO[i, 0] = self._score_sub(Xobin[i])
            if fitO[i, 0] < fitF:
                Xf[0, :] = Xo[i, :]
                fitF = fitO[i, 0]

        # --- Merge opposite & current population, and select best N
        XX = np.concatenate((X, Xo), axis=0)
        FF = np.concatenate((fit, fitO), axis=0)
        # --- Sort in ascending order
        ind = np.argsort(FF, axis=0)
        for i in range(N):
            X[i, :] = XX[ind[i, 0], :]
            fit[i, 0] = FF[ind[i, 0]]

        for t in range(max_iter):
            # Compute coefficient, c1 (2)
            c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)

            for i in range(N):
                # First leader update
                if i == 0:
                    for d in range(dim):
                        # Coefficient c2 & c3 [0 ~ 1]
                        c2 = rand()
                        c3 = rand()
                        # Leader update (1)
                        if c3 >= 0.5:
                            X[i, d] = Xf[0, d] + c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])
                        else:
                            X[i, d] = Xf[0, d] - c1 * ((ub[0, d] - lb[0, d]) * c2 + lb[0, d])

                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                        # Salp update
                elif i >= 1:
                    for d in range(dim):
                        # Salp update by following front salp (3)
                        X[i, d] = (X[i, d] + X[i - 1, d]) / 2
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                        # Binary conversion
            Xbin = binary_conversion(X, thres, N, dim)

            # Fitness
            for i in range(N):
                fit[i, 0] = self._score_sub(Xbin[i])
                if fit[i, 0] < fitF:
                    Xf[0, :] = X[i, :]
                    fitF = fit[i, 0]

            # --- Local search algorithm
            Lt = 0
            temp = np.zeros([1, dim], dtype='float')
            temp[0, :] = Xf[0, :]

            while Lt < max_local_iter:
                # --- Random three features
                RD = np.random.permutation(dim)
                for d in range(3):
                    index = RD[d]
                    # --- Flip the selected three features
                    if temp[0, index] > thres:
                        temp[0, index] = temp[0, index] - thres
                    else:
                        temp[0, index] = temp[0, index] + thres

                # --- Binary conversion
                temp_bin = binary_conversion(temp, thres, 1, dim)

                # --- Fitness
                Fnew = self._score_sub(temp_bin[0])
                if Fnew < fitF:
                    fitF = Fnew
                    Xf[0, :] = temp[0, :]

                Lt += 1

            if self.evar.gross_counter >= self.maxEva:
                break


if __name__ == '__main__':
    dataset = 'Yale'
    X, y = load_ds_name(dataset)

    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    slctr = ISSA(X, y, clf, cv, outpath=dataset)
    slctr.fit()
