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


class TMGWO(object):
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
        from paper: A new fusion of grey wolf optimizer algorithm with a two-phase mutation for feature selection
        :return:
        """
        # self.evar.evaluate(self.A.tolist())
        dim = self.X.shape[1]
        max_iter = 100
        N = int(self.maxEva / max_iter)

        lb, ub = 0, 1
        thres = 0.5
        Mp = 0.5  # mutation probability

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

        # --- transfer function update binary position (4.3.2)
        def transfer_function(x):
            Xs = abs(np.tanh(x))
            return Xs

        ################################################################

        if np.size(lb) == 1:
            ub = ub * np.ones([1, dim], dtype='float')
            lb = lb * np.ones([1, dim], dtype='float')

        # Initialize position
        X = init_position(lb, ub, N, dim)

        # --- Binary conversion
        X = binary_conversion(X, thres, N, dim)

        # Fitness at first iteration
        fit = np.zeros([N, 1], dtype='float')
        Xalpha = np.zeros([1, dim], dtype='int')
        Xbeta = np.zeros([1, dim], dtype='int')
        Xdelta = np.zeros([1, dim], dtype='int')
        Falpha = float('inf')
        Fbeta = float('inf')
        Fdelta = float('inf')

        for i in range(N):
            fit[i, 0] = self._score_sub(X[i])
            if fit[i, 0] < Falpha:
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]
            if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]
            if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]

        for t in range(max_iter):
            # Coefficient decreases linearly from 2 to 0 (3.5)
            a = 2 - t * (2 / max_iter)

            for i in range(N):
                for d in range(dim):
                    # Parameter C (3.4)
                    C1 = 2 * rand()
                    C2 = 2 * rand()
                    C3 = 2 * rand()
                    # Compute Dalpha, Dbeta & Ddelta (3.7 - 3.9)
                    Dalpha = abs(C1 * Xalpha[0, d] - X[i, d])
                    Dbeta = abs(C2 * Xbeta[0, d] - X[i, d])
                    Ddelta = abs(C3 * Xdelta[0, d] - X[i, d])
                    # Parameter A (3.3)
                    A1 = 2 * a * rand() - a
                    A2 = 2 * a * rand() - a
                    A3 = 2 * a * rand() - a
                    # Compute X1, X2 & X3 (3.7 -3.9)
                    X1 = Xalpha[0, d] - A1 * Dalpha
                    X2 = Xbeta[0, d] - A2 * Dbeta
                    X3 = Xdelta[0, d] - A3 * Ddelta
                    # Update wolf (3.6)
                    Xn = (X1 + X2 + X3) / 3
                    # --- transfer function
                    Xs = transfer_function(Xn)
                    # --- update position (4.3.2)
                    if rand() < Xs:
                        X[i, d] = 0
                    else:
                        X[i, d] = 1

            # Fitness
            for i in range(N):
                fit[i, 0] = self._score_sub(X[i])
                if fit[i, 0] < Falpha:
                    Xalpha[0, :] = X[i, :]
                    Falpha = fit[i, 0]
                if fit[i, 0] < Fbeta and fit[i, 0] > Falpha:
                    Xbeta[0, :] = X[i, :]
                    Fbeta = fit[i, 0]
                if fit[i, 0] < Fdelta and fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                    Xdelta[0, :] = X[i, :]
                    Fdelta = fit[i, 0]

            # --- two phase mutation: first phase
            # find index of 1
            idx = np.where(Xalpha == 1)
            idx1 = idx[1]
            Xmut1 = np.zeros([1, dim], dtype='int')
            Xmut1[0, :] = Xalpha[0, :]
            for d in range(len(idx1)):
                r = rand()
                if r < Mp:
                    Xmut1[0, idx1[d]] = 0
                    Fnew1 = self._score_sub(Xmut1[0])
                    if Fnew1 < Falpha:
                        Falpha = Fnew1
                        Xalpha[0, :] = Xmut1[0, :]

            # --- two phase mutation: second phase
            # find index of 0
            idx = np.where(Xalpha == 0)
            idx0 = idx[1]
            Xmut2 = np.zeros([1, dim], dtype='int')
            Xmut2[0, :] = Xalpha[0, :]
            for d in range(len(idx0)):
                r = rand()
                if r < Mp:
                    Xmut2[0, idx0[d]] = 1
                    Fnew2 = self._score_sub(Xmut2[0])
                    if Fnew2 < Falpha:
                        Falpha = Fnew2
                        Xalpha[0, :] = Xmut2[0, :]

            if self.evar.gross_counter >= self.maxEva:
                break


if __name__ == '__main__':
    dataset = 'Yale'
    X, y = load_ds_name(dataset)

    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    slctr = TMGWO(X, y, clf, cv, outpath=dataset)
    slctr.fit()
