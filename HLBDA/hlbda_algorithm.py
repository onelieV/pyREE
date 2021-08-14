# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/12
"""
import numpy as np
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from Handler.evaluation import Evaluator
from Handler.loadDS import load_ds_name

import warnings

warnings.filterwarnings('ignore')


class HLBDA(object):
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
        return 0.99 * (1 - eva) + 0.01 * sum(ind) / len(self.A)

    def fit(self):
        """
        The sourcecodes are from in Matlab code.
        https://seyedalimirjalili.com/da
        :return:
        """
        # self.evar.evaluate(self.A.tolist())
        D = self.X.shape[1]  # 特征维数
        T = 100  # 最大迭代次数
        N = int(self.maxEva / T)  # 种群数量  >3000/100
        pp, pg = 0.4, 0.7

        X = np.random.randint(0, 2, size=(N, D))
        DX = np.zeros_like(X)

        fitness = np.zeros(N)
        fitF, fitE = np.inf, -np.inf
        Xnew = np.zeros_like(X)
        Dmax = 6

        fitPB = np.ones_like(fitness)
        X_pb = np.zeros_like(X)
        fitPW = np.zeros_like(fitness)
        X_pw = np.zeros_like(X)

        for t in range(T):
            for i in range(N):
                fitness[i] = self._score_sub(X[i])
                if fitness[i] < fitF:
                    fitF, Xf = fitness[i], X[i]
                if fitness[i] > fitE:
                    fitE, Xe = fitness[i], X[i]
                if fitness[i] > fitPW[i]:
                    fitPW[i], X_pw[i] = fitness[i], X[i]
                if fitness[i] < fitPB[i]:
                    fitPB[i], X_pb[i] = fitness[i], X[i]

            w = 0.9 - t * ((0.9 - 0.4) / T)
            rate = 0.1 - t * ((0.1 - 0) / (T / 2))
            if rate < 0:
                rate = 0
            s = 2 * np.random.rand() * rate
            a = 2 * np.random.rand() * rate
            c = 2 * np.random.rand() * rate
            f = 2 * np.random.rand()
            e = rate

            Xn, DXn = deepcopy(X), deepcopy(DX)

            for i in range(N):
                S = -np.sum(Xn - X[i], axis=0)
                A = (np.sum(DXn, axis=0) - DXn[i]) / (N - 1)
                C = (np.sum(Xn, axis=0) - Xn[i]) / (N - 1) - X[i]
                F = ((X_pb[i] - X[i]) + (Xf - X[i])) / 2
                E = ((X_pw[i] + X[i]) + (Xe + X[i])) / 2
                DX[i] = s * S + a * A + c * C + f * F + e * E + w * DX[i]
                DX[i][DX[i] > Dmax] = Dmax
                DX[i][DX[i] < -Dmax] = -Dmax

                TF = np.abs(DX[i] / (DX[i] ** 2 + 1) ** 0.5)
                R1 = np.random.rand(*TF.shape)

                Xnew[i] = Xf
                index1 = (0 <= R1) == (R1 < pp)
                Xnew[i][index1] = X[i][index1]
                index2 = index1 == (np.random.rand(*TF.shape) < TF)
                Xnew[i][index2] = 1 - X[i][index2]
                index3 = (pp <= R1) == (R1 < pg)
                Xnew[i][index3] = X_pb[i][index3]

            X = deepcopy(Xnew)


if __name__ == '__main__':
    dataset = 'Yale'
    X, y = load_ds_name(dataset)

    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    slctr = HLBDA(X, y, clf, cv, outpath=dataset)
    slctr.fit()
