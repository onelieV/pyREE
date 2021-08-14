# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/11
"""
import numpy as np
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from Handler.evaluation import Evaluator
from Handler.loadDS import load_ds_name

import warnings

warnings.filterwarnings('ignore')


class BDA(object):
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
        https://github.com/JingweiToo/Binary-Dragonfly-Algorithm-for-Feature-Selection/blob/main/jBDA.m
        https://ww2.mathworks.cn/matlabcentral/fileexchange/51032-bda-binary-dragonfly-algorithm
        :return:
        """
        # self.evar.evaluate(self.A.tolist())
        dim = self.X.shape[1]  # 特征维数
        max_Iter = 100  # 最大迭代次数
        N = int(self.maxEva / max_Iter)  # 种群数量  >3000/100

        X = np.random.randint(0, 2, size=(N, dim))  # 初始化
        DX = np.zeros_like(X)

        fitF, fitE = np.inf, -np.inf
        fits = np.zeros(N)
        Xnew = np.zeros_like(X)
        Dmax = 6

        for t in range(max_Iter):
            for i in range(N):
                fits[i] = self._score_sub(X[i])
                if fits[i] < fitF:
                    fitF = fits[i]
                    Xf = X[i]
                if fits[i] > fitE:
                    fitE = fits[i]
                    Xe = X[i]
            w = 0.9 - t * ((0.9 - 0.4) / max_Iter)
            rate = 0.1 - t * ((0.1 - 0) / (max_Iter / 2))
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
                F = Xf - X[i]
                E = Xe + X[i]
                DX[i] = s * S + a * A + c * C + f * F + e * E + w * DX[i]
                DX[i][DX[i] > Dmax] = Dmax
                DX[i][DX[i] < -Dmax] = -Dmax

                TF = np.abs(DX[i] / (DX[i] ** 2 + 1) ** 0.5)
                Xnew[i] = X[i]
                index = np.random.rand(*TF.shape) < TF
                Xnew[i][index] = 1 - X[i][index]

            X = deepcopy(Xnew)


if __name__ == '__main__':
    dataset = 'Yale'
    X, y = load_ds_name(dataset)

    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    slctr = BDA(X, y, clf, cv, outpath=dataset)
    slctr.fit()
