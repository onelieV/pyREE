# -*- coding: utf-8 -*-
"""
Created on 2019-10-xx
@author: Veileno
"""
import sys
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

from Handler.evaluation import Evaluator
from Handler.loadDS import load_ds_name

import warnings

warnings.filterwarnings('ignore')


# sys.setrecursionlimit(2000)


class REE(object):
    def __init__(self, X, y, clf, cv, outpath="dataset"):
        self.X, self.y, self.clf, self.cv = X, y, clf, cv
        self.evar = Evaluator(X, y, clf, cv, outpath)
        self.maxEva = max(3000, 100 * int(X.shape[1] ** 0.5))
        print("\n最大评估次数：", self.maxEva)
        # self.opt_fitness = 0
        self.opt_subset = []

    def betterSubOf(self, e0, S0):  # e0, S0
        gap = len(S0) ** 0.5 if len(S0) > 100 else 2
        # gap = 2

        def randBisectElim(S, E):
            if len(E) < gap or self.evar.gross_counter >= self.maxEva:
                return 0, []
            else:
                random.shuffle(E)
                E_left, E_right = E[:int(0.5 * len(E))], E[int(0.5 * len(E)):]  # randomly bisect E
                S_left, S_right = ([f for f in S if f not in EE] for EE in (E_left, E_right))
                eva_L, eva_R = self.evar.evaluate(S_left), self.evar.evaluate(S_right)
                e_t, S_t, E_t = max((eva_L, S_left, E_left), (eva_R, S_right, E_right),
                                    key=lambda t: (t[0], -len(t[1])))
                e_n, S_n = randBisectElim(S.copy(), E_t.copy())
                return max((e_t, S_t), (e_n, S_n), key=lambda t: (t[0], -len(t[1])))

        return randBisectElim(S0.copy(), S0.copy())  # 调用的时候，一定要用列表参数的copy, !!!make sure using copy()!!!

    def greedyBinElect(self, e, S):  # optimaSubOf # 2020/11/18
        elite_Q = [(e, S)]
        T = [self.betterSubOf(e, S) for _ in (0, 1)]
        for ee, SS in T:
            if ee >= e:
                e_m, S_m = self.greedyBinElect(ee, SS)
                elite_Q.append((e_m, S_m))
        return max(elite_Q, key=lambda t: (t[0], -len(t[1])))

    def fit(self):
        A = [i for i in range(self.X.shape[1])]
        e = self.evar.evaluate(A)
        e_tmp, S_tmp = e, A
        while self.evar.gross_counter < self.maxEva:
            e_tmp, S_tmp = max((e_tmp, S_tmp), self.greedyBinElect(e, A))
        self.opt_subset = S_tmp
        return self


if __name__ == '__main__':
    import time

    dataset = 'Madelon'
    X, y = load_ds_name(dataset)

    clf = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    start_time = time.time()

    slctr = REE(X, y, clf, cv, outpath=dataset)
    slctr.fit()

    print(' 运行时长{:.3f}秒'.format(time.time() - start_time))
