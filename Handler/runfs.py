# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2020/11/18
"""
import os
import time
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from Handler.loadDS import load_ds_name

DATASETS = [
    # 'Wine',
    # 'Zoo',
    # 'German',
    # 'Ionosphere',
    # 'Spectf',
    # 'Sonar',
    # ###########
    # 'HillValley',
    # 'Musk1',
    # 'Madelon',
    # 'Isolet5',
    # ##########
    # 'Yale',
    # 'Lung',
    # 'Prostate_GE',
    # 'Arcene',
]


def go_FS(slctr, ds, repeated=None):  # repeated=None
    if not repeated:
        repeated = [r for r in range(20)]
    elif not isinstance(repeated, list):
        print("'repeated' must be a list/tuple with the independent run serial number!")
        raise ValueError

    print("Start the process of {} on {}".format(slctr.__name__, ds).center(100, '*'))
    X, y = load_ds_name(ds)
    result_path = "../XResult/{}/{}".format(slctr.__name__, ds)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if X.shape[1] >= 500 and X.shape[0] > 1000:
        X, X_, y, y_ = train_test_split(X, y, train_size=500, stratify=y, random_state=0)

    print(X.shape)

    # X = MinMaxScaler().fit_transform(X)
    clf = KNeighborsClassifier(n_neighbors=5)
    # clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for r in repeated:
        selector = slctr(X, y, clone(clf), CV, outpath=result_path + "/{}_{}".format(ds, r))
        t_begin = time.time()
        selector.fit()
        t_done = time.time()
        print("  {}_{} completed!".format(ds, r), "用时：{:.0f}s".format(t_done - t_begin))

    print("End the process of {} on {}".format(slctr.__name__, ds).center(100, '='))
    print("\n")
