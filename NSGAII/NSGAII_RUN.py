# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2020/11/18
"""
from Handler.runfs import go_FS, DATASETS
from NSGAII.nsga_algorithm import NSGAII as slctr  # selector alias

import warnings

warnings.filterwarnings('ignore')

# DATASETS = [
#     'Wine',
#     'Zoo',
#     'German',
#     'Ionosphere',
#     'Spectf',
#     'Sonar',
#     ###########
#     'HillValley',
#     'Musk1',
#     'Madelon',
#     'Isolet5',
#     ##########
#     'Yale',
#     'Lung',
#     'Prostate_GE',
#     'Arcene',
# ]

if __name__ == '__main__':
    for ds in DATASETS:
        go_FS(slctr, ds)
