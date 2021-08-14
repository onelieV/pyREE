# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/13
"""
from Handler.runfs import go_FS, DATASETS
from BWOA.bwoa_algorithm import BWOA as slctr  # selector alias

import warnings

warnings.filterwarnings('ignore')

# DATASETS = [
#     # 'Wine',
#     # 'Zoo',
#     # 'German',
#     # 'Ionosphere',
#     # 'Spectf',
#     # 'Sonar',
#     # ###########
#     # 'HillValley',
#     # 'Musk1',
#     # 'Madelon',
#     # 'Isolet5',
#     # ##########
#     # 'Yale',
#     # 'Lung',
#     # 'Prostate_GE',
#     # 'Arcene',
# ]

if __name__ == '__main__':
    for ds in DATASETS:
        go_FS(slctr, ds,)
