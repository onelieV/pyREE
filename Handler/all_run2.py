# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/14
"""

from Handler.runfs import go_FS

from BDA.bda_algorithm import BDA
from BWOA.bwoa_algorithm import BWOA
from HLBDA.hlbda_algorithm import HLBDA
from ISSA.issa_algorithm import ISSA
from NSGAII.nsga_algorithm import NSGAII
from REE.ree_algorithm import REE
from TMGWO.tmgwo_algorithm import TMGWO

algorithms = {
    'BDA': BDA,
    'BWOA': BWOA,
    'HLBDA': HLBDA,
    'ISSA': ISSA,
    'TMGWO': TMGWO,
    'NSGAII': NSGAII,
    'REE': REE,
}

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
    'Isolet5',
    # ##########
    # 'Yale',
    # 'Lung',
    # 'Prostate_GE',
    # 'Arcene',
]

if __name__ == '__main__':
    # 下面两个循环，决定是数据集优先还是算法优先
    for ds in DATASETS:  # 实验数据集
        for name, algor in algorithms.items():  # 使用算法
            go_FS(algor, ds, [i for i in range(1, 20)])  # 第3个参数指明仿真序号，默认设置None执行[r for r in range(20)]
    #
    # for i in range(1, 20):  # 相同序号的实验结果，所有算法同一批执行
    #     for ds in DATASETS:  # 实验数据集
    #         for name, algor in algorithms.items():  # 使用算法
    #             go_FS(algor, ds, [i])  # 第3个参数指明仿真序号，默认设置None执行[r for r in range(20)]
