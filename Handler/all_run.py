# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2020/11/17
"""
import os

# ALL_Algorithms = ['BDA', 'HLBDA', 'REE', 'REC']

Algs = ['BDA', 'BWOA', 'HLBDA', 'ISSA', 'NSGAII', 'TMGWO', 'REE']

# 运行的数据集和独立次数需要到 runfs.py里去修改
for alg in Algs:
    os.chdir("../{}".format(alg))
    os.system("python {}_RUN.py".format(alg))
