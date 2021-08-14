# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/3/8
"""

import ASUdataset.load_asu as asu
import UCIdataset.load_uci as uci


def load_ds_name(name):
    if name in asu.DATASETS:
        return asu.load(name)
    elif name in uci.DATASETS:
        return uci.load(name)
    else:
        print('The dataset {} is not in our repository!'.format(name))
        return None, None


if __name__ == '__main__':
    X, y = load_ds_name('GO')
    print(X)
