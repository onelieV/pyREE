# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2021/2/2
"""

import numpy as np

DATASETS = {
    # 'Arcene': 'load_Arcene',
    'German': 'load_German',
    'HillValley': 'load_HillValley',
    'Ionosphere': 'load_Ionosphere',
    'Isolet5': 'load_Isolet5',
    'Madelon': 'load_Madelon',
    'Musk1': 'load_Musk1',
    'Sonar': 'load_Sonar',
    'Soybean': 'load_Soybean',
    'Spectf': 'load_Spectf',
    'WDBC': 'load_WDBC',
    'Wine': 'load_Wine',
    'Zoo': 'load_Zoo',
}


#############basic function################
def load_X(path, start_line=0, sep=','):
    lines = [line.strip('\n').strip() for line in open(path)][start_line:]  # 每行去掉换行符、首尾空格;从start_line开始
    X_ls = [[eval(x) for x in line.split(sep)] for line in lines]  # 按分割符划分并读取成二维列表
    return np.asarray(X_ls)


def load_y(path, start_line=0):
    lines = [line.strip('\n').strip() for line in open(path)][start_line:]  # 每行去掉换行符、首尾空格;从start_line开始
    y_ls = lines
    return np.asarray(y_ls)


def load_X_y(path, start_line=0, sep=',', y_index=-1, start_column=0):
    lines = [line.strip('\n').strip() for line in open(path)][start_line:]  # 每行去掉换行符、首尾空格;从start_line开始
    X_y_str_ls = [[x.strip() for x in line.split(sep)][start_column:] for line in lines]  # 每行按分隔符拆开，取Xy内容
    y_ls = [line.pop(y_index) for line in X_y_str_ls]  # 弹出标签数据y, X_y_str_ls中标签数据y已被删除
    X_ls = [[eval(x) for x in line] for line in X_y_str_ls]  # 重新整合x数据并读取为数值列表
    # X_ls = [eval('[{}]'.format(','.join(line))) for line in X_y_str_ls]  # 重新整合x数据并读取为数值列表
    return np.asarray(X_ls), np.asarray(y_ls)


def load(dataset):
    if dataset in DATASETS:
        return eval(DATASETS[dataset])()
    else:
        print('dataset {} is invalid'.format(dataset))


#################load_DataSet()#################

# def load_Arcene():
#     name = 'Arcene'
#     path_X = '../UCIdataset/Arcene/arcene_train.data'
#     path_y = '../UCIdataset/Arcene/arcene_train.labels'
#     X = load_X(path_X, sep=None)
#     y = load_y(path_y)
#     print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
#     return X, y


def load_German():
    name = 'German'
    path = '../UCIdataset/German/german.data-numeric'
    X, y = load_X_y(path, start_line=0, sep=None, y_index=-1)  # 任意多空格时使用sep=None
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


# def load_Gisette(): #数据量太大，不建议使用
#     name = 'Gisette'
#     path_X = '../UCIdataset/Gisette/gisette_train.data'
#     path_y = '../UCIdataset/Gisette/gisette_train.labels'
#     X = load_X(path_X, sep=None)
#     y = load_y(path_y)
#     print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
#     return X, y


def load_HillValley():
    name = 'HillValley'
    path = '../UCIdataset/HillValley/Hill_Valley_without_noise_Training.data'
    X, y = load_X_y(path, start_line=1, sep=',', y_index=-1)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Ionosphere():
    name = 'Ionosphere'
    path = '../UCIdataset/Ionosphere/ionosphere.data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=-1)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Isolet5():
    name = 'Isolet5'
    path = '../UCIdataset/Isolet5/isolet5.data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=-1)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Madelon():
    name = 'Madelon'
    path_X = '../UCIdataset/Madelon/madelon_train.data'
    path_y = '../UCIdataset/Madelon/madelon_train.labels'
    X = load_X(path_X, sep=None)
    y = load_y(path_y)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Musk1():
    name = 'Musk1'
    path = '../UCIdataset/Musk1/clean1.data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=-1, start_column=2)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Sonar():
    name = 'Sonar'
    path = '../UCIdataset/Sonar/sonar.all-data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=-1)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Soybean():
    name = 'Soybean'
    path = '../UCIdataset/Soybean/soybean-small.data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=-1)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Spectf():
    name = 'Spectf'
    path_1 = '../UCIdataset/Spectf/SPECTF.train'
    X1, y1 = load_X_y(path_1, start_line=0, sep=',', y_index=0)
    path_2 = '../UCIdataset/Spectf/SPECTF.test'
    X2, y2 = load_X_y(path_2, start_line=0, sep=',', y_index=0)
    X, y = np.vstack((X1, X2)), np.hstack((y1, y2))
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_WDBC():
    name = 'WDBC'
    path = '../UCIdataset/WDBC/wdbc.data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=0, start_column=1)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Wine():
    name = 'Wine'
    path = '../UCIdataset/Wine/wine.data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=0)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


def load_Zoo():
    name = 'Zoo'
    path = '../UCIdataset/Zoo/zoo.data'
    X, y = load_X_y(path, start_line=0, sep=',', y_index=-1, start_column=1)
    print("数据集 " + name, "样本*维度 ", X.shape, "类别：数量", {l: y.tolist().count(l) for l in np.unique(y)})
    return X, y


if __name__ == '__main__':
    load_Wine()
    load_Zoo()
    load_German()
    load_Ionosphere()
    load_Spectf()
    load_Sonar()

    load_HillValley()
    load_Musk1()
    # load_Madelon()
    # load_Isolet5()

    load_WDBC()
    load_Soybean()

    # load_Arcene()

    # eval(DATASETS['HillValley'])()

    # Xt, yt, Xs, ys = eval(DATASETS['HillValley'])()
    #
    # # print(Xt)
    # # print(yt)
    #
    # split_ratio = (3, 1, 1)  # (train, validate, test) or (train, test)
    #
    # clusters = {label: Xt[np.where(yt == label)] for label in np.unique(yt)}
    #
    #
    # # print(clusters.keys())
    #
    # def split_matrix(X):
    #     x_mean = np.average(X, axis=0)
    #     dist_ls = [(np.sqrt(np.sum((x - x_mean) ** 2)), x) for x in X]
    #     dist_ls.sort(key=lambda t: t[0], reverse=True)
    #     X_ls_descent = [x for (_, x) in dist_ls]
    #     sub_X_ls = [[] for _ in split_ratio]
    #     ind = 0
    #     while ind < len(X_ls_descent):
    #         for order, i in enumerate(split_ratio[::-1]):
    #             sub_X_ls[order].extend(X_ls_descent[ind:ind + i])
    #             ind += i
    #
    #     return sub_X_ls[::-1]
    #
    #
    # clusters_splited = {key: split_matrix(val) for key, val in clusters.items()}
    #
    # X_splits = [([], []) for _ in split_ratio]
    # for i, (X, y) in enumerate(X_splits):
    #     for key, val in clusters_splited.items():
    #         X_splits[i][0].extend(val[i])
    #         X_splits[i][1].extend([key] * len(val[i]))
    #
    # for X, y in X_splits:
    #     print(len(X), len(y))
    #     print(y)

    # split_sets = []
    # for num in split_ratio:
    #     for label in clusters.keys():
    #         pass
