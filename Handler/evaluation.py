# -*- coding: utf-8 -*-
"""
Created by: Veloci
Created on: 2020/10/27
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, cross_validate


class Evaluator(object):
    def __init__(self, X, y, clf, cv, result_name='default'):  # input training dataset
        self.X, self.y, self.clf, self.cv, = X, y, clf, cv
        self.ReName = result_name
        self.HASH = dict()  # 一个用子集哈希值索引的字典，用于记录一次运行时评估过的子集的评估值
        self.gross_counter = 0  # 总评估次数计数器
        self.net_counter = 0  # 净次数计数器
        self.best_ = (0, 0, 0)  # , [])  # (目标精度，特征数量, 产生序号)#, 索引)
        with open(self.ReName, 'w') as f:  # 生成一个文件记录评估过程，用于后续分析
            f.write('')
            # f.write("Create time {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
            # f.write("gross,  net,  obj_acc, train_acc, test_acc,  no.,  subset_hash_key\n")

    def evaluate(self, subset):  # 评估一个子集在训练数据集上的性能时，顺便计算了该子集在测试集上的性能
        # subset must be list form with the indices for example [0,2,4,10,11]
        hs = hash(str(subset))
        self.gross_counter += 1
        if hs in self.HASH:
            cv_ = self.HASH[hs]  # 如果已存在则直接获得，不用调用学习算法，节省时间
        else:
            if not len(subset):  # subset为空
                cv_ = 0
            else:
                # scores = cross_validate(clone(self.clf), self.X[:, subset], self.y, cv=self.cv, n_jobs=-1,
                #                         scoring=('accuracy',))
                # cv_ = scores['test_accuracy'].mean().round(5)
                # tmp_clf = clone(self.clf).fit(self.X[:, subset], self.y)
                # tr_ = tmp_clf.score(self.X[:, subset], self.y).__round__(5)
                # ts_ = tmp_clf.score(self.X_s[:, subset], self.y_s).__round__(5)
                scores = cross_val_score(clone(self.clf), self.X[:, subset], self.y, cv=self.cv, n_jobs=-1)
                cv_ = scores.mean().round(5)
            self.net_counter += 1
            self.HASH[hs] = cv_  # 新子集评估后放入字典

        with open(self.ReName, 'a') as f:
            f.write("{:>4d}, ".format(self.gross_counter))  # 评价序号
            f.write("{:>4d}, ".format(self.net_counter))  # 净评价个数
            f.write("({:>8.5f}, ".format(cv_))  # 评估值  # cv_
            # f.write("{:>8.5f}, ".format(tr_))  # 训练评估值  # tr_
            # f.write("{:>8.5f}, ".format(ts_))  # 测试评估值  # ts_
            f.write("{:>4d}), ".format(len(subset)))  # 子集大小
            f.write("{}".format(hs))  # 子集哈希
            # f.write("{}".format(subset))  # 子集
            f.write("\n")

        # 目标精度越大，数量越少更好
        self.best_ = max(self.best_, (cv_, len(subset), self.gross_counter),
                         key=lambda t: (t[0], -t[1], -t[2]))

        print('\r', self.net_counter, '/', self.gross_counter, cv_, len(subset), end='')
        print(" optimizing objectives", self.best_, end='')
        return cv_
