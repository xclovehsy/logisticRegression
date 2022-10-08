#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-10-04 19:25
# software: PyCharm

import numpy as np


class LogisticModel:
    def __init__(self):
        pass

    def h_theta(self, X, theta):
        """
        计算逻辑回归定义式
        :param X: 特征向量
        :param theta: 参数
        :return:
        """
        return self.sigmoid(np.dot(X, theta.T))

    def train(self, x, y, alpha, epochs):
        """
        对数回归模型训练函数
        :param x: 特征向量
        :param y: 标记
        :param alpha: 学习率
        :param epochs: 训练次数
        :return:
        """
        # 初始化参数以及数据
        num_train, num_feature = x.shape
        X = np.append(np.ones((num_train, 1)), x, axis=1)
        theta = np.zeros((1, num_feature + 1))
        cost_list = []

        # 训练模型
        for epoch in range(epochs):
            h_theta_x = self.h_theta(X, theta)
            # 损失值
            cost = -1 / num_train * np.sum(y * np.log(h_theta_x) + (1 - y) * np.log(1 - h_theta_x))
            # 计算theta偏导
            d_theta = 1 / num_train * np.sum((h_theta_x - y) * X, axis=0)
            # 更新theta
            theta = theta - alpha * d_theta

            if epoch % 100 == 0:
                cost_list.append(cost)
                print(f"epoch={epoch}, cost={cost}")

        return theta, cost_list

    def predict(self, x, theta):
        """
        二分类模型预测
        :param x: 特征向量
        :param theta: 参数
        :return:
        """
        num_predict, num_feature = x.shape
        X = np.append(np.ones((num_predict, 1)), x, axis=1)
        y_predict = self.h_theta(X, theta)
        for i in range(len(y_predict)):
            if y_predict[i] > 0.5:
                y_predict[i] = 1
            else:
                y_predict[i] = 0
        return y_predict

    def sigmoid(self, x):
        """
        定义sigmoid函数
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    def multi_predict(self, x, theta_list):
        """
        多分类器预测
        :param x: 特征向量
        :param theta_list: 参数
        :return:
        """
        num_predict, num_feature = x.shape
        X = np.append(np.ones((num_predict, 1)), x, axis=1)

        Y_predict = self.h_theta(X, theta_list[0])
        y_predict = np.ones((num_predict, 1))
        for theta in theta_list[1:]:
            Y_predict = np.append(Y_predict, self.h_theta(X, theta), axis=1)

        # print(Y_predict)

        for i in range(num_predict):
            y_predict[i] = np.argmax(Y_predict[i, :])
        return y_predict


    def multi_train(self, x, y, alpha, epochs):
        """
        多分类器训练
        :param x: 特征向量
        :param y: 标签
        :param alpha: 学习率
        :param epochs: 训练次数
        :return:
        """
        # 获取数据集参数
        sort_list = np.unique(y)
        num_sort = len(sort_list)
        num_train, num_feature = x.shape

        theta_list = []
        cost_list = []
        for i in range(num_sort):
            print(f"==== classifier {sort_list[i]} train begin ====")
            # 获取当前分类label
            sort = sort_list[i]
            # 将当前分类的label变为1，其余变为0
            Y = np.copy(y)
            Y[Y != sort] = -1
            Y[Y == sort] = 1
            Y[Y == -1] = 0

            # 存储参数
            theta, costs = self.train(x, Y, alpha, epochs)
            theta_list.append(theta)
            cost_list.append(costs[-1])

        return theta_list, cost_list
