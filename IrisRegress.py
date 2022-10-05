#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-10-05 14:39
# software: PyCharm


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import pandas as pd

# 导入Logistic回归模型
from LogisticModel import LogisticModel

# 获取鸢尾花数据集
iris = load_iris()
x, y = iris.data, iris.target.reshape(-1, 1).astype('i4')
data = np.append(x, y, axis=1)
df = pd.DataFrame(data)
df.columns = ["花萼长", "花萼宽", "花瓣长", "花瓣宽", "品种"]
print(df)

# 训练集和测试集切分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 训练模型
lm = LogisticModel()
theta_list, cost_list = lm.multi_train(x_train, y_train, 0.01, 2000)
print(f"theta_list={theta_list}\ncost_list={cost_list}")

# 模型准确性分析
print(classification_report(y_test, lm.multi_predict(x_test, theta_list)))
