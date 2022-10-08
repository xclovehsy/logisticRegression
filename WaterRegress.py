#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: 徐聪
# datetime: 2022-10-04 21:17
# software: PyCharm

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# 导入Logistic回归模型
from LogisticModel import LogisticModel

# 初始化设置
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

# 西瓜数据集 3.0
data = np.array(
    [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1], [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
     [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0], [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
     [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0], [0.593, 0.042, 0], [0.719, 0.103, 0]])
label = np.array(["密度", "含糖率", "好瓜"])
cnt, dim = data.shape[0], data.shape[1] - 1
df = pd.DataFrame(data)
df.columns = label
x = data[:, 0:2]
y = data[:, 2].reshape(cnt, 1)

# 绘制图像
print("西瓜数据集3.0:\n")
print(df)
plt.figure()
plt.plot(x[0:8, 0], x[0:8, 1], "bo", label="好瓜")
plt.plot(x[8:16, 0], x[0:8, 1], "ro", label="坏瓜")
plt.xlabel("密度")
plt.ylabel("含糖率")
plt.legend()
plt.grid()
# plt.show()


# 获取训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 训练模型
lm = LogisticModel()
# theta, cost_list = lm.train(x_train, y_train, 0.05, 1000)
theta, cost_list = lm.train(x, y, 0.01, 5000)

# 模型评估
print(f"params = {theta}")
print(classification_report(y, lm.predict(x, theta)))




