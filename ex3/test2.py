import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

#                       神经网络前向传播
path = "E:/BaiduNetdiskDownload/data_sets/ex3data1.mat"
data = sio.loadmat(path)
# print(data)

raw_x = data['X']
raw_y = data['y']
x = np.insert(raw_x, 0, values=1, axis=1)
print(x.shape)  # (5000, 401)

# 转换为一维数组
y = raw_y.flatten()
print(y.shape)  # (5000,)

theta = sio.loadmat('E:/BaiduNetdiskDownload/data_sets/ex3weights.mat')
# print(theta)
print(type(theta))  # 字典
print(theta.keys())  # dict_keys(['__header__', '__version__', '__globals__', 'Theta1', 'Theta2'])
# Theta1 是从输入层到隐藏层的权重，Theta2 是从隐藏层到输出层的权重
theta1 = theta['Theta1']
print(theta1.shape)  # (25, 401)
theta2 = theta['Theta2']
print(theta2.shape)  # (10, 26)


# 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


a1 = x  # 第一层输入特征
z2 = x @ theta1.T
a2 = sigmoid(z2)  # 隐藏层输入特征
print(a2.shape)  # (5000, 25)
# 加入偏置项
a2 = np.insert(a2, 0, values=1, axis=1)
print(a2.shape)  # (5000, 26)

z3 = a2 @ theta2.T
a3 = sigmoid(z3)
print(a3.shape)  # (5000, 10)

# np.argmax找到输出层中每一行的最大值的索引，这代表每个样本的预测类别
y_final = np.argmax(a3, axis=1)
# 由于 y 的标签从1开始，所以在预测结果中加1。
y_final = y_final + 1
# 预测
accuracy = np.mean(y_final == y)
print(accuracy)
