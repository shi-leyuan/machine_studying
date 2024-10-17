import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "E:/BaiduNetdiskDownload/data_sets/ex1data2.txt"
data = pd.read_csv(path, names=['size', 'bedrooms', 'price'])
print(data)


# 进行特征归一化：为了消除特征间单位和尺度差异的影响，以对每维特征同等看待，需要对特征进行归一化
# 可以使不同的影响因子的权重相同
# z = （x（i）-均值）/方差
# data.mean()求均值
# data.std()求方差
def stand_feature(data):
    return (data - data.mean()) / data.std()


data = stand_feature(data)
print(data)

data.insert(0, 'ones', 1)

x = data.iloc[:, 0:-1]
print(x)
x = x.values
print(x.shape)

y = data.iloc[:, -1]
print(y)

y = y.values
print(y.shape)
y = y.reshape(47, 1)
print(y.shape)


# 损失函数
def cost_func(x, y, theta):
    step1 = np.power(x @ theta - y, 2)
    return np.sum(step1) / (2 * len(x))


theta = np.zeros((3, 1))
cost_init = cost_func(x, y, theta)
print(cost_init)


# 梯度下降
def gradient_Abscent(x, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        theta = theta - alpha * (x.T @ (x @ theta - y)) / len(x)
        cost = cost_func(x, y, theta)
        costs.append(cost)
        if i % 100 == 0:
            print(cost)
    return costs, iters


# 初始化alpha,iters
# 找几个alpha值分别看一下
alpha_arr = [0.0003, 0.003, 0.03, 0.3, 0.0001, 0.001, 0.01, 0.1]
iters = 2000

fig, ax = plt.subplots()
for alpha in alpha_arr:
    costs, iters = gradient_Abscent(x, y, theta, alpha, iters)
    ax.plot(np.arange(iters), costs, label=alpha)
    ax.legend()
ax.set(xlabel='iters',
       ylabel='costs',
       title='iters vs costs')
plt.show()

alpha = 0.003

