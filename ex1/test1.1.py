import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "E:/BaiduNetdiskDownload/data_sets/ex1data1.txt"
data = pd.read_csv(path, names=['population', 'profit'])
print(data)

data.insert(loc=0, column='ones', value=1)
print(data)

data.plot.scatter('population', 'profit')
plt.show()

x = data.iloc[:, 0:-1]
print(x)
x.head()
x = x.values
print(x)
print(x.shape)

y = data.iloc[:, -1]
print(y)
y.head()
y = y.values
print(y.shape)
y = y.reshape(97, 1)
print(y.shape)


# 定义损失函数
# 损失函数j=sum((x*θ-y)²)/2m
# θ为损失函数，m为样本数，矩阵相乘可以用@
def cost_func(x, y, theta):
    step1 = np.power(x @ theta - y, 2)
    return np.sum(step1) / (2 * len(x))


# 初始theta值。特征值x为(97, 2)矩阵，测试集y为(97, 1)
# x*theta=y为(97, 1)矩阵，所以theta应为（2，1）矩阵
# np.zeros函数返回来一个给定形状和类型的用0填充的数组
theta = np.zeros((2, 1))
theta.shape
cost = cost_func(x, y, theta)
print(cost)


# 梯度下降函数：theta=theta-alpha*(x转置*（x*theta-y）)/m
# alpha为学习速率，转置用.T表示。  iters为迭代次数
def radient_Abscent(x, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        theta = theta - alpha * (x.T @ (x @ theta - y)) / len(x)
        cost = cost_func(x, y, theta)
        costs.append(cost)
        if i % 100 == 0:
            print(cost)

    return theta, costs


alpha = 0.02
iters = 1000
theta, costs = radient_Abscent(x, y, theta, alpha, iters)

# 绘图(损失函数)迭代次数和损失,曲线用plot，散点图scatter
fig, ax = plt.subplots()
ax.plot(np.arange(iters), costs)
ax.set(xlabel='iters',
       ylabel='costs',
       title='iters vs costs')
plt.show()

# 预测直线
# np.linspace是python中创建数值序列工具，生成结构与Numpy数组类似的均匀分布的数值序列
# 如np.linspace(start = 0, stop = 100, num = 5)
# 结果是array([ 0., 25., 50., 75., 100.])
x_predict = np.linspace(y.min(), y.max(), 100)
y_predict = theta[0, 0] + theta[1, 0] * x_predict

fig, ax = plt.subplots()
ax.scatter(x[:, 1], y, label='training')
ax.plot(x_predict, y_predict, 'r', label='predict')
ax.legend()
ax.set(xlabel='population', ylabel='profit')
plt.show()
