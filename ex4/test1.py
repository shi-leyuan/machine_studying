import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

# 三层神经网络，对手写数字图像进行识别和分类
path = "E:/BaiduNetdiskDownload/data_sets/ex4data1.mat"
data = sio.loadmat(path)
# print(data)
raw_x = data['X']
raw_y = data['y']
print(raw_x.shape)  # (5000, 400)
print(raw_y.shape)  # (5000, 1)

x = np.insert(raw_x, 0, values=1, axis=1)
print(x.shape)  # (5000, 401)


# 对y进行one-hot编码
# one_hot_code函数将每个数字标签转换为One-Hot编码格式。比如，标签3被编码为 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
def one_hot_code(raw_y):
    """

    :param raw_y:向量标签
    :return:返回二维数组
    """
    # 初始化空列表
    result = []
    for i in raw_y:   # i的范围是1-10
        # 创建一个大小为 10 的零向量
        y_temp = np.zeros(10)
        y_temp[i - 1] = 1
        result.append(y_temp)

    return np.array(result)


y = one_hot_code(raw_y)
# print(y)
# print(y.shape)  # (5000, 10)

theta = sio.loadmat('E:/BaiduNetdiskDownload/data_sets/ex3weights.mat')
theta1 = theta['Theta1']
# print(theta1.shape)        # (25, 401)
theta2 = theta['Theta2']
# print(theta2.shape)       # (10, 26)

# 序列化权重参数
# serialize 函数的作用是将两个矩阵（即权重矩阵 a 和 b）展平并连接为一个一维数组，可以将多个矩阵参数压缩到一个向量中，便于传递给优化器
def serialize(a, b):
    """

    :param a: 矩阵a
    :param b:矩阵b
    :return:返回一维数组
    """
    return np.append(a.flatten(), b.flatten())


theta_serialize = serialize(theta1, theta2)


# print(theta_serialize.shape)     # (10285,)

# 解序列化权重参数
# deserialize 函数的作用是将一个一维数组 theta_serialize 拆分并还原为两个权重矩阵 theta1 和 theta2
def deserialize(theta_serialize):
    # theta_serialize[:25 * 401] 从 theta_serialize 数组中取出前 25 * 401 个元素。重塑为一个形状为 (25, 401) 的二维数组
    theta1 = theta_serialize[:25 * 401].reshape(25, 401)
    # 取出剩余元素，重塑为形状为 (10, 26) 的二维数组 theta2。
    theta2 = theta_serialize[25 * 401:].reshape(10, 26)
    return theta1, theta2


# 激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播
# 计算每一层的激活值和最终输出
def feed_spread(theta_serialize, x):
    """

    :param theta_serialize: 解序列化权重参数
    :param x:输入矩阵
    :return:a1 是输入层的输入值。
            z2 和 a2 是隐藏层的线性组合和激活值。
            z3 和 h 是输出层的线性组合和激活值。
    """
    # 将一维数组 theta_serialize 转换回原始的两个权重矩阵 theta1 和 theta2，对应神经网络的两层。
    theta1, theta2 = deserialize(theta_serialize)
    a1 = x
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


# 不带正则化的损失函数
def cost1(theta_serialize, x, y):
    a1, z2, a2, z3, h = feed_spread(theta_serialize, x)
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / len(x)


cost1(theta_serialize, x, y)


# print(cost1(theta_serialize, x, y))

# 带正则化的损失函数
def reg_cost(theta_serialize, x, y, lamda):
    sum1 = np.sum(np.power(theta1[:, 1:], 2))  # 计算 theta1 中不包含偏置项的权重的平方和
    sum2 = np.sum(np.power(theta2[:, 1:], 2))  # 计算 theta2 中不包含偏置项的权重的平方和
    reg = (sum1 + sum2) * lamda / (2 * len(x))
    return reg + cost1(theta_serialize, x, y)


lamda = 1


# print( reg_cost(theta_serialize, x, y, lamda))

# 不带正则化的梯度
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def gradient(theta_serialize, x, y):
    """

    :param theta_serialize:
    :param x:输入数据
    :param y:输出数据
    :return:返回梯度
    """
    theta1, theta2 = deserialize(theta_serialize)
    a1, z2, a2, z3, h = feed_spread(theta_serialize, x)
    d3 = h - y  # 计算输出层的误差
    # theta2[:, 1:]代表隐藏层权重矩阵去掉偏置列。
    d2 = d3 @ theta2[:, 1:] * sigmoid_gradient(z2)  # 计算隐藏层的误差
    D2 = (d3.T @ a2) / len(x)  # 计算对 Theta2 的梯度，表示输出层权重的变化率
    D1 = (d2.T @ a1) / len(x)  # 计算对 Theta1 的梯度，表示隐层权重的变化率
    return serialize(D1, D2)  # 返回梯度序列化结果


# 带正则化的梯度
# lamda为正则化参数，控制正则化项的强度
def reg_gradient(theta_serialize, x, y, lamda):
    D = gradient(theta_serialize, x, y)
    D1, D2 = deserialize(D)
    theta1, theta2 = deserialize(theta_serialize)
    # 添加正则化项到梯度
    D1[:, 1:] = D1[:, 1:] + theta1[:, 1:] * lamda / len(x)
    D2[:, 1:] = D2[:, 1:] + theta2[:, 1:] * lamda / len(x)
    return serialize(D1, D2)


from scipy.optimize import minimize


def training(x, y):
    init_theta = np.random.uniform(-0.5, 0.5, 10285)
    res = minimize(fun=reg_cost,
                   x0=init_theta,
                   args=(x, y, lamda),
                   method='TNC',
                   jac=reg_gradient,
                   options={'maxfun': 300})   # 设置最大迭代次数300
    return res


lamda = 10
res = training(x, y)
raw_y = data['y'].reshape(5000, )
a1, z2, a2, z3, h = feed_spread(res.x, x)
y_pred = np.argmax(h, axis=1) + 1
accuracy = np.mean(y_pred == raw_y)
print(accuracy)
