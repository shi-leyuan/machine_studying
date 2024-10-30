# 逻辑回归解决多分类问题,识别手写数字(从0到9)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

# ex3data1中有5000个训练样例，其中每个训练样例是一个20像素×20像素灰度图像的数字，每个像素由一个浮点数表示，该浮点数表示该位置的灰度强度。
# scipy.io读取.mat文件
path = "E:/BaiduNetdiskDownload/data_sets/ex3data1.mat"
data = sio.loadmat(path)
print(data)
print(type(data))  # 字典
print(data.keys())
# 获取key为X的字典
raw_x = data['X']
# 获取key为y的字典
raw_y = data['y']  # y，“0”的数字标记为“10”，而“1”到“9”的数字按自然顺序标记为“1”到“9”
print(raw_x.shape)  # (5000, 400),本来是二维20×20，拉伸成一维变为400
print(raw_y.shape)  # (5000, 1)


# # 随机打印一张图
# def print_one_picture(x):
#     random_one = np.random.randint(5000)
#     image = x[random_one, :]
#     fig, ax = plt.subplots(figsize=(1, 1))  # 图像大小
#     # imshow() 函数是 Matplotlib 库中的一个函数，用于显示图像。cmap为颜色
#     ax.imshow(image.reshape(20, 20).T, cmap='gray_r')   # .T用来使图像旋转
#     # 删除横轴坐标
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
# print_one_picture(raw_x)

# 打印一百张图
def print_100_picture(x):
    #  从0到4999的范围内随机选择100个唯一的索引。
    random_100 = np.random.choice(5000, 100)
    images = x[random_100, :]
    # plt.subplots 创建一个10x10的子图网格
    # ncols列，nrows行，sharex子图之间共享 X 轴
    fig, ax = plt.subplots(ncols=10, nrows=10, figsize=(6, 6), sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(images[10 * i + j].reshape(20, 20).T, cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


print_100_picture(raw_x)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 损失函数（返回损失值，即误差）
def cost_func(theta, x, y, lamda):  # theta必须放在第一位，lamda为正则化系数
    y_hat = sigmoid(x @ theta)
    # 正则化,防止过拟合
    reg = theta[1:] @ theta[1:] * (lamda / (2 * len(x)))
    return -(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / len(x) + reg


# 梯度向量（表示损失函数对每个参数的导数，表示模型参数更新的方向）
def gradient_reg(theta, x, y, lamda):
    y_hat = sigmoid(x @ theta)
    reg = theta[1:] * (lamda / len(x))
    reg = np.insert(reg, 0, values=0, axis=0)
    return (x.T @ (y_hat - y)) / len(x) + reg


# 在x的第一列加上全为1的列
x = np.insert(raw_x, 0, values=1, axis=1)
print(x.shape)
# 将y变为一维
y = raw_y.flatten()
print(y.shape)

# 优化函数
from scipy.optimize import minimize


# 一对所有（one-vs-all）分类器的训练
def one_vs_all(x, y, lamda, k):  # k为标签个数
    """

    :param x: 特征矩阵，形状为 m×n（m 是样本数，n 是特征数）
    :param y:标签向量，形状为 m。
    :param lamda:正则化系数
    :param k:标签个数
    :return:返回包含所有类别参数的矩阵
    """
    # 获取维度
    n = x.shape[1]
    # theta_all二维素组存放所有参数.创建一个 k×n 的零矩阵，用于存放每个分类器的参数
    theta_all = np.zeros((k, n))
    # 训练每个分类器
    for i in range(1, k + 1):
        # 第i个分类器的参数
        # np.zeros(n,)创建一个一维数组，其长度为 n，数组中的所有元素都是 0。
        theta_i = np.zeros(n, )  # 一维数组
        # 优化
        res = minimize(fun=cost_func,
                       x0=theta_i,
                       args=(x, y == i, lamda),
                       method='TNC',
                       jac=gradient_reg)
        theta_all[i - 1, :] = res.x
    return theta_all


lamda = 1
k = 10
theta_final = one_vs_all(x, y, lamda, k)
print(theta_final)


def predict(x, theta_final):
    """

    :param x: 输入特征矩阵
    :param theta_final:训练好的参数矩阵
    :return:包含每个样本预测类别的数组
    """
    # 计算每个样本属于每个类别的预测概率
    h = sigmoid(x @ theta_final.T)  # (5000,401)@(10,401).T = (5000,10)
    # 获取每个样本的最大概率对应的类别索引
    h_max = np.argmax(h, axis=1)
    return h_max + 1


y_predict = predict(x, theta_final)

final = np.mean(y_predict == y)
print(final)
