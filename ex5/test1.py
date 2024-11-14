import numpy as np
import numpy as py
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
                                          # 方差和偏差
path = "E:/BaiduNetdiskDownload/data_sets/ex5data1.mat"
data = sio.loadmat(path)
# print(data)
print(data.keys())

# 训练集
x_train = data['X']
y_train = data['y']
# print(x_train.shape)     # (12, 1)
# print(y_train.shape)     # (12, 1)

# 验证集
x_val = data['Xval']
y_val = data['yval']
# print(x_val.shape)       # (21, 1)
# print(y_val.shape)       # (21, 1)

# 测试集
x_test = data['Xtest']
y_test = data['ytest']
# print(x_test.shape)     # (12, 1)
# print(y_test.shape)     # (12, 1)

x_train = np.insert(x_train, 0, 1, 1)
x_val = np.insert(x_val, 0, 1, 1)
x_test = np.insert(x_test, 0, 1, 1)


def data_plot():
    fig, ax = plt.subplots()
    ax.scatter(x_train[:, 1], y_train)
    ax.set(xlabel='change in water level(x)',
           ylabel='water flowing out og the dam(y)')


# 损失函数
def reg_cost(theta, x, y, lamda):
    cost = np.sum(np.power((x @ theta - y.flatten()), 2))
    reg = theta[1:] @ theta[1:] * lamda
    return (cost + reg) / (2 * len(x))


# 梯度下降
def reg_gradient(theta, x, y, lamda):
    grad = (x @ theta - y.flatten()) @ x
    reg = lamda * theta
    reg[0] = 0  # 给第一项赋0，相当于剔除第一项
    return (reg + grad) / len(x)


from scipy.optimize import minimize


# 训练
def train_model(x, y, lamda):
    # np.ones()函数返回给定形状和数据类型的新数组
    theta = np.ones(x.shape[1])  # 初始化theta为全1的向量
    res = minimize(fun=reg_cost,
                   x0=theta,
                   args=(x, y, lamda),
                   method='TNC',
                   jac=reg_gradient)
    return res.x  # 返回优化后的 theta


theta_final = train_model(x_train, y_train, lamda=0)
data_plot()
plt.plot(x_train[:, 1], x_train @ theta_final, c='r')
plt.show()


# 使训练样本数从1开始增加，比较训练集和测试集上损失函数的变化,判断是否过拟合或者欠拟合
def plot_learning(x_train, y_train, x_val, y_val, lamda):
    x = range(1, len(x_train) + 1)
    training_cost = []  # 训练集损失函数（空列表）
    cv_cost = []  # 测试集损失函数（空列表）
    for i in x:
        res = train_model(x_train[:i, :], y_train[:i, :], lamda)  # theta
        training_cost_i = reg_cost(res, x_train[:i, :], y_train[:i, :], lamda)
        cv_cost_i = reg_cost(res, x_val, y_val, lamda)
        training_cost.append(training_cost_i)
        cv_cost.append(cv_cost_i)

    plt.plot(x, training_cost, label='training cost')
    plt.plot(x, cv_cost, label='cv cost')
    plt.legend()
    plt.xlabel('number of training examples')
    plt.ylabel('error')
    plt.show()


plot_learning(x_train, y_train, x_val, y_val, lamda=0)


# 构造多项式特征，解决高偏差
def poly_feature(x, power):
    """

    :param x: 输入特征矩阵，通常是一个二维数组
    :param power:表示多项式的最高次方
    :return:返回修改后的特征矩阵 x
    """
    for i in range(2, power + 1):
        # x.shape[1] 是 x 的列数，表示要在最后一列插入
        # axis=1 表示沿着列的方向插入新特征
        x = np.insert(x, x.shape[1], np.power(x[:, 1], i), axis=1)
    return x


# 获取均值和标准差,使得各个特征的尺度一致,加速梯度下降收敛,避免某些特征主导模型
def get_means_stds(x):
    # axis=0 表示沿着行的方向计算均值
    means = np.mean(x, axis=0)  # 计算每一列的均值
    stds = np.std(x, axis=0)  # 计算每一列的标准差
    return means, stds


# 特征归一化,使得每个特征的均值为 0，标准差为 1
def feature_normalize(x, means, stds):
    # x[:, 1:]选择 x 中从第二列开始
    # (x[:, 1:] - means[1:])：对每个特征进行中心化，减去每个特征的均值 means[1:]
    # means[1:] 表示除偏置项外的所有特征的均值
    # /stds[1:]：将中心化后的每个特征除以该特征的标准差 stds[1:]，实现标准化
    x[:, 1:] = (x[:, 1:] - means[1:]) / stds[1:]
    return x  # 返回标准化后的矩阵


power = 6
x_train_poly = poly_feature(x_train, power)
x_val_poly = poly_feature(x_val, power)
x_test_poly = poly_feature(x_test, power)

train_means, train_stds = get_means_stds(x_train_poly)
# 对训练集.验证集和测试集进行标准化处理
x_train_norm = feature_normalize(x_train_poly, train_means, train_stds)
x_val_norm = feature_normalize(x_val_poly, train_means, train_stds)
x_test_norm = feature_normalize(x_test_poly, train_means, train_stds)

theta_fit = train_model(x_train_norm, y_train, lamda=0)


# 绘制多项式回归拟合曲线
def plot_poly_fit():
    # 绘制原始数据点
    data_plot()

    # 生成新的 x 数据（用于拟合曲线的绘制）
    x = np.linspace(-60, 60, 100)  # 生成从 -60 到 60 之间的 100 个数据点
    xx = x.reshape(100, 1)  # 转换为列向量
    # 插入偏置项（全为 1 的列）
    xx = np.insert(xx, 0, 1, axis=1)  # 假设第一列是偏置项
    # 生成多项式特征（例如，x^1, x^2, ..., x^6）
    xx = poly_feature(xx, power=6)
    # 标准化特征
    xx = feature_normalize(xx, train_means, train_stds)
    # 绘制多项式回归拟合曲线，并加上标签
    plt.plot(x, xx @ theta_fit, 'r')  # 给拟合曲线加上标签
    plt.show()


plot_poly_fit()
# 在验证集和训练集的情况
plot_learning(x_train_norm, y_train, x_val_norm, y_val, lamda=0)
plt.show()

# 寻找合适的lamda
lamdas = [0, 0.1, 0.01, 0.001, 0.3, 0.03, 0.003, 1, 3, 10]
training_cost = []
cv_cost = []
for lamda in lamdas:
    res = train_model(x_train_norm, y_train, lamda)
    t_c = reg_cost(res, x_train_norm, y_train, lamda=0)
    cv_c = reg_cost(res, x_val_norm, y_val, lamda=0)
    training_cost.append(t_c)
    cv_cost.append(cv_c)

plt.plot(lamdas, training_cost, label="training cost")
plt.plot(lamdas, cv_cost, label="cv cost")
plt.legend()
plt.show()
print(lamdas[np.argmin(cv_cost)])  # 3

# 当lamda等于3时，在测试集的效果
res = train_model(x_train_norm, y_train, lamda=3)
test_cost = reg_cost(res, x_test_norm, y_test, lamda=0)
print(test_cost)
