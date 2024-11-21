import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.io as sio

#                    线性可分svm

path = "E:/BaiduNetdiskDownload/data_sets/ex6data1.mat"
data = sio.loadmat(path)
# print(data)
print(data.keys())
x = data['X']  # 二维特征矩阵，包含 51 个样本，每个样本有 2 个特征。
y = data['y']  # 目标标签，51 个样本的类别标签（1 或 0）
print(x.shape)  # (51, 2)
print(y.shape)  # (51, 1)


# 数据的可视化
def data_plot():
    # c=y.flatten()：根据标签颜色绘制点，1 和 0 使用不同颜色。
    # cmap='jet'：使用颜色映射。
    plt.scatter(x[:, 0], x[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


data_plot()
plt.show()
# C为正则化参数，kernel为使用的核函数
svc1 = SVC(C=1, kernel='linear')
# fit(x, y.flatten())：训练模型
svc1.fit(x, y.flatten())

print(svc1.predict(x))
print(svc1.score(x, y.flatten()))


# 绘制决策边界
def plot_boundary(model):
    x_min, x_max = -0.5, 4.5
    y_min, y_max = 1.3, 5
    # 创建网格
    # np.meshgrid()创建一个二维网格，用于生成决策边界的背景数据。
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # 预测边界
    # ravel函数的作用是让多维数组变成一维数组,假设xx的形状是 (500, 500)，展平后的形状变为 (250000,)。
    # np.c_ 将 xx.ravel() 和 yy.ravel()（一维化的网格点）按列拼接成二维数组，每行对应一个网格点。拼接后形成一个形状为 (250000, 2) 的二维数组
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # 拼接 xx 和 yy 的网格点
    z = z.reshape(xx.shape)  # 还原为网格形状以便绘图

    # 绘制等高线
    # levels=[0.5] 指定决策边界为分类概率为 0.5 的线
    plt.contour(xx, yy, z, levels=[0.5], colors='red')


plot_boundary(svc1)
data_plot()
plt.show()


# 当正则化参数c为100时
svc100 = SVC(C=100, kernel='linear')
svc100.fit(x, y.flatten())
print(svc100.predict(x))
print(svc100.score(x, y.flatten()))
plot_boundary(svc100)
data_plot()
plt.show()