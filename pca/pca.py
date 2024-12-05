# pca降维
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

path = "E:/BaiduNetdiskDownload/data_sets/ex7data1.mat"
data = sio.loadmat(path)
# print(data)
print(data.keys())
x = data['X']
print(x.shape)  # (50, 2)

# 绘制数据集
plt.scatter(x[:, 0], x[:, 1])
plt.show()

# 对x去均值化
x_demean = x - np.mean(x, axis=0)
plt.scatter(x_demean[:, 0], x_demean[:, 1])
plt.show()

# 计算协方差矩阵
c = x_demean.T @ x_demean / len(x)

# 计算特征值，特征向量.特征值分解（SVD）
# 使用 SVD（奇异值分解）对协方差矩阵 c 进行分解。SVD 会返回三个矩阵：
# U：包含左奇异向量（即主成分的方向）。
# S：奇异值，表示数据在主成分方向上的方差。
# V：包含右奇异向量，表示特征空间的投影方向。
U, S, V = np.linalg.svd(c)
U1 = U[:, 0]  # U[:, 0] 取出第一个主成分的方向（即最大方差的方向）。
print(U1.shape)  # (2,)
# 降维
plt.figure(figsize=(7, 7))
x_reduction = x_demean @ U1
print(x_reduction.shape)  # (50,)
plt.scatter(x_demean[:, 0], x_demean[:, 1])
plt.plot([0, U1[0]], [0, U1[1]], c='r')  # 绘制第一个主成分方向（红色线）
plt.plot([0, U[:, 1][0]], [0, U[:, 1][1]], c='k')  # 绘制第二个主成分方向（黑色线）
plt.show()
