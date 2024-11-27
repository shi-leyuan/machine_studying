import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

#                   Kmeans聚类(无监督学习)
path = "E:/BaiduNetdiskDownload/data_sets/ex7data2.mat"
data = sio.loadmat(path)
# print(data)
print(data.keys())
x = data['X']
print(x.shape)  # (300, 2)

plt.scatter(x[:, 0], x[:, 1])
plt.show()


# 获取每个样本所属的类别
def find_class(x, centros):
    """

    :param x: 数据点，形状为 (m, n)，其中 m 是数据点的个数，n 是每个数据点的特征数。
    :param centros:聚类中心
    :return:每个数据点所属的类别的索引
    """
    idx = []
    for i in range(len(x)):
        # x[i]为一维数组，centros为二维，但是最终结果会自动变维度
        # np.linalg.norm: 计算欧氏距离，用于度量每个数据点与每个聚类中心的距离。
        distance = np.linalg.norm((x[i] - centros), axis=1)
        id_i = np.argmin(distance)  # 返回距离最小值的索引，即找到最近的聚类中心。
        idx.append(id_i)

    return np.array(idx)


# 定义三个初始的聚类中心，即坐标
centros = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_class(x, centros)


# print(idx)


# 计算聚类中心点
def compute_centros(x, idx, k):
    """

    :param x:形状为 (m, n) 的数据集，m 是数据点的数量，n 是每个数据点的特征数。
    :param idx:是一个数组，包含每个数据点所属的聚类中心的索引。
    :param k:聚类的数量
    :return:将centros列表转换为 NumPy 数组并返回
    """
    centros = []
    for i in range(k):  # 遍历每个聚类
        # 筛选属于当前类别 i 的数据点，x[idx == i] 会根据布尔数组筛选出属于聚类 i 的所有数据点
        points_in_cluster = x[idx == i]
        if len(points_in_cluster) == 0:
            # 如果当前类别没有数据点，保持原聚类中心位置
            centros.append(centros[-1] if centros else np.zeros(x.shape[1]))
        else:
            # 正常计算均值
            centros_i = np.mean(points_in_cluster, axis=0)
            centros.append(centros_i)
    return np.array(centros)


compute_centros(x, idx, k=3)


# print(compute_centros(x, idx, k=3))

# 运行
def run_kmeans(x, centros, iters):
    k = len(centros)  # 聚类的数量
    centros_all = []  # 存储每次迭代的聚类中心
    centros_all.append(centros)
    centros_i = centros  # 当前的聚类中心初始化为初始值
    for i in range(iters):
        idx = find_class(x, centros_i)
        centros_i = compute_centros(x, idx, k)
        centros_all.append(centros_i)
    return idx, np.array(centros_all)


# 可视化
def plot_data(x, centros_all, idx):
    plt.figure(figsize=(8, 6))
    # c=idx 表示使用聚类标签 idx 来为每个数据点着色
    # cmap='rainbow' 是指定颜色映射
    plt.scatter(x[:, 0], x[:, 1], c=idx, cmap='rainbow')
    for i in range(centros_all.shape[1]):
        # centros_all[:, i, 0] 和 centros_all[:, i, 1] 分别表示每次迭代的聚类中心在 x 和 y 轴的坐标
        # 绘制每次迭代的聚类中心，'kx--' 表示黑色的叉号，虚线连接
        plt.plot(centros_all[:, i, 0], centros_all[:, i, 1], 'kx--', lw=2, markersize=8)
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)


idx, centros_all = run_kmeans(x, centros, iters=10)
plot_data(x, centros_all, idx)
plt.show()
