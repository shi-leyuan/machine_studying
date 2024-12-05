# pca对图像进行降维
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

path = "E:/BaiduNetdiskDownload/data_sets/ex7faces.mat"
data = sio.loadmat(path)
# print(data)
print(data.keys())
x = data['X']
print(x.shape)  # (5000, 1024)


def plot_100_images(x):
    fig, axs = plt.subplots(ncols=10, nrows=10, figsize=(10, 10))
    for c in range(10):
        for r in range(10):
            axs[c, r].imshow(x[10 * c + r].reshape(32, 32).T, cmap='Greys_r')
            axs[c, r].set_xticks([])
            axs[c, r].set_yticks([])


plot_100_images(x)
plt.show()

# 对x去均值化
x_demean = x - np.mean(x, axis=0)
c = x_demean.T @ x_demean / len(x)
U, S, V = np.linalg.svd(c)
U1 = U[:, :36]
print(U1.shape)  # (1024, 36)
x_reduction = x_demean @ U1
print(x_reduction.shape)  # (5000, 36)

x_recover = x_reduction @ U1.T + np.mean(x, axis=0)
plot_100_images(x_recover)
plt.show()
