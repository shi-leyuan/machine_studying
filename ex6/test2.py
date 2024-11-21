import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.io as sio

#                     线性不可分svm

path = "E:/BaiduNetdiskDownload/data_sets/ex6data2.mat"
data = sio.loadmat(path)
# print(data)
print(data.keys())
x = data['X']
y = data['y']
print(x.shape)  # (863, 2)
print(y.shape)  # (863, 1)


# 数据可视化
def data_plot():
    plt.scatter(x[:, 0], x[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


data_plot()
plt.show()

svc1 = SVC(C=1, kernel='rbf', gamma=1)
svc1.fit(x, y.flatten())

print(svc1.score(x, y.flatten()))


def plot_boundary(model):
    x_min, x_max = 0, 1.2
    y_min, y_max = 0.3, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, levels=[0.5], colors='red')


plot_boundary(svc1)
data_plot()
plt.show()

svc2 = SVC(C=1, kernel='rbf', gamma=100)
svc2.fit(x, y.flatten())
plot_boundary(svc2)
data_plot()
plt.show()
