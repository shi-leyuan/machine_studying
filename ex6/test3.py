import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.io as sio

#                    寻找最优参数c和gamma

path = "E:/BaiduNetdiskDownload/data_sets/ex6data3.mat"
data = sio.loadmat(path)
# print(data)
# print(data.keys())
x = data['X']
y = data['y']
x_val = data['Xval']  # 验证集
y_val = data['yval']
print(x.shape)  # (211, 2)
print(y.shape)  # (211, 1)
print(x_val.shape)  # (200, 2)
print(y_val.shape)  # (200, 1)


# 数据可视化
def data_plot():
    plt.scatter(x[:, 0], x[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


data_plot()
plt.show()
# c的候选值
c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 50, 100]
# gammma的候选值
gammas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 50, 100]

# 初始化最好得分
best_score = 0
# 初始化最优参数(c和gamma）
best_params = (0, 0)

# 遍历参数
for c in c_values:
    for gamma in gammas:
        svc = SVC(C=c, kernel='rbf', gamma=gamma)
        svc.fit(x, y.flatten())
        score = svc.score(x_val, y_val.flatten())
        if score > best_score:
            best_score = score
            best_params = (c, gamma)

print(best_score)  # 0.965
print(best_params)  # (0.3, 100)

svc2 = SVC(C=0.3, kernel='rbf', gamma=100)
svc2.fit(x, y.flatten())


# 绘图
def plot_boundary(model):
    x_min, x_max = -0.6, 0.3
    y_min, y_max = -0.5, 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    z = model.predict(np.c_[xx.flatten(), yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx, yy, zz)


plot_boundary(svc2)
data_plot()
plt.show()
