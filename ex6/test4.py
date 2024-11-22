import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.io as sio

#                    判断一封邮件是否为垃圾邮件

path = "E:/BaiduNetdiskDownload/data_sets/spamTrain.mat"
data1 = sio.loadmat(path)
# print(data)
print(data1.keys())
# 训练集
x = data1['X']
y = data1['y']
print(x.shape)  # (4000,1899)
print(y.shape)  # (4000,1)
# 测试集
data2 = sio.loadmat("E:/BaiduNetdiskDownload/data_sets/spamTest.mat")
print(data2.keys())
x_test = data2['Xtest']
y_test = data2['ytest']
print(x_test.shape)  # (1000, 1899)
print(y_test.shape)  # (1000, 1)

c_values = [0.03, 0.01, 0.3, 0.1, 1, 3, 10, 30, 100]
# 初始化最好得分
best_score = 0
# 初始化最优参数c
best_params = 0

for c in c_values:
    svc = SVC(C=c, kernel='linear')
    svc.fit(x, y.flatten())
    score = svc.score(x_test, y_test.flatten())
    if score > best_score:
        best_score = score
        best_params = c
print(best_score)  # 0.99
print(best_params)  # 0.03

svc = SVC(C=0.03, kernel='linear')
svc.fit(x, y.flatten())
score_train = svc.score(x, y.flatten())
score_test = svc.score(x_test, y_test.flatten())
print(score_train)  # 0.99425
print(score_test)  # 0.99
