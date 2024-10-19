import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 线性可分
# 要求：根据学生两门课的成绩，判断能否会录取
path = "E:/BaiduNetdiskDownload/data_sets/ex2data1.txt"
data = pd.read_csv(path, names=['grade1', 'grade2', 'if_accept'])
print(data)

# 绘制是否录取的散点图，并将成绩分别显示
# maker为标记
fig, ax = plt.subplots()  # 显示图像
ax.scatter(data[data['if_accept'] == 0]['grade1'], data[data['if_accept'] == 0]['grade2'], c='r', marker='x',
           label='fail')  # maker支持'x','o','^'等，x轴为grade1，y轴为grade2
ax.scatter(data[data['if_accept'] == 1]['grade1'], data[data['if_accept'] == 1]['grade2'], c='b', marker='o',
           label='win')
ax.legend()  # 显示标签
ax.set(xlabel='grade1',
       ylabel='grade2')
plt.show()

# 给数据添加一列全为1的列
data.insert(loc=0, column='ones', value=1)
print(data)

# 构造数据集
x = data.iloc[:, 0:-1]
x = x.values
print(x.shape)
y = data.iloc[:, -1]
y = y.values
print(y.shape)
y = y.reshape(100, 1)
print(y.shape)


# 定义sigmoid函数（输出值只在0-1）
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# 损失函数
def cost_func(x, y, theta):
    y_hat = sigmoid(x @ theta)
    # 使用sum是因为y*np.log(y_hat)+(1-y)*np.log(1-y_hat)为向量，进行累计
    return -(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / len(x)


theta = np.zeros((3, 1))
print(theta.shape)
# 看一下初始的损失函数
cost_init = cost_func(x, y, theta)
print(cost_init)


# 梯度下降函数
def gradientDescent(x, y, theta, alpha, iters):
    costs = []
    for i in range(iters):
        y_hat = sigmoid(x @ theta)
        theta = theta - (alpha / len(x)) * x.T @ (y_hat - y)
        cost = cost_func(x, y, theta)
        costs.append(cost)
        if i % 1000 == 0:
            print(cost)
    return costs, theta


alpha = 0.004
iters = 200000
costs, theta_final = gradientDescent(x, y, theta, alpha, iters)
print(theta_final)


# 预测准确率
def predict_accurate(x, theta):
    #  prob表示一个包含概率的列表
    prob = sigmoid(x @ theta)
    list = []
    for x in prob:
        if x > 0.5:
            list.append(1)
        else:
            list.append(0)
    return list


print(predict_accurate(x, theta_final))
# 将预测结果转变为数组
predict_arr = np.array(predict_accurate(x, theta_final))
print(predict_arr)
# 将预测结果转换为一列
predict_final = predict_arr.reshape((len(predict_arr)), 1)
# 进行准确率预测
result = np.mean(predict_final == y)  # mean()函数的功能是求取平均值
print(result)

# 绘制线性可分决策边界
# θ(0) + θ(1)X1 + θ(2)X2 = 0
theta1 = - theta_final[0, 0] / theta_final[2, 0]
theta2 = - theta_final[1, 0] / theta_final[2, 0]
x = np.linspace(20, 100, 100)  # 表示从20到100生成100个数
f = theta1 + theta2 * x
fig, ax = plt.subplots()
ax.scatter(data[data['if_accept'] == 0]['grade1'], data[data['if_accept'] == 0]['grade2'], c='r', marker='x',
           label='fail')  # maker支持'x','o','^'等，x轴为grade1，y轴为grade2
ax.scatter(data[data['if_accept'] == 1]['grade1'], data[data['if_accept'] == 1]['grade2'], c='b', marker='o',
           label='win')  # c=''表示颜色
ax.legend()  # 显示标签
ax.set(xlabel='grade1',
       ylabel='grade2')
ax.plot(x, f, c='g')
plt.show()
