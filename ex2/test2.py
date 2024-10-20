import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 线性不可分

path = "E:/BaiduNetdiskDownload/data_sets/ex2data2.txt"
data = pd.read_csv(path, names=['test1', 'test2', 'if_accept'])
print(data)

fig, ax = plt.subplots()
ax.scatter(data[data['if_accept'] == 0]['test1'], data[data['if_accept'] == 0]['test2'], c='r', marker='x',
           label='fail')
ax.scatter(data[data['if_accept'] == 1]['test1'], data[data['if_accept'] == 1]['test2'], c='b', marker='o',
           label='win')
ax.legend()
ax.set(xlabel='test1',
       ylabel='test2')
plt.show()


# 特征映射
# power代表阶次，二阶
def feature_mapping(x1, x2, power):
    data = {}
    # 外循环从0——阶次
    for i in np.arange(power + 1):
        for j in np.arange(i + 1):
            # '{}{}'.format(i-j, j)将i-j和j的值插入到{}的位置
            # 如果i=3，j=1，则'{}{}'.format(i-j, j)就为字符串'21'
            # 将两个幂运算结果的乘积存入字典data中，键名为 '{}{}'.format(i-j, j)
            data['{}{}'.format(i - j, j)] = np.power(x1, i - j) * np.power(x2, j)
    # 将字典转为DataFrame数据格式
    return pd.DataFrame(data)


x1 = data['test1']
x2 = data['test2']
data2 = feature_mapping(x1, x2, 6)
# print(data2)

x = data2.values
print(x.shape)
y = data.iloc[:, -1]
y = y.values
y = y.reshape(118, 1)
print(y.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 损失函数
def cost_func(x, y, theta, learning_rate):
    y_hat = sigmoid(x @ theta)
    # 正则化，防止过拟合
    reg = np.sum(np.power(theta[1:], 2)) * (learning_rate / (2 * len(x)))  # learning_rate 为学习速率
    # 使用sum是因为y*np.log(y_hat)+(1-y)*np.log(1-y_hat)为向量，进行累计
    return -(np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / len(x) + reg


theta = np.zeros((28, 1))
print(theta.shape)
learning_rate = 1
cost_init = cost_func(x, y, theta, learning_rate)
print(cost_init)


# 梯度下降函数
def gradientDescent(x, y, theta, alpha, iters, learning_rate):
    costs = []
    for i in range(iters):
        # theta从1开始的，为了维度匹配，加上一行
        reg = theta[1:] * (learning_rate / len(x))
        reg = np.insert(reg, 0, values=0, axis=0)
        y_hat = sigmoid(x @ theta)
        theta = theta - (x.T @ (y_hat - y)) * (alpha / len(x)) - reg
        cost = cost_func(x, y, theta, learning_rate)
        costs.append(cost)
        # if i % 1000 == 0:
        #     print(cost)
    return costs, theta


alpha = 0.001
iters = 200000
learning_rate = 0.001
costs, theta = gradientDescent(x, y, theta, alpha, iters, learning_rate)


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


print(predict_accurate(x, theta))
# 将预测结果转变为数组
predict_arr = np.array(predict_accurate(x, theta))
print(predict_arr)
# 将预测结果转换为一列
predict_final = predict_arr.reshape((len(predict_arr)), 1)
# 进行准确率预测
result = np.mean(predict_final == y)  # mean()函数的功能是求取平均值
print(result)

# 可视化
# 创建从 -1.2 到 1.2 之间的 200 个等间距点
x = np.linspace(-1.2, 1.2, 200)
# np.meshgrid()生成网格点坐标矩阵
xx, yy = np.meshgrid(x, x)  # 200×200的矩阵
# ravel()将数组维度拉成一维数组,以数组形式返回
z = feature_mapping(xx.ravel(), yy.ravel(), 6).values
# 计算网格点上的预测值
zz = z @ theta
# 将预测值重塑为 (200, 200) 的二维数组
zz = zz.reshape(xx.shape)

fig, ax = plt.subplots()
ax.scatter(data[data['if_accept'] == 0]['test1'], data[data['if_accept'] == 0]['test2'], c='r', marker='x',
           label='fail')
ax.scatter(data[data['if_accept'] == 1]['test1'], data[data['if_accept'] == 1]['test2'], c='b', marker='o',
           label='win')
ax.legend()
ax.set(xlabel='test1',
       ylabel='test2')
# plt.contour绘制等高线
plt.contour(xx, yy, zz, levels=[0])  # levels=[0] 表示绘制预测值为 0 的等高线
plt.show()
