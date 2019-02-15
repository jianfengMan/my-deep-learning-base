'''
@Desc  : 
@Date  : 2019/2/15
@Author: zhangjianfeng 
'''

'''
从曲线（面）拟合的角度来看待机器学习是合适的。多个直线相加依然是直线，
为了使直线出现更复杂的情况，我们需要对曲线进行弯折，这就是激活函数的作用。

深度学习训练过程最关键的部分就在于梯度的计算。
而梯度的计算中必须要做的就是链式求导。由此就产生了很多概念，
比如反向传播算法（Back Propagation，BP），
比如时间反向求导（Back Propagation Through Time，BPTT）
激活函数 ReLU 是没有二阶导数的
'''

import tensorflow as tf

# x、label分别为图形数据和标签数据
x = tf.placeholder(tf.float32, [None, 20])
label = tf.placeholder(tf.float32, [None, 10])

# 构建第一层网络中的权值和偏置
W1 = tf.Variable(tf.zeros([20, 10]))
b1 = tf.Variable(tf.zeros([10]))
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# 构建第二层网络中的权值和偏置
W2 = tf.Variable(tf.zeros([10, 10]))
b2 = tf.Variable(tf.zeros([10]))
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

# 构建第三层网络中的权值和偏置
W3 = tf.Variable(tf.zeros([10, 10]))
b3 = tf.Variable(tf.zeros([10]))
y = tf.matmul(y2, W3) + b3

# 交叉熵计算
prob = tf.nn.softmax(y)
ce = - label * tf.log(prob)

# 使用自带交叉熵函数
# ce=tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)
loss = tf.reduce_mean(ce)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.style.use('ggplot')

N = 20
# 定义坐标点
x = np.zeros([11 * N * 2, 2])
x[:11 * N, 0] = np.array([itr / N for itr in range(N)] * 11)
x[:11 * N, 1] = np.array(sum([[itr * 0.1] * N for itr in range(11)], []))
x[11 * N:, 0] = x[:11 * N, 1]
x[11 * N:, 1] = x[:11 * N, 0]
# 进行线性变换
y = np.dot(x, np.array([[1, 0.5], [-0.5, 0.7]]))
# 绘制图形
plt.scatter(x[:, 0], x[:, 1], c='orange', marker='o', alpha=0.6)
plt.scatter(y[:, 0], y[:, 1], c='blue', marker='+', alpha=0.6)
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

N = 20
#定义坐标点
x = np.zeros([11*N*2, 2])
x[:11*N, 0] = np.array([itr/N for itr in range(N)]*11)
x[:11*N, 1] = np.array(sum([[itr*0.1]*N for itr in range(11)], []))
x[11*N:, 0] = x[:11*N, 1]
x[11*N:, 1] = x[:11*N, 0]
#进行线性变换
y = sigmoid(np.dot(x, np.array([[1, 0.5], [-0.5, 0.7]])))
plt.scatter(x[:, 0], x[:, 1], c='orange', marker='o', alpha=0.6)
plt.scatter(y[:, 0], y[:, 1], c='blue', marker='+', alpha=0.6)
plt.show()