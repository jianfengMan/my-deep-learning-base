'''
@Desc  : 
@Date  : 2019/2/16
@Author: zhangjianfeng 
'''

'''
卷积神经网络是对传统全链接网络的优化，这种优化的前提是图形、信号的特征具有局部性。
由于这种局部性，使得全链接网络权值出现冗余，因此需要对部分权值进行删减。
也就是说全链接权值 W 变为稀疏的（大部分为0）,这一过程称为稀疏交互（sparse interactions）

信号不同位置的特征是相似的，因此为进一步压缩权值产生出了权值共享的概念，也就是说不同位置所用的权值是一个权值：
举例：
假设 m=5，那么在计算 y_iyi时只用到了5个权值 W，其他权值为0，权值个数共为 N*5。
此时权值浮点型数字被压缩到了5个，这个数字被称为卷积核心大小（kernel size）
卷积神经网络另一个重要的概念就是感受野（receptive field），举个例子来说， 影响 y_iyi输出所对应的输入 x_ixi
的个数为5，此时感受野大小为5

增大感受野的最简单方式就是增加 m 的数值，也就是增大卷积核心大小
'''

# 引入库函数
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal


# 卷积操作
def filter(indata):
    out = np.zeros_like(indata)
    # 低通滤波器、模糊
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.
    # 高通滤波器
    # kernel = np.array([[0, 1, 0], [1, 9, 1], [0, 1, 0]])
    # 边缘检测1
    # kernel = np.array([[-1, 0, -1], [0, 4, 0], [-1, 0, -1]])
    # 边缘检测2
    # kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # 模糊
    # kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
    for itr in range(len(indata[0, 0, :])):
        out[:, :, itr] = signal.convolve2d(indata[:, :, itr], kernel, boundary='symm', mode='same')
    return out


# 读取图片
data = img.imread("demo.png")
# 图形输出
plt.figure(1)
plt.subplot(121)
plt.title("Orignal")
plt.imshow(data)
plt.subplot(122)
plt.title("Filter")
plt.imshow(filter(data))
plt.show()

'''
机器学习中为了一次输入多个数据则需要增加一维 BATCHSIZE，因此是四维数据。
卷积核心对应的四维数据分别为[KernelSize1，KernelSize2，输入通道数（特征图数量）， 输出通道数（需要自行进行定义）]：
'''

'''
data = img.imread("demo.png")
h, w, c = np.shape(data)
# 对数据进行归一化
data = data / 255
# 初始化卷积核心
kernel_init = np.zeros([3, 3, 3, 3])
kernel_init[:, :, 0, 0] = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.
kernel_init[:, :, 1, 1] = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.
kernel_init[:, :, 2, 2] = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.

kernel = tf.cast(tf.constant(kernel_init), tf.float32)
# 转换为 TensorFlow 数据格式
x = tf.cast(tf.constant(np.reshape(data, [1, h, w, c])), tf.float32)
# 卷积操作
y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
# 运行计算
sess = tf.Session()
filter_data = sess.run(y)
# 滤波后数据
filter_data = np.reshape(filter_data, [h, w, c])
# 绘制图形
plt.figure(1)
plt.subplot(121)
plt.title("Orignal")
plt.imshow(data)
plt.subplot(122)
plt.title("Filter")
plt.imshow(filter_data)
plt.show()
'''

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("data/", one_hot=True)


# 卷积函数
def conv2d_layer(input_tensor, size=1, feature=128, name='conv1d'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        kernel = tf.get_variable('kernel',
                                 (size, size, shape[-1], feature),
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(
                                     stddev=0.1))  # 初始化值很重要，但像文中那种不好的初始化值会导致迭代收敛极为缓慢。
        b = tf.get_variable('b', [feature], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.conv2d(input_tensor, kernel, strides=[1, 2, 2, 1], padding='SAME') + b
    return tf.nn.relu(out)


# 全链接函数
def full_layer(input_tensor, out_dim, name='full'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (shape[1], out_dim), dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.matmul(input_tensor, W) + b
    return out


with tf.variable_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    label = tf.placeholder(tf.float32, [None, 10], name="input_label")
    x2d = tf.reshape(x, [-1, 28, 28, 1])
# 第一层卷积
net = conv2d_layer(x2d, size=4, feature=32, name='conv1')
# 加入池化层，用于提取特征
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 定义卷积层
net = conv2d_layer(net, size=4, feature=32, name='conv2')
net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# flatten 层，用于将三维的图形数据展开成一维数据，用于全链接层
net = tf.contrib.layers.flatten(net)
y = full_layer(net, 10, name='full')

with tf.variable_scope("loss"):
    # 定义 loss 函数
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)
    loss = tf.reduce_mean(ce)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 用于训练参数的保存
saver = tf.train.Saver()
# 载入保存的权值
# saver.restore(sess, tf.train.latest_checkpoint('model'))
for itr in range(501):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                         label: mnist.test.labels}))
        saver.save(sess, os.path.join(os.getcwd(), 'model', 'handwriting'), global_step=itr)
