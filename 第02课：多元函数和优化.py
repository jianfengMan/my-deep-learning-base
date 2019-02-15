'''
@Desc  : 
@Date  : 2019/2/15
@Author: zhangjianfeng 
'''

'''
在优化算法中，学习率难以给定，因此很多优化策略的关注点都在于如何改进学习率。
比较有代表性的就是模拟退火算法，该算法可以使学习率随着迭代不断减少。
而另外一个有代表性的算法就是学习率为历史梯度平方根之和的倒数，这个算法称为 AdaGrad
'''
# 矩阵乘法
# 引入库
import tensorflow as tf
import numpy as np

# 定义 placeholder
x = tf.placeholder(dtype=tf.float32, shape=[5, 1])
d = tf.placeholder(dtype=tf.float32, shape=[5, 1])

# 定义 W
w0 = tf.Variable(tf.zeros([1, 1]))
w1 = tf.Variable(tf.zeros([1, 1]))
w2 = tf.Variable(tf.zeros([1, 1]))

# 定义模型
y = w0 + w1 * x + w2 * x ** 2

# 定义损失函数
loss = tf.reduce_mean((y - d) ** 2)

# 定义优化算法
# opt = tf.train.GradientDescentOptimizer(0.1)
#定义 Adam 优化算法
opt = tf.train.AdamOptimizer()

# 计算loss 关于 w 的偏导数
grad = opt.compute_gradients(loss, [w0, w1, w2])

# 执行 w=w+dw
train_step = opt.apply_gradients(grad)

# variable 需要初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for itr in range(100):
    # batch_size = 5
    in_x = np.random.random([5, 1])
    # 假设 x、d 满足如下关系
    in_d = 1 + 0.4 * in_x + 0.3 * in_x ** 2
    sess.run(train_step, feed_dict={x: in_x, d: in_d})
    if itr % 10 == 0:
        print(sess.run([w0.value(), w1.value(), w2.value()]))

# other方法
# 定义数据维度
N = 10
# 定义 placeholder
x = tf.placeholder(dtype=tf.float32, shape=[5, N])
d = tf.placeholder(dtype=tf.float32, shape=[5, N])
# 定义 W
w0 = tf.Variable(tf.zeros([1]))
w1 = tf.Variable(tf.zeros([N, 1]))
# 定义模型
y = tf.matmul(x, w1) + w0
# 定义损失函数
loss = tf.reduce_mean((y - d) ** 2)
# 定义优化算法
opt = tf.train.GradientDescentOptimizer(0.1)
# 计算 w 的增量 dw
grad = opt.compute_gradients(loss)
# 执行 w=w+dw
train_step = opt.apply_gradients(grad)
# variable 需要初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# for itr in range(100):
#     in_x = ...
#     in_y = ...
#     sess.run(train_step,feed_dict={x: in_x, d: in_d})
#     if itr % 10 == 0:
#         print(sess.run([w0.value(), w1.value()]))
