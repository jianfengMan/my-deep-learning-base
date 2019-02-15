'''
@Desc  : 
@Date  : 2019/2/15
@Author: zhangjianfeng 
'''
'''
方差（variance）概念的产生是为了描述变量的离散程度
协方差（covariance）用来描述两个随机向量的线性相关性
PCA 变换的目标之一就是使各个列之间的线性相关性最小，也就是说使协方差矩阵可以对角化
信息熵，也称香农熵（Shannon entropy）∑p(xi)log(1/p(xi))
    其中-log(p(x))−log(p(x)) 称为自信息，符号表示为 II，log 以2为底时单位为 bit。
交叉熵（cross entropy）：用来衡量两个分布相似度 H(p,q)=∑p(xi)log(1/q(xi))
softmax:它实际上解决的问题是将函数表示为概率表示的问题
'''

'''
是在实际工作中，向量的维度可能会达到上千维。
这时如果再使用向量距离作为损失函数的话，计算可能会出现问题，有用输出会淹没于噪声之中。
因此需要使用的损失函数为交叉熵,获取输出之后需要进行 softmax 用于计算交叉熵
'''
# 引入自带的测试数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 获取数据如果没有会自动下载
mnist = input_data.read_data_sets("data/", one_hot=True)

# 构建网络模型
# x，label 分别为图形数据和标签数据
x = tf.placeholder(tf.float32, [None, 784])
d = tf.placeholder(tf.float32, [None, 10])

# 构建单层网络中的权值和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 此为所建立的模型
y = tf.matmul(x, W) + b

# 可以将损失函数改为欧氏距离再次试验
# loss = tf.reduce_mean(tf.square(y-label))
# 交叉熵计算
prob = tf.nn.softmax(y)
ce = - d * tf.log(prob)

# 使用自带交叉熵函数代替上述代码
# ce=tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)

loss = tf.reduce_mean(ce)

# 用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 用于验证，argmax 为取最大值所在索引，1为所在维度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
# 验证精确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 定义会话
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())
# 迭代过程
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, d: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:" % itr,
              sess.run(accuracy, feed_dict={x: mnist.test.images, d: mnist.test.labels}))
