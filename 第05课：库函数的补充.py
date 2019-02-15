'''
@Desc  : 模型的保存和使用
@Date  : 2019/2/15
@Author: zhangjianfeng 
'''

# 引入自带的测试数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 获取数据，如果没有会自动下载
mnist = input_data.read_data_sets("data/", one_hot=True)

# 构建网络模型
# x、label 分别为图形数据和标签数据
x = tf.placeholder(tf.float32, [None, 784])
d = tf.placeholder(tf.float32, [None, 10])

# 构建单层网络中的权值和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 此为所建立的模型
y = tf.matmul(x, W) + b

# 可以将损失函数改为欧氏距离再次实验
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
train_writer = tf.summary.FileWriter("mylogdir", sess.graph)

# 初始化所有变量
sess.run(tf.global_variables_initializer())

# 定义保存类
saver = tf.train.Saver()

# 模型载入
saver.restore(sess, "model/cp5-990")
# 迭代过程
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, d: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                         d: mnist.test.labels}))
        # 迭代过程进行保存
        saver.save(sess, "model/cp5", global_step=itr)

'''
checkpoint 保存了最近几次保存的权值。名称后的“迭代步”代表第几次迭代。
每个迭代步会保存三个文件。默认情况会保存五个迭代步
saver.restore(sess, "model/cp5-990")在迭代开始之前将变量加载进模型之中。

查看tensorboard
tensorboard --logdir mylogdir
'''

'''
# 获取数据，如果没有会自动下载
mnist = input_data.read_data_sets("data/", one_hot=True)
# 构建网络模型
# x、label 分别为图形数据和标签数据
x = tf.placeholder(tf.float32, [None, 784])
d = tf.placeholder(tf.float32, [None, 10])
# 构建单层网络中的权值和偏置
W = tf.Variable(tf.zeros([784, 10]))
# 加入柱状图观测
tf.summary.histogram("Weigh", W)
b = tf.Variable(tf.zeros([10]))
# 此为所建立的模型
y = tf.matmul(x, W) + b
# 可以将损失函数改为欧氏距离再次实验
# loss = tf.reduce_mean(tf.square(y-label))
# 交叉熵计算
prob = tf.nn.softmax(y)
ce = - d * tf.log(prob)
# 使用自带交叉熵函数代替上述代码
# ce=tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)
loss = tf.reduce_mean(ce)
# 加入对于 loss 函数的观测
tf.summary.scalar('loss', loss)
# 用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 用于验证，argmax 为取最大值所在索引，1为所在维度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
# 验证精确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
# 定义会话
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())
# 定义保存类
all_w = tf.trainable_variables()
saver = tf.train.Saver(tf.trainable_variables())

# 定义 tensorboard 查看所需内容
train_writer = tf.summary.FileWriter("logdir", sess.graph)
merged = tf.summary.merge_all()
# 迭代过程
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, d: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                         d: mnist.test.labels}))
        # 迭代过程进行保存
        saver.save(sess, "model/cp5", global_step=itr)
        # 定义summary输出
        summary = sess.run(merged, feed_dict={x: mnist.test.images,
                                              d: mnist.test.labels})
        train_writer.add_summary(summary, itr)

'''
'''
#用于定义加入输出的柱状图
tf.summary.histogram("Weigh", W)
...
#用于获取输出
merged = tf.summary.merge_all()
...
#依然需要使用 session 来执行
summary = sess.run(merged, feed_dict={x: mnist.test.images,
                                        d: mnist.test.labels})
#执行结果用 FileWriter 写入，注意其与 saver 的区别。
train_writer.add_summary(summary, itr)
'''


# 获取数据，如果没有会自动下载
mnist = input_data.read_data_sets("data/", one_hot=True)
# 构建网络模型
# x、label 分别为图形数据和标签数据
with tf.variable_scope("input-layer") as scope:
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    d = tf.placeholder(tf.float32, [None, 10], name='d')
# 构建单层网络中的权值和偏置
with tf.variable_scope("nn-layer") as scope:
    W = tf.Variable(tf.zeros([784, 10]))
    tf.summary.histogram("Weigh", W)
    b = tf.Variable(tf.zeros([10]))
    # 此为所建立的模型
    y = tf.matmul(x, W) + b
# 可以将损失函数改为欧氏距离再次试验
# loss = tf.reduce_mean(tf.square(y-label))
# 交叉熵计算
with tf.variable_scope("loss-layer") as scope:
    prob = tf.nn.softmax(y)
    ce = - d * tf.log(prob)
    # 使用自带交叉熵函数代替上述代码
    # ce=tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=y)
    loss = tf.reduce_mean(ce)
# 加入对于 loss 函数的观测
tf.summary.scalar('loss', loss)
# 用梯度迭代算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 用于验证，argmax 为取最大值所在索引，1为所在维度
with tf.variable_scope("accuracy") as scope:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
    # 验证精确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
# 定义会话
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())
# 定义保存类
all_w = tf.trainable_variables()
saver = tf.train.Saver(tf.trainable_variables())

# 定义 tensorboard 查看所需内容
train_writer = tf.summary.FileWriter("logdir", sess.graph)
merged = tf.summary.merge_all()
# 迭代过程
for itr in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, d: batch_ys})
    if itr % 10 == 0:
        print("step:%6d  accuracy:" % itr, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                         d: mnist.test.labels}))
        # 迭代过程进行保存
        saver.save(sess, "model/cp5", global_step=itr)
        # 定义 summary 输出
        summary = sess.run(merged, feed_dict={x: mnist.test.images,
                                              d: mnist.test.labels})
        train_writer.add_summary(summary, itr)
