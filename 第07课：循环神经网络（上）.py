'''
@Desc  : 
@Date  : 2019/2/16
@Author: zhangjianfeng 
'''

'''
为防止数值无限增长，激活函数可以选择 tanh

LSTM 中有三个门：g1 用于控制上一时刻记忆到下一时刻的多少；
g2 用于控制新输入 r_t进入记忆量 C 的多少;
g3 用于控制输出
全链接和时序问题的区别，上一个输出是否影响下一个？？

由于整形数字不利于神经网络进行处理，因此需要将其转换为向量形式表示，最简单的方式为 one-hot 编码
对于一般的神经网络训练，由于需要给定多个文本，因此转换后的矩阵形式为：[batchsize,time,feature]
'''

import tensorflow as tf
import numpy as np

# 给定embedding矩阵w
init_w = np.random.random([4, 2])
W = tf.constant(init_w)

# 对于文字进行编号，这里文章仅包含四个字符
x = tf.constant([[0, 2, 1, 3]])

y = tf.nn.embedding_lookup(W, x)
sess = tf.Session()
print(sess.run(y))

batch_size = 1
# 数据有10个时间步，每个时间步向量长度为6
indata = tf.constant(np.random.random([batch_size, 10, 6]))
# 定义单一 RNN 函数
cell = tf.nn.rnn_cell.BasicLSTMCell(6, state_is_tuple=True)
'''
tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True):
n_hidden表示神经元的个数，forget_bias就是LSTM们的忘记系数，
如果等于1，就是不会忘记任何信息。如果等于0，就都忘记。
state_is_tuple默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示。
这个里面存在一个状态初始化函数，就是zero_state（batch_size，dtype）两个参数。
batch_size就是输入样本批次的数目，dtype就是数据类型。
'''

# 对于多层 RNN 可以 hi 使用辅助函数进行：
# cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(6, state_is_tuple=True) for _ in range(n_layers)], state_is_tuple=True)
# 将第一个 h0 赋值为0
state = cell.zero_state(batch_size, tf.float64)

# 定义输出列表
outputs = []
for time_step in range(10):
    # 循环的输入每一个时间步并获取输出和状态向量
    (cell_output, state) = cell(indata[:, time_step, :], state)
    # 存储列表
    outputs.append(cell_output)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 对结果进行输出
for idx, itr in enumerate(sess.run(outputs)):
    print("step%d:" % idx, itr)
