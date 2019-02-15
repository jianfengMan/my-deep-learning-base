'''
@Desc  : 
@Date  : 2019/2/15
@Author: zhangjianfeng 
'''

'''
线性独立的概念很重要。如果几个向量线性不独立，即某个向量可以用其他向量表示，
那么这个向量就没有存储的必要了。这是信息压缩最原始的思想。

对于 SVD 分解而言，其有一个非常大的问题就是约束过于严格，比如矩阵 M 与 V 为正交矩阵。
这就导致了在计算过程中，为了满足分解条件，信息压缩的质量可能会降低。因此产生了另外一个更加宽泛的约束方式：
Amn ≈Mnp⋅Npm 假设条件是 N 足够稀疏，此时 M 就称为字典。此时弱化了正交性假设，因此所得到的信息压缩效果会更加出色。
'''
#
# 矩阵乘法
# 引入库
import tensorflow as tf
import numpy as np

# 定义常量并用 numpy 进行初始化
a1 = tf.constant(np.ones([4, 4]))
a2 = tf.constant(np.ones([4, 4]))
# 矩阵乘法
a1_dot_a2 = tf.matmul(a1, a2)
# 定义会话用于执行计算
sess = tf.Session()
# 执行并输出
print(sess.run(a1_dot_a2))

# variable 需要初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(a1_dot_a2))

# 定义变量并用 numpy 进行初始化
a1 = tf.Variable(np.ones([4, 4]))
# 定义 placeholder 用于从外部接收数据
a2 = tf.placeholder(dtype=tf.float64, shape=[4, 4])
# 矩阵乘法
a1_dot_a2 = tf.matmul(a1, a2)
# variable 需要初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 需要给 placeholder 提供数据
print(sess.run(a1_dot_a2, feed_dict={a2: np.ones([4, 4])}))
