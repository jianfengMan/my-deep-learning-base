'''
@Desc  : 
@Date  : 2019/2/16
@Author: zhangjianfeng 
'''

'''
超参数是在模型训练之前调整的参数。包括隐藏节点数、DropOut 比例、学习率等

学习率是重要的超参数之一，好的学习率可以有效地加快迭代收敛速度，并能避免迭代发散问题。
学习率的值可以按10的倍数调整，比如取{0.1,0.01，…}。对于已经训练好的网络而言，
可以使用较小的学习率来继续训练。而对于新的网络，可以选择比较大的学习率
现有的优化算法，比如 AdaGrad、Adam 等，就是自适应的调整学习

BATCHSIZE 可以在32范围内调整，大的 BATCHSIZE 可以使梯度的估计更加准确，但是内存消耗也在不断地增多，同时迭代速度并未明显升高。

Embedding 大小与 dropout 数值:
自然语言处理过程需要将词向量进行降维。EmbeddingSize 可以从128左右调整。
Drop 参数选择0.5是合适的。

网格搜索方法:本质是有限集合的笛卡尔积


数据预处理：
深度神经网络虽然弱化了数据特征工程，
但是依然需要对数据进行预处理。这些预处理方式包括去均值以及协方差均衡：

梯度剪裁：
为了避免训练过程出现梯度膨胀（gradient explosion）问题，可以对梯度进行剪裁（clip gradient）
去掉trshold小的

网络结构优化：
么扩大感受野的最简单方式就是增加神经网络深度，或者增大卷积核心。两种方式都会有效地增大感受野。
但是单纯增加网络深度或者增大卷积核心会导致整个计算过程难以进行。
因此可以进行相应优化：[5,5]的卷积核心与两层的[3,3]卷积核心感受野相同，但是可训练参数的数量变为了3*3*2=
与原有大小为25的可训练参数数量相比，进行了有效缩减，因此训练过程更容易执行
深度神经网络的可训练参数是冗余的，适当减少可训练参数并不会影响神经网络的表达能力
'''

'''
Inception 网络的思想就是将不同感受野大小的卷积网络的结果进行连接。由此可以学习不同尺度的特征信息，从而增强特征的有效性
ResNet:
进行深度神经网络结构优化时，另外一个思路就是增加支路 y=f(x)+x(1.4)式中
f 为卷积结构，x 为神经网络输入。这样通过添加支路的方式可以使训练过程更加有效。
原因是添加了支路，整个梯度传播过程可以沿支路进行。这样一来，整个深度神经网络更加类似于一个浅层神经网络，
因此深度残差网络可以变得很深，有些网络会达到1000层。
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim

def InceptionResnetDemo(net, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """
    本例是将 Inception 网络与 ResNet 结构做融合。
    """
    with tf.variable_scope(scope, 'InceptionResnet', [net], reuse=reuse):
        # Inception 网络部分
        with tf.variable_scope('Branch_0'):
            # 第一个支路
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            # 第二个支路
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            # 第三个支路
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        # 学习到的所有特征都进行保存
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        # 残差网络部分
        net = net + up
        if activation_fn:
            net = activation_fn(net)
    return net
