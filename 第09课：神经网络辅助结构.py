'''
@Desc  : 
@Date  : 2019/2/16
@Author: zhangjianfeng 
'''

'''
样本数据的分布变化（CovariateShift）

去均值处理可以减缓出现梯度问题的可能。为了使样本分布尽可能地统一，需要做的就是利用方差进行归一化

Dropout 层，此层产生的主要目的在于防止神经网络过拟合。

在解决过拟合问题时常用的方法为集成学习（Ensemble）。
将多个容易过拟合的网络（高方差、低偏差分类器）求平均以期获得更好的结果

Attention 机制用于改进 EncoderDecoder 的效果，它增加了 Encoder 与 Decoder 之间的信息通道。
最早的 EncoderDecoder 结构中，Encoder 传入 Decoder 的是最后一个状态向量，这个向量的长度是有限的，因此携带的信息有限。从另一个角度来说，需要引入更多向量来进行解码

'''