#!/usr/bin/env python
# coding=utf-8

from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    '''
        这个方法其实是 dilated causal convolutional

        causal convolutional 的思路是 (卷积核 为 1 × 2 时)
            每次在前面添加两个0占位，经过卷积的结果再去掉最后一个，造成每次卷积完输入向右移动一格用0占位的效果
            （补充0的数量根据 卷积核的大小决定）
            输入: n1 n2 n3 n4 n5
            layer1:                     n1 n2 n3 n4 n5 -> 0 0 n1 n2 n3 n4 n5
            layer2: 0 n1 n2 n3 n4 n5 ->  0 n1 n2 n3 n4 -> 0 0  0 n1 n2 n3 n4
            layer3: 0  0 n1 n2 n3 n4 ->  0  0 n1 n2 n3 -> 0 0  0  0 n1 n2 n3
            layer4: 0  0  0 n1 n2 n3 ->  0  0  0 n1 n2 -> 0 0  0  0  0 n1 n2
            layer5: 0  0  0  0 n1 n2 ->  0  0  0  0 n1
            最终的label是 n2 n3 n4 n5 0
            看中间那列（每层的n1-n5是不同的），layer5 的 n1 是由 layer1 的 n1 n2 n3 n4 n5 决定的
            y(n1_layer5) = p(n1_layer5|n1_layer1, n2_layer1, n3_layer1, n4_layer1, n5_layer1)
            如果看 layer4 的 n1 会有 y(n1_layer4) = p(n1_layer4 | n1_layer1, n2_layer1, n3_layer1, n4_layer1) 而忽略layer5的情况下，与 n1_layer4 对应的是 n5_label
            这样 n1_layer4 就会拟合 n5_label 而输入层的下一个就是 n5 所以正好把输出变成了输入
            就有了 wavenet 官网 https://deepmind.com/blog/wavenet-generative-model-raw-audio/ 上的那个动图的效果了

            为什么 n1_layer5 对应的是 0_label 是因为论文中的联合概率公式定义是：p(x) = (连乘 t->1...n) p(xt | x1, x2, ....., xt-1)
            公式的定义是第 t 个元素是由 前 t-1 个元素决定的，而 n1_layer5 是由 n1_layer1, n2_layer1, n3_layer1, n4_layer1, n5_layer1 决定的，
            根据公式 n1_layer1, n2_layer1, n3_layer1, n4_layer1, n5_layer1 决定的是 n6_layer1 而 n6_layer1 应该对应的是输入的 n6 但是 n5 就已经结束了
            所以 n1_layer5 就定义为对应 0_label 可以当做结束标志

        dilated causal convolutional 的思路是：
            causal convolutional 的思路同上
            dilated 的思路是
            输入: n1 n2 n3 n4 n5 n6 n7 n8 （内个n向量是256维）
            shape: 1, 8, 256          -> [[[n1][n2][n3][n4][n5][n6][n7][n8]]]
            dilated: 4 的情况下
            转变shape为: 4, 2, 256    -> [[[n1][n2]]
                                         [[n3][n4]]
                                         [[n5][n6]]
                                         [[n7][n8]]]

            再 transpose 成 2, 4, 256 -> [[[n1][n3][n5][n7]]
                                         [[n2][n4][n6][n8]]]
            这样在用普通的卷积层计算就能达到论文中的 Figure 3 的效果
    '''
    with tf.name_scope(name):
        # Pad beforehand to preserve causality.
        filter_width = tf.shape(filter_)[0]
        padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
        padded = tf.pad(value, padding)
        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='SAME')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, filter_, stride=1, padding='SAME')
        # Remove excess elements at the end.
        result = tf.slice(restored, [0, 0, 0], [-1, tf.shape(value)[1], -1])
        return result


def mu_law_encode(audio, quantization_channels):
    '''
        Quantizes waveform amplitudes.
        把输入的数据归一化到 0～255
    '''
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988).
        magnitude = tf.log(1 + mu * tf.abs(audio)) / tf.log(1. + mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        casted = tf.cast(output, tf.float32)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude
