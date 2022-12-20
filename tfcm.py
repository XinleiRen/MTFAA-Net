import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

from cm import cmp

def tfcmb(inputs, channel, kernel=(3,3), dilation=(1,1), pad='same'):
    # shape: [B, T, F, C]
    p_conv1 = cmp(inputs, channel=channel, pad=pad)
    d_conv2d = cmp(p_conv1, channel=channel, kernel=kernel, dilation=dilation, group=channel, pad=pad)
    p_conv2 = Conv2D(filters=channel, kernel_size=(1,1), padding=pad)(d_conv2d)
    
    outputs = inputs + p_conv2
    return outputs

def tfcm(inputs, channel, kernel=(3,3), pad='same', block_num=6):
    # shape: [B, T, F, C]
    outputs = inputs
    for i in range(block_num):
        outputs = tfcmb(outputs, channel=channel, kernel=kernel, dilation=(2**i, 1), pad=pad)
        
    return outputs


if __name__ == "__main__":
    input_shape = (200, 257, 2)
    inputs = Input(shape=input_shape)
    outputs = tfcm(inputs, channel=2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()