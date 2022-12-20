import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

from cm import cmp, cmt, ctmp
from asam import asa
from tfcm import tfcm

def fu_block(inputs, skip_inputs, channel, kernel=(1,7), stride=(1,4), group=2, pad='same'):
    # shape: [B, T, F, C]
    conv1 = cmt(tf.concat([inputs, skip_inputs], axis=-1), channel, pad=pad)
    conv1 = tf.math.multiply(conv1, skip_inputs)
    conv2 = cmp(conv1, channel, pad=pad)
    deconv = ctmp(conv2, channel, kernel=kernel, stride=stride, group=group, pad=pad)
    
    return deconv
    

def encoder_decoder(inputs, channel=(48, 96, 192), kernel=(1,7), stride=(1,4), group=2, pad='same'):
    # shape: [B, T, F, C]
    outputs = inputs
    ed_block_num = len(channel)
    bl_block_num = 2
    
    # encoder
    fd_out = []
    for i in range(ed_block_num):
        outputs = cmp(outputs, channel[i], kernel=kernel, stride=stride, group=group, pad=pad)
        fd_out.append(outputs)
        outputs = tfcm(outputs, channel[i], pad=pad)
        outputs = asa(outputs, channel[i], pad=pad)
        
    # bottleneck
    for i in range(bl_block_num):
        outputs = tfcm(outputs, channel[-1], pad=pad)
        outputs = asa(outputs, channel[-1], pad=pad)
    
    # decoder
    for i in range(ed_block_num):
        outputs = fu_block(outputs, fd_out[ed_block_num-1-i], channel[ed_block_num-1-i], kernel=kernel, stride=stride, group=group, pad=pad)
        outputs = tfcm(outputs, channel[ed_block_num-1-i], pad=pad)
        outputs = asa(outputs, channel[ed_block_num-1-i], pad=pad)
        
    return outputs


if __name__ == "__main__":
    input_shape = (200, 256, 4)
    inputs = Input(shape=input_shape)
    outputs = encoder_decoder(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()