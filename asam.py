import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

from cm import cmp

def asaf(inputs, channel, pad='same'):
    # shape: [B, T, F, C]
    k = cmp(inputs, channel, pad=pad)
    q = cmp(inputs, channel, pad=pad)
    v = cmp(inputs, channel, pad=pad)
    
    # shape: [B, T, F, C] -> [B, T, C, F]
    k = tf.transpose(k, [0, 1, 3, 2])
    
    # shape: [B, T, F, F]
    fam = tf.matmul(q, k) / tf.sqrt(channel * 1.0)
    fam = Softmax()(fam)
    # shape: [B, T, F, C]
    fa = tf.matmul(fam, v)

    return fa

def asat(inputs, f_att, channel, a_frames=300, pad='same'):
    # shape: [B, T, F, C]
    k = cmp(inputs, channel, pad=pad)
    q = cmp(inputs, channel, pad=pad)

    # shape: [B, T, F, C] -> [B, F, T, C]
    q = tf.transpose(q, [0, 2, 1, 3])
    # shape: [B, T, F, C] -> [B, F, C, T]
    k = tf.transpose(k, [0, 2, 3, 1])
        
    # shape: [B, F, T, T]
    tam = tf.matmul(q, k) / tf.sqrt(channel * 1.0)
        
    a_mask = tf.math.equal(tf.linalg.band_part(tf.ones_like(tam, dtype=tf.int32), a_frames // 2, a_frames // 2), tf.ones_like(tam, dtype=tf.int32))
    a_mask_value = tf.where(a_mask, tf.zeros_like(tam), -1e10 * tf.ones_like(tam))
    tam = tam + a_mask_value
        
    tam = Softmax()(tam)
    # shape: [B, F, T, C]
    aa = tf.matmul(tam, tf.transpose(f_att, [0, 2, 1, 3]))
    # shape: [B, F, T, C] -> [B, T, F, C]
    aa = tf.transpose(aa, [0, 2, 1, 3])

    return aa

def asa(inputs, in_channel, a_frames=300, pad='same'):
    # shape: [B, T, F, C]
    fa = asaf(inputs, int(in_channel / 4), pad=pad)
    aa = asat(inputs, fa, int(in_channel / 4), a_frames, pad=pad)
    conv = cmp(aa, in_channel, pad=pad)
    outputs = inputs + conv
    
    return outputs


if __name__ == "__main__":
    input_shape = (200, 257, 4)
    inputs = Input(shape=input_shape)
    outputs = asa(inputs, in_channel=4)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()