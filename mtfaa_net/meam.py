import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

def mea(inputs, stft_spec, channel=2):
    # shape: [B, T, F, C]
    complex_mask = Conv2D(filters=channel, kernel_size=(1,1))(inputs)
    
    # shape: [B, T, F]
    real_mask = complex_mask[..., 0]
    imag_mask = complex_mask[..., 1]
    complex_mask = tf.complex(real_mask, imag_mask)
    
    est_stft = tf.math.multiply(stft_spec[..., 0], complex_mask)

    return est_stft


if __name__ == "__main__":
    inputs = tf.ones([8, 200, 384, 1])
    stft_spec = tf.complex(tf.ones([8, 200, 384, 1]), tf.ones([8, 200, 384, 1]))
    outputs = mea(inputs, stft_spec)
    
    print('inputs.shape:', inputs.shape)
    print('outputs.shape:', outputs.shape)
