import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

def mea(inputs, stft_spec, channel=1, rm_size=(1,3)):
    # shape: [B, T, F, C]
    real_spec = tf.math.real(stft_spec)
    imag_spec = tf.math.imag(stft_spec)
    
    mag = tf.math.sqrt(real_spec ** 2 + imag_spec ** 2 + tf.keras.backend.epsilon())
    pha = tf.math.atan2(imag_spec, real_spec + tf.keras.backend.epsilon())
    
    # stage 1
    t_size = rm_size[0]
    f_size = rm_size[1]
    padding = tf.constant([[0,0], [(t_size-1)//2,(t_size-1)//2], [(f_size-1)//2,(f_size-1)//2], [0,0]])
    rm_kernel = tf.ones([t_size, f_size, channel, channel])
    
    mag_mask1 = Conv2D(filters=channel, kernel_size=(1,1))(inputs)
    mag_mask1 = tf.sigmoid(mag_mask1)
    est_mag1 = tf.math.multiply(mag, mag_mask1)
    est_mag1 = tf.pad(est_mag1, padding)
    est_mag1 = tf.nn.conv2d(est_mag1, tf.constant(rm_kernel), strides=1, padding='VALID')
        
    # stage 2
    real_mask2 = Conv2D(filters=channel, kernel_size=(1,1))(inputs)
    imag_mask2 = Conv2D(filters=channel, kernel_size=(1,1))(inputs)

    mag_mask2 = tf.math.sqrt(real_mask2 ** 2 + imag_mask2 ** 2 + tf.keras.backend.epsilon())
    mag_mask2 = tf.math.tanh(mag_mask2)
    pha_mask2 = tf.math.atan2(imag_mask2, real_mask2 + tf.keras.backend.epsilon())

    est_mag2 = tf.math.multiply(est_mag1, mag_mask2)
    est_real = est_mag2 * tf.math.cos(pha + pha_mask2)
    est_imag = est_mag2 * tf.math.sin(pha + pha_mask2)

    # multi -> mono
    # est_real = tf.math.reduce_mean(est_real, axis=-1, keepdims=True)
    # est_imag = tf.math.reduce_mean(est_imag, axis=-1, keepdims=True)
        
    # shape: [B, T, F]
    est_stft = tf.complex(est_real[..., 0], est_imag[..., 0])

    return est_stft


if __name__ == "__main__":
    inputs = tf.ones([8, 200, 384, 1])
    stft_spec = tf.complex(tf.ones([8, 200, 384, 1]), tf.ones([8, 200, 384, 1]))
    outputs = mea(inputs, stft_spec)
    
    print('inputs.shape:', inputs.shape)
    print('outputs.shape:', outputs.shape)
