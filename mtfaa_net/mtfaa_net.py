import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

root_dir = os.path.dirname(__file__)
sys.path.append(root_dir)

from pem import phase_encoder
from ed import encoder_decoder
from meam import mea

def lossFunction(y_true, y_pred):
    # shape: [B, T, F, C]
    speech = y_true[..., 0]
    noise = y_true[..., 1]
    est_speech = y_pred[..., 0]
    est_noise = y_pred[..., 1]
    
    speech_real = tf.math.real(speech)
    speech_imag = tf.math.imag(speech)
    noise_real = tf.math.real(noise)
    noise_imag = tf.math.imag(noise)
    est_speech_real = tf.math.real(est_speech)
    est_speech_imag = tf.math.imag(est_speech)
    est_noise_real = tf.math.real(est_noise)
    est_noise_imag = tf.math.imag(est_noise)
    
    loss = tf.math.abs(tf.math.log(tf.math.abs(speech_real) + tf.math.abs(speech_imag) + 1) - tf.math.log(tf.math.abs(est_speech_real) + tf.math.abs(est_speech_imag) + 1))
    loss = loss + tf.math.abs(tf.math.log(tf.math.abs(noise_real) + tf.math.abs(noise_imag) + 1) - tf.math.log(tf.math.abs(est_noise_real) + tf.math.abs(est_noise_imag) + 1))
    
    loss = tf.reduce_mean(loss, axis=-1)
    loss = tf.reduce_mean(loss, axis=-1)
    
    return loss


def mtfaa_net(input_shape, lr):
    # shape: [B, T, F, C]
    inputs = Input(shape=input_shape, dtype=tf.complex64)
    cin = inputs.shape[-1]
    
    pe = phase_encoder(inputs, channel=int(cin*4))
    bm = Conv2D(filters=int(cin*4), kernel_size=(1,2), strides=(1,1))(pe)
    ed = encoder_decoder(bm)
    bs = Conv2DTranspose(filters=cin, kernel_size=(1,2), strides=(1,1))(ed)
    outputs = mea(bs, inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=lr)
    model.compile(loss=lossFunction, optimizer=optimizer)    
    
    return model


if __name__ == "__main__":
    # shape: [B, T, F, C]
    input_shape = (200, 257, 1)
    model = mtfaa_net(input_shape, lr=0.001)
    model.summary()
