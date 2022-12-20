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
    # shape: [B, T, F]
    speech_real = tf.math.real(y_true)
    speech_imag = tf.math.imag(y_true)
    est_speech_real = tf.math.real(y_pred)
    est_speech_imag = tf.math.imag(y_pred)
    
    loss = tf.math.abs(tf.math.log(tf.math.abs(speech_real) + tf.math.abs(speech_imag) + 1) - tf.math.log(tf.math.abs(est_speech_real) + tf.math.abs(est_speech_imag) + 1))
    loss = tf.reduce_mean(loss, axis=-1)
    loss = tf.reduce_mean(loss, axis=-1)
    
    return loss


def mtfaa_net(input_shape, lr):
    # shape: [B, T, F, C]
    inputs = Input(shape=input_shape)
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