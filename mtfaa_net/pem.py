import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

def phase_encoder(inputs, channel=4, kernel=(3,1), pad='same', com_factor=0.5):
    real_conv2d = Conv2D(filters=channel, kernel_size=kernel, padding=pad)
    imag_conv2d = Conv2D(filters=channel, kernel_size=kernel, padding=pad)
    
    # shape: [B, T, F, C]
    real_input = tf.math.real(inputs)
    imag_input = tf.math.imag(inputs)

    real2real = real_conv2d(real_input)
    real2imag = imag_conv2d(real_input)

    imag2real = real_conv2d(imag_input)
    imag2imag = imag_conv2d(imag_input)

    real_outputs = real2real - imag2imag
    imag_outputs = real2imag + imag2real
        
    # modulo
    outputs = tf.math.sqrt(real_outputs ** 2 + imag_outputs ** 2 + tf.keras.backend.epsilon())
    # compression
    outputs = tf.math.pow(outputs, com_factor)

    return outputs


if __name__ == "__main__":
    input_shape = (200, 257, 1)
    inputs = Input(shape=input_shape)
    outputs = phase_encoder(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()