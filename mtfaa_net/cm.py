from tensorflow.keras.layers import *
from tensorflow.keras.activations import *

def cmp(inputs, channel, kernel=(1,1), stride=(1,1), dilation=(1,1), group=1, pad='same'):
    # shape: [B, T, F, C]
    outputs = Conv2D(filters=channel, kernel_size=kernel, strides=stride, dilation_rate=dilation, groups=group, padding=pad)(inputs)
    outputs = BatchNormalization()(outputs)
    # outputs = PReLU()(outputs)
    outputs = ELU()(outputs)
    
    return outputs

def cmt(inputs, channel, kernel=(1,1), stride=(1,1), dilation=(1,1), group=1, pad='same'):
    # shape: [B, T, F, C]
    outputs = Conv2D(filters=channel, kernel_size=kernel, strides=stride, dilation_rate=dilation, groups=group, padding=pad)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = tanh(outputs)
    
    return outputs

def ctmp(inputs, channel, kernel=(1,1), stride=(1,1), dilation=(1,1), group=1, pad='same'):
    # shape: [B, T, F, C]
    outputs = Conv2DTranspose(filters=channel, kernel_size=kernel, strides=stride, dilation_rate=dilation, groups=group, padding=pad)(inputs)
    outputs = BatchNormalization()(outputs)
    # outputs = PReLU()(outputs)
    outputs = ELU()(outputs)
    
    return outputs