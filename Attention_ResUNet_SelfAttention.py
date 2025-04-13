
'''
Squeeze and excitation UNet with Self Attention Block


Dependencies:
    Tensorflow 2.16

'''


import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

tf.keras.backend.set_image_data_format('channels_last')
# input data
INPUT_SIZE = 128
INPUT_CHANNEL = 3 # 1-grayscale, 3-RGB scale
OUTPUT_MASK_CHANNEL = 1


NUM_FILTER = 32
FILTER_SIZE = 3
UP_SAMP_SIZE = 2

def squeeze_and_excitation_block(input_X, reduction_ratio=16):
    """
    SE-Block: Betont relevante Feature Maps durch globales Kontextverständnis
    :param input_X: Eingabetensor (Feature-Map)
    :param reduction_ratio: Reduktion der Kanalanzahl für den Squeeze-Schritt
    """
    channels = input_X.shape[-1]  # Anzahl der Kanäle
    squeeze = tf.reduce_mean(input_X, axis=[1, 2], keepdims=True)
    
    excitation = layers.Dense(units=channels // reduction_ratio, activation='relu')(squeeze)
    excitation = layers.Dense(units=channels, activation='sigmoid')(excitation)
    
    return input_X * excitation

def conv_block(x, filters):
    """
    Standard Convolution Block: 2x Convolution + BatchNorm + ReLU
    """
    x = layers.Conv2D(filters, (FILTER_SIZE, FILTER_SIZE), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, (FILTER_SIZE, FILTER_SIZE), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def upsample_block(x, filters):
    """
    Upsampling Block: UpSampling2D + Convolution zur Kanalanpassung
    """
    x = layers.UpSampling2D((UP_SAMP_SIZE, UP_SAMP_SIZE))(x)
    x = layers.Conv2D(filters, (UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def gating_signal(input_tensor, out_size):
    """
    Gating Signal: 1x1 Convolution zum Anpassen der Kanäle
    """
    gating = layers.Conv2D(out_size, kernel_size=(1, 1), padding="same")(input_tensor)
    gating = layers.BatchNormalization()(gating)
    gating = layers.Activation('relu')(gating)
    return gating


class SelfAttentionBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.softmax = layers.Softmax(axis=-1)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        reduced_channels = tf.math.floordiv(tf.cast(num_channels, tf.int32), 8)

        self.query_conv = layers.Conv2D(reduced_channels, kernel_size=1, padding='same', activation='relu')
        self.key_conv = layers.Conv2D(reduced_channels, kernel_size=1, padding='same', activation='relu')
        self.value_conv = layers.Conv2D(num_channels, kernel_size=1, padding='same')

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        width = shape[1]
        height = shape[2]
        num_channels = shape[3]
        reduced_channels = self.query_conv.filters

        query = self.query_conv(inputs)
        key = self.key_conv(inputs)
        value = self.value_conv(inputs)

        query_reshaped = tf.reshape(query, (batch_size, -1, reduced_channels))
        key_reshaped = tf.reshape(key, (batch_size, -1, reduced_channels))
        value_reshaped = tf.reshape(value, (batch_size, -1, num_channels))

        attention = tf.matmul(query_reshaped, tf.transpose(key_reshaped, perm=[0, 2, 1]))
        attention = self.softmax(attention)

        out = tf.matmul(attention, value_reshaped)
        out = tf.reshape(out, (batch_size, width, height, num_channels))

        return out + inputs # Optional: Add residual connection

def Attention_ResUNet_SelfAttention(input_shape=( 512 , 512 , 9  )):
    inputs = tf.keras.Input(input_shape)

    NUM_FILTER = 32
    UP_SAMP_SIZE = 2

    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def upsample_block(x, filters):
        x = layers.UpSampling2D((UP_SAMP_SIZE, UP_SAMP_SIZE))(x)
        x = layers.Conv2D(filters, (UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def gating_signal(input_tensor, out_size):
        gating = layers.Conv2D(out_size, kernel_size=(1, 1), padding="same")(input_tensor)
        gating = layers.BatchNormalization()(gating)
        gating = layers.Activation('relu')(gating)
        return gating

    # Encoder-Pfad
    c1 = conv_block(inputs, NUM_FILTER)
    p1 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c1)

    c2 = conv_block(p1, NUM_FILTER * 2)
    p2 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c2)

    c3 = conv_block(p2, NUM_FILTER * 4)
    p3 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c3)

    c4 = conv_block(p3, NUM_FILTER * 8)
    p4 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c4)

    c5 = conv_block(p4, NUM_FILTER * 16)
    p5 = layers.MaxPooling2D((UP_SAMP_SIZE, UP_SAMP_SIZE), padding="same")(c5)
    # Bottleneck
    c6 = conv_block(p5, NUM_FILTER * 32)
    gating = gating_signal(c6, NUM_FILTER * 32)

    # Decoder-Pfad mit Self-Attention-Blocks und Skip-Connections
    u7 = upsample_block(gating, NUM_FILTER * 16)
    # Downsample before self-attention
    down_sa7 = layers.MaxPooling2D((2, 2))(u7)
    sa7 = SelfAttentionBlock()(down_sa7)
    # Upsample after self-attention
    up_sa7 = layers.UpSampling2D((2, 2))(sa7)
    c7 = conv_block(layers.concatenate([up_sa7, c5]), NUM_FILTER * 16)

    u8 = upsample_block(c7, NUM_FILTER * 8)
    sa8 = SelfAttentionBlock()(u8) # Apply at original resolution for others
    c8 = conv_block(layers.concatenate([sa8, c4]), NUM_FILTER * 8)

    u9 = upsample_block(c8, NUM_FILTER * 4)
    sa9 = SelfAttentionBlock()(u9)
    c9 = conv_block(layers.concatenate([sa9, c3]), NUM_FILTER * 4)

    u10 = upsample_block(c9, NUM_FILTER * 2)
    # sa10 = SelfAttentionBlock()(u10) # OOM Issue removed 
    # c10 = conv_block(layers.concatenate([sa10, c2]), NUM_FILTER * 2)
    c10 = conv_block(layers.concatenate([u10, c2]), NUM_FILTER * 2)

    u11 = upsample_block(c10, NUM_FILTER)
    # sa11 = SelfAttentionBlock()(u11)   # OOM Issue removed 
    # c11 = conv_block(layers.concatenate([sa11, c1]), NUM_FILTER)
    c11 = conv_block(layers.concatenate([u11, c1]), NUM_FILTER)
    # Ausgabeschicht
    conv_final = layers.Conv2D(1, (1, 1), activation="sigmoid")(c11)

    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)

    model = tf.keras.models.Model(inputs=[inputs], outputs=conv_final , name="Attention_ResUNet_SelfAttention")
    return model
