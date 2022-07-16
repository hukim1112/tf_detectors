import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, ReLU, Add # Popular layers

def dconv(in_depth, out_depth):
    return tf.keras.Sequential([Conv2D(in_depth, 1, padding='same'),
                    BatchNormalization(),
                    ReLU(),

                    DepthwiseConv2D(3, padding='same'),
                    BatchNormalization(),
                    ReLU(),

                    Conv2D(out_depth, 1, padding='same'),
                    BatchNormalization()])

def dconv_downsampled(in_depth, out_depth):
    return tf.keras.Sequential([tf.keras.layers.Conv2D(in_depth, 1, padding='same'),
                    BatchNormalization(),
                    ReLU(),

                    DepthwiseConv2D(3, strides=(2,2), padding='same'),
                    BatchNormalization(),
                    ReLU(),

                    Conv2D(out_depth, 1, padding='same'),
                    BatchNormalization()])


def classification_head(_input, channel_depth, fmap_size, num_classes, num_anchors):
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(_input)
    x = ReLU()(x)
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(num_classes*num_anchors, kernel_size=3, padding='same')(x)
    #x = tf.keras.activations.sigmoid(x)
    x = tf.reshape(x, [-1, fmap_size*fmap_size*num_anchors,num_classes])
    return x

def bbox_regression_head(_input, channel_depth, fmap_size, coords, num_anchors):
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(_input)
    x = ReLU()(x)
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(channel_depth, kernel_size=3, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(coords*num_anchors, kernel_size=3, padding='same')(x)
    x = tf.reshape(x, [-1, fmap_size*fmap_size*num_anchors,coords])
    return x
