"""
********************************************************************************************
file: VGGNet.py
author: @Prateek Mishra
Description: VGG implementation for Fashion MNIST data
********************************************************************************************
"""

import tensorflow as tf
from d2l import tensorflow as d2l

"""The function takes two arguments corresponding to the number of convolutional layers num_convs and the number of output channels num_channels."""

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk

""" Define the convolution architecture for VGGs (num_convs, num_channels) """
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))    

"""The following code implements VGG-11. This is a simple matter of executing a for-loop over conv_arch."""
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(
        tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)

"""construct a single-channel data example with a height and width of 224 to observe the output shape of each layer."""
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

"""Since VGG-11 is more computationally-heavy than AlexNet we construct a network with a smaller number of channels. This is more than sufficient for training on Fashion-MNIST."""
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = lambda: vgg(small_conv_arch)

""" Train the model """
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
