"""
********************************************************************************************
file: NuerNet.py
author: @Prateek Mishra
Description: AlexNet implementation for Fashion MNIST data
********************************************************************************************
"""
import requests
import tensorflow as tf
from d2l import tensorflow as d2l

def net():
    return tf.keras.models.Sequential([
            """Filters = 96, Kernel size = 11x11, stride = 4, activation = relu"""
            tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,activation='relu'),
            """maxpooling layer of size 3 and stride 2"""
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            """Filters = 256, Kernel size = 5x5, activation = relu"""
            tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',activation='relu'),
            """maxpooling layer of size 3 and stride 2"""        
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            """Use three successive convolutional layers and a smaller convolution window"""
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',activation='relu'),
            tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',activation='relu'),
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            """Flatten out the output for dense layer"""
            tf.keras.layers.Flatten(),
            """Start MLP"""
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            """For Fashion MNIST the output classes are 10"""
            tf.keras.layers.Dense(10)])

"""construct a single-channel data example with both height and width of 224 to observe the output shape of each layer."""
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

""" Train the model"""
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())