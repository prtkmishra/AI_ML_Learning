"""
********************************************************************************************
file: NuerNet.py
author: @Prateek Mishra
Description: Neural Network implementation for MNIST data
********************************************************************************************
"""

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization,AveragePooling2D
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
# import relevant library
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from emnist import extract_training_samples
from emnist import extract_test_samples

"""A dataset is required in order to train and test a neural network. Some standard datasets are
provided as part of Keras (https://keras.io/datasets/) and can be easily loaded. For example, to
load the MNIST dataset use the following python code:"""
# (xm_train, labelsm_train), (xm_test, labelsm_test) = mnist.load_data()

# (x_train, labels_train) = extract_training_samples('digits')
# (x_test, labels_test) = extract_test_samples('digits')

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

(xm_train, labelsm_train) = extract_training_samples('digits')
(xm_test, labelsm_test) = extract_test_samples('digits')

"""The MNIST images are stored in the form of integers with values in the range [0,255]. To convert
to floating-point numbers in the range [0,1] use the following python code:"""
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""The category labels are in the form of integers 0 to 9. To define the output that the network
should produce in response to each sample (a one hot encoding) use the following python code:"""
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

"""If the data is to be used as input to a convolutional layer, then it should be reshaped into a 
fourdimensional matrix where the first dimension corresponds to the number of exemplars, the
second and third dimensions correspond to the width and height of each image, and the fourth
dimension corresponds to the number of colour channels in each image:"""
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
datagen = ImageDataGenerator(rotation_range=10,zoom_range = 0.10,width_shift_range=0.1,height_shift_range=0.1)
datagen.fit(x_train)
"""The first method allows you to define a sequence of layers, it is assumed that the output of one
layer provides the input to the next. For example, to build a three layer MLP network called “net”
use the following python code:"""
net = Sequential()
"""Note that when adding a layer to the network, we can define parameters, such as: the number of
neurons (in the example above the 1st layer has 800 neurons, the second has 400 neurons, and
the 3rd layer has 10 neurons), and the activation function (in the example above the 1st and 2nd
layers use RELU, and the 3rd layer uses softmax). Other activation functions are also available
(see https://keras.io/activations/)."""
net.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',input_shape=(28,28,1)))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
net.add(BatchNormalization())
net.add(Dropout(rate=0.5))
net.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
net.add(BatchNormalization())
net.add(Dropout(rate=0.5))
net.add(Conv2D(64, (3, 3), activation='relu'))
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(BatchNormalization())
net.add(Conv2D(128, (3, 3), activation='relu'))
net.add(MaxPool2D(pool_size=(2, 2)))
net.add(BatchNormalization())
net.add(Flatten())
net.add(Dropout(rate=0.5))
net.add(Dense(256, activation='relu'))
net.add(Dropout(rate=0.5))
net.add(Dense(512, activation='relu'))
net.add(Dropout(rate=0.5))
net.add(Dense(1024, activation='relu'))
net.add(Dropout(rate=0.5))
net.add(Dense(512, activation='relu'))
net.add(Dropout(rate=0.5))
net.add(Dense(256, activation='relu'))
net.add(Dropout(rate=0.5))
net.add(Dense(10, activation='softmax'))
net.summary()
# plot_model(net, to_file='network_structure.png', show_shapes=True)

net.compile(loss='categorical_crossentropy', optimizer='adam')
# history = net.fit(x_train, y_train,validation_data=(x_test, y_test), epochs=120, batch_size=256)
history = net.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                    steps_per_epoch=len(x_train) / 256, epochs=120)
net.save("network_for_mnist.h5")
outputs=net.predict(x_test)
labels_predicted=np.argmax(outputs, axis=1)
misclassified=sum(labels_predicted!=labels_test)
print('Percentage misclassified = ',100*misclassified/labels_test.size)

xm_test = xm_test.astype('float32')
xm_test /= 255
xm_test = xm_test.reshape(xm_test.shape[0], 28, 28, 1)
outputs=net.predict(xm_test)
labels_predicted=np.argmax(outputs, axis=1)
misclassified=sum(labels_predicted!=labelsm_test)
print('Percentage misclassified = ',100*misclassified/labelsm_test.size)

xm_train = xm_train.astype('float32')
xm_train /= 255
xm_train = xm_train.reshape(xm_train.shape[0], 28, 28, 1)
outputs=net.predict(xm_train)
labels_predicted=np.argmax(outputs, axis=1)
misclassified=sum(labels_predicted!=labelsm_train)
print('Percentage misclassified = ',100*misclassified/labelsm_train.size)
