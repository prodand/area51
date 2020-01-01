import signal

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

from cnn import Cnn
from plain.layers.flatten import Flatten
from plain.layers.maxPooling import MaxPooling
from plain.layers.softmax import Softmax
from vectorized.layers.conv2d import Conv2d
from vectorized.layers.fully_connected import FullyConnected

cnn = Cnn(300, 0.07)


def keyboard_interrupt_handler(signal, frame):
    cnn.save()
    exit(0)


def main():
    plt.ion()
    # cnn = Cnn.load("saved/cnn_12-28-2019_19-36")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    labels = np.zeros((y_train.shape[0], 10, 1))
    for index, x in enumerate(y_train):
        labels[index, x] = [1]

    images = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])) / 255
    test_images = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])) / 255

    # cnn.add_layer(Conv2d(16, 5, 1))
    # cnn.add_layer(MaxPooling((16, 24, 24), 2, 2))
    # cnn.add_layer(Conv2d(32, 3, 16))
    # cnn.add_layer(MaxPooling((32, 10, 10), 2, 2))
    # cnn.add_layer(Flatten((32, 5, 5)))
    # cnn.add_layer(FullyConnected(800, 10))
    # cnn.add_layer(Softmax())

    cnn.add_layer(Conv2d(16, 5, 1))
    cnn.add_layer(MaxPooling((16, 24, 24), 2, 2))
    cnn.add_layer(Conv2d(32, 3, 16))
    cnn.add_layer(MaxPooling((32, 10, 10), 2, 2))
    cnn.add_layer(Flatten((32, 5, 5)))
    cnn.add_layer(FullyConnected(800, 128))
    cnn.add_layer(FullyConnected(128, 10))
    cnn.add_layer(Softmax())

    # cnn.add_layer(Conv2d(16, 5, 1))
    # cnn.add_layer(MaxPooling((16, 24, 24), 2, 2))
    # cnn.add_layer(Conv2d(32, 5, 16))
    # cnn.add_layer(MaxPooling((32, 8, 8), 2, 2))
    # cnn.add_layer(Flatten((32, 4, 4)))
    # cnn.add_layer(FullyConnected(512, 10))
    # cnn.add_layer(Softmax())

    # cnn.add_layer(Conv2d(32, 5, 1))
    # cnn.add_layer(MaxPooling((32, 24, 24), 2, 2))
    # cnn.add_layer(Conv2d(64, 5, 32))
    # cnn.add_layer(MaxPooling((64, 8, 8), 2, 2))
    # cnn.add_layer(Flatten((64, 4, 4)))
    # cnn.add_layer(FullyConnected(1024, 10))
    # cnn.add_layer(Softmax())

    signal.signal(signal.SIGINT, keyboard_interrupt_handler)
    print('Start learning')
    cnn.learn(images, labels)

    index = cnn.predict(test_images[0])
    print(y_test[0] + ' ' + index)


if __name__ == '__main__':
    main()
