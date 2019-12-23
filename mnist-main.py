from keras.datasets import mnist
import numpy as np
from cnn import Cnn
from layers.conv2d import Conv2d
from layers.maxPooling import MaxPooling

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 32x32x3 | cv(5x5x8) -> 28x28x8 | mp(2x2) -> 14x14x8 | cv(3x3x16) -> 12x12x16 | mp(2x2) -> 6x6x16 | cv(3x3x32) -> 4x4x32

cnn = Cnn()
cnn.add_layer(Conv2d(8, 5, 1))
cnn.add_layer(MaxPooling(28, 2, 2))
cnn.add_layer(Conv2d(16, 3, 8))
cnn.add_layer(MaxPooling(12, 2, 2))
cnn.add_layer(Conv2d(32, 3, 16))
