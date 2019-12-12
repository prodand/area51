from keras.datasets import mnist
import numpy as np
from cnn import Cnn
from layers.conv2d import Conv2d

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 32x32x3 | cv(3x3x32) -> 30x30x32 | mp(2x2) -> 15x15x32
cnn = Cnn()
cnn.add_layer(Conv2d(3, 32))
