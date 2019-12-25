from keras.datasets import mnist
import numpy as np
from cnn import Cnn
from layers.conv2d import Conv2d
from layers.flatten import Flatten
from layers.fully_connected import FullyConnected
from layers.maxPooling import MaxPooling
from layers.softmax import Softmax

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 28x28x1 | cv(5x5x8) -> 24x24x8 | mp(2x2) -> 12x12x8 | cv(3x3x16) -> 10x10x16 |
# mp(2x2) -> 5x5x16 | cv(3x3x32) -> 3x3x32

labels = np.zeros((y_train.shape[0], 10, 1))
for index, x in enumerate(y_train):
    labels[index, x] = [1]

images = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])) / 255
test_images = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])) / 255

cnn = Cnn(600)
cnn.add_layer(Conv2d(8, 5, 1))
cnn.add_layer(MaxPooling((8, 24, 24), 2, 2))
cnn.add_layer(Conv2d(16, 3, 8))
cnn.add_layer(MaxPooling((16, 10, 10), 2, 2))
# cnn.add_layer(Conv2d(32, 3, 16))
# cnn.add_layer(Flatten((32, 3, 3)))
cnn.add_layer(Flatten((16, 5, 5)))
# cnn.add_layer(FullyConnected(288, 10))
cnn.add_layer(FullyConnected(400, 10))
cnn.add_layer(Softmax())
print('Start learning')
cnn.learn(images, labels)

index = cnn.predict(test_images[0])
print(y_test[0] + ' ' + index)
