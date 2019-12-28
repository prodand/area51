import numpy as np

from plain.layers.base_layer import BaseLayer


class Flatten(BaseLayer):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, image):
        if image.shape != self.shape:
            raise RuntimeError('Image shape does not match')

        return image.reshape((self.shape[0] * self.shape[1] * self.shape[2], 1))

    def back(self, activation_theta):
        return activation_theta.reshape(self.shape)

    def update_weights(self, layer_cache, learning_rate):
        pass

    def save(self, folder: str):
        file_name = "/flatten_%s_%s_%s" % self.shape
        np.save(folder + file_name, np.array([self.shape[0], self.shape[1], self.shape[2]]))
        return file_name

    @staticmethod
    def load(folder, file):
        arr = np.load(folder + file + ".npy")
        return Flatten((arr[0], arr[1], arr[2]))