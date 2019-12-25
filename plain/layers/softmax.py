import numpy as np

from plain.layers.base_layer import BaseLayer


class Softmax(BaseLayer):

    def forward(self, input):
        input_shift = input - np.max(input)
        exp = np.exp(input_shift)
        return exp / np.sum(exp)

    def back(self, theta):
        return theta

    def update_weights(self, layer_cache, learning_rate):
        pass
