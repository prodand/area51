import numpy as np

from layers.base_layer import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.random.randn(output_size, 1) * 0.1

    def forward(self, image_vector):
        return self.weights.dot(image_vector) + self.bias

    def back(self, activation_theta):
        prev_layer_error = self.weights.T.dot(activation_theta)
        return prev_layer_error

    def update_weights(self, layer_cache, learning_rate):
        derivative_weights = np.zeros(self.weights.shape)
        derived_biases = np.zeros(self.bias.shape)
        for (image_vector, activation_theta) in layer_cache:
            derivative_weights += activation_theta.dot(image_vector.T)
            derived_biases += activation_theta
        self.weights = self.weights - learning_rate * (derivative_weights / len(layer_cache))
        self.bias = self.bias - learning_rate * (derived_biases / len(layer_cache))