import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size)
        self.cached_image_vector = np.array([])

    def forward(self, image_vector):
        self.cached_image_vector = image_vector
        return self.weights.dot(image_vector) + self.bias

    def back(self, activation_theta, learning_rate):
        derivative_weights = activation_theta.T.dot(self.cached_image_vector)
        prev_layer_error = self.weights.T.dot(activation_theta)
        self.weights = self.weights - learning_rate * derivative_weights
        return prev_layer_error
