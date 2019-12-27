import numpy as np

from plain.layers.base_layer import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * (2 / np.sqrt(input_size))
        self.bias = np.random.randn(output_size, 1) * (2 / np.sqrt(output_size))

    def forward(self, image_vector):
        return self.relu(self.weights.dot(image_vector) + self.bias)

    def back(self, activation_theta):
        prev_layer_error = self.weights.T.dot(activation_theta)
        return prev_layer_error

    def update_weights(self, layer_cache, learning_rate):
        images_matrix = []
        activation_theta_matrix = []
        derived_biases = np.zeros(self.bias.shape)
        for (image_vector, activation_theta) in layer_cache:
            images_matrix = np.column_stack((images_matrix, image_vector)) \
                if len(images_matrix) > 0 else image_vector
            activation_theta_matrix = np.column_stack((activation_theta_matrix, activation_theta)) \
                if len(activation_theta_matrix) > 0 else activation_theta
            derived_biases += activation_theta

        derivative_weights = activation_theta_matrix.dot(images_matrix.T)
        self.weights = self.weights - learning_rate * (derivative_weights / len(layer_cache))
        self.bias = self.bias - learning_rate * (derived_biases / len(layer_cache))

    def relu(self, image):
        return np.maximum(image, 0)
