import numpy as np

from vectorized.layers.base_layer import BaseLayer


class Conv2d(BaseLayer):
    def __init__(self, features, kernel_size, channels, values=None, bias=None):
        self.features = features
        self.kernel_size = kernel_size
        self.channels = channels
        if values is None:
            self.kernel = np.random.randn(features, channels, kernel_size, kernel_size) * 0.1
        else:
            self.kernel = np.array(values, dtype=np.float64) \
                .reshape((features, channels, kernel_size, kernel_size))
        if bias is None:
            self.bias = np.random.randn(features, 1)  * 0.1
        else:
            self.bias = np.array(bias, dtype=np.float64).reshape((features, 1))

    def forward(self, image):
        if image.ndim != 3:
            raise RuntimeError()
        return self.relu(self.convolve(image))

    def convolve(self, image):
        dims = image.shape
        result_width = dims[1] - self.kernel_size + 1
        result_height = dims[2] - self.kernel_size + 1
        result = np.zeros((self.features, result_width, result_height))
        for c in range(self.features):
            for i in range(0, result_width):
                for j in range(0, result_height):
                    result[c, i, j] = np.sum(
                        np.multiply(
                            image[0:, i:i + self.kernel_size, j:j + self.kernel_size],
                            self.kernel[c]
                        )
                    ) + self.bias[c]
        return result

    def relu(self, image):
        return np.maximum(image, 0)

    def back(self, activation_theta):
        if activation_theta.ndim != 3:
            raise RuntimeError('Activation must be 3 dimensional')
        if activation_theta.shape[0] != self.features:
            raise RuntimeError('Activation channels not equal to feature numbers')

        return self.calculate_prev_layer_error(activation_theta)

    def calculate_prev_layer_error(self, activation_theta):
        kernel_size = self.kernel_size
        result_size = activation_theta.shape[1] + kernel_size - 1
        result = np.zeros((self.channels, result_size, result_size))
        padded_theta = np.pad(activation_theta,
                              ((0, 0),
                               (kernel_size - 1, kernel_size - 1),
                               (kernel_size - 1, kernel_size - 1)),
                              'constant', constant_values=0)
        flipped = np.flip(self.kernel, (2, 3))
        for f in range(0, self.features):
            for c in range(0, self.channels):
                for i in range(0, padded_theta.shape[1] - kernel_size + 1):
                    for j in range(0, padded_theta.shape[2] - kernel_size + 1):
                        result[c, i, j] += np.sum(
                            np.multiply(
                                padded_theta[f, i:i + kernel_size, j:j + kernel_size],
                                flipped[f, c]
                            )
                        )
        return result

    def update_weights(self, layer_cache, learning_rate):
        average_kernel_weights = self.calculate_average_weights_derivative(layer_cache)
        average_biases = self.calculate_average_biases(layer_cache)
        self.kernel -= learning_rate * average_kernel_weights
        self.bias -= learning_rate * average_biases

    def calculate_average_weights_derivative(self, layer_cache):
        image_dims = layer_cache[0][0].shape
        theta_size = layer_cache[0][1].shape[1]
        width = image_dims[1] - theta_size + 1
        height = image_dims[2] - theta_size + 1
        result = np.zeros(self.kernel.shape)
        for (image, activation_theta) in layer_cache:
            for f in range(0, self.features):
                for c in range(0, len(image)):
                    for i in range(0, width):
                        for j in range(0, height):
                            result[f, c, i, j] += np.sum(
                                np.multiply(
                                    image[c, i:i + theta_size, j:j + theta_size],
                                    activation_theta[f]
                                )
                            )
        return result / len(layer_cache)

    def calculate_average_biases(self, layer_cache):
        biases_sum = np.zeros(self.bias.shape)
        for (image, activation_theta) in layer_cache:
            for f in range(0, self.features):
                biases_sum[f] += np.sum(activation_theta[f])

        return biases_sum / len(layer_cache)

    def __str__(self) -> str:
        return '%d x %d x %d' % (self.features, self.channels, self.kernel_size)
