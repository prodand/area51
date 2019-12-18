import numpy as np

from layers.base_layer import BaseLayer


class Conv2d(BaseLayer):
    def __init__(self, features, kernel_size, channels, values=None):
        self.features = features
        self.kernel_size = kernel_size
        self.channels = channels
        if values is None:
            self.kernel = np.random.rand(features, channels, kernel_size, kernel_size)
        else:
            self.kernel = np.array(values, dtype=np.float64) \
                .reshape((features, channels, kernel_size, kernel_size))
        self.cached_image = np.array([])

    def forward(self, image):
        if image.ndim != 3:
            raise RuntimeError()
        self.cached_image = image.astype(np.float64)
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
                    )
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

    # TODO: make it to return just derivative for weights
    def update_weights(self, activation_theta, learning_rate):
        image_dims = self.cached_image.shape
        theta_size = activation_theta.shape[1]
        width = image_dims[1] - theta_size + 1
        height = image_dims[2] - theta_size + 1
        result = np.zeros(self.kernel.shape)
        for f in range(0, self.features):
            for c in range(0, len(self.cached_image)):
                for i in range(0, width):
                    for j in range(0, height):
                        result[f, c, i, j] = np.sum(
                            np.multiply(
                                self.cached_image[c, i:i + theta_size, j:j + theta_size],
                                activation_theta[f]
                            )
                        )
        return self.kernel - learning_rate * result
