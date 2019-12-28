import numpy as np

from plain.layers.base_layer import BaseLayer


class Conv2d(BaseLayer):
    def __init__(self, features, kernel_size, channels, values=None, bias=None):
        self.features = features
        self.kernel_size = kernel_size
        self.channels = channels
        if values is None:
            self.kernel = np.random.normal(0.0, 0.5, size=(features, channels, kernel_size, kernel_size)) * 0.1
        else:
            self.kernel = np.array(values, dtype=np.float64) \
                .reshape((features, channels, kernel_size, kernel_size))
        if bias is None:
            self.bias = np.random.normal(0.0, 0.5, size=(features, 1)) * 0.1
        else:
            self.bias = np.array(bias, dtype=np.float64).reshape((features, 1))

    def forward(self, image):
        if image.ndim != 3:
            raise RuntimeError()
        return self.relu(self.convolve(image))

    def convolve(self, image):
        img_vec = self.img2vec(image)
        kernel_vec_size = self.kernel_size * self.kernel_size * self.channels
        kernel_vec = self.kernel.reshape((self.features, kernel_vec_size))
        result_width = image.shape[1] - self.kernel_size + 1
        result = kernel_vec.dot(img_vec) + self.bias
        return result.reshape((self.features, result_width, result_width))

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
        padded_theta = np.pad(activation_theta,
                              ((0, 0),
                               (kernel_size - 1, kernel_size - 1),
                               (kernel_size - 1, kernel_size - 1)),
                              'constant', constant_values=0)
        flipped = np.flip(self.kernel, (2, 3))
        kernel_vec_size = self.kernel_size * self.kernel_size
        padded_theta_vec = self.matrix2vec(padded_theta, self.kernel_size)
        flipped_kernel_vec = flipped.reshape((self.features, self.channels, kernel_vec_size))
        result = np.matmul(flipped_kernel_vec, padded_theta_vec)
        return np.sum(result, axis=0).reshape((self.channels, result_size, result_size))

    def update_weights(self, layer_cache, learning_rate):
        average_kernel_weights = self.calculate_average_weights_derivative(layer_cache)
        average_biases = self.calculate_average_biases(layer_cache)
        self.kernel -= learning_rate * average_kernel_weights
        self.bias -= learning_rate * average_biases

    def calculate_average_weights_derivative(self, layer_cache):
        theta_size = layer_cache[0][1].shape[1]
        result = np.zeros((self.features, self.channels, 1, self.kernel_size * self.kernel_size))
        for (image, activation_theta) in layer_cache:
            image_2d_vectorized = self.matrix2vec(image, theta_size)
            activation_theta_vec = activation_theta.reshape((self.features, 1, 1, theta_size * theta_size))
            result += np.matmul(activation_theta_vec, image_2d_vectorized)
        return result.reshape((self.features, self.channels, self.kernel_size, self.kernel_size)) / len(layer_cache)

    def calculate_average_biases(self, layer_cache):
        biases_sum = np.zeros(self.bias.shape)
        for (image, activation_theta) in layer_cache:
            for f in range(0, self.features):
                biases_sum[f] += np.sum(activation_theta[f])

        return biases_sum / len(layer_cache)

    def img2vec(self, image):
        dims = image.shape
        result_width = dims[1] - self.kernel_size + 1
        result_height = dims[2] - self.kernel_size + 1
        vec_rows_number = self.kernel_size * self.kernel_size * dims[0]
        vector = np.zeros((vec_rows_number, result_height * result_height))
        vector_col = 0
        for i in range(0, result_width):
            for j in range(0, result_height):
                col = image[:, i:i + self.kernel_size, j:j + self.kernel_size].reshape((1, vec_rows_number))
                vector[:, vector_col] = col
                vector_col += 1
        return vector

    def matrix2vec(self, theta, kernel_size: int):
        dims = theta.shape
        result_width = dims[1] - kernel_size + 1
        result_height = dims[2] - kernel_size + 1
        vec_rows_number = kernel_size * kernel_size
        vector = np.zeros((dims[0], kernel_size * kernel_size,
                           result_height * result_height))
        for f in range(0, len(theta)):
            vector_col = 0
            for i in range(0, result_width):
                for j in range(0, result_height):
                    col = theta[f, i:i + kernel_size, j:j + kernel_size].reshape((1, vec_rows_number))
                    vector[f, :, vector_col] = col
                    vector_col += 1
        return vector

    def __str__(self) -> str:
        return '%d x %d x %d' % (self.features, self.channels, self.kernel_size)

    def save(self, folder: str):
        file_name = "/conv_%s_%s_%s_%s" % (self.features, self.channels, self.kernel_size, self.kernel_size)
        np.save(folder + file_name, self.kernel)
        bias_file = "/conv_bias_%s_%s_%s_%s" % (self.features, self.channels, self.kernel_size, self.kernel_size)
        np.save(folder + bias_file, self.bias)
        return "%s,%s" % (file_name, bias_file)

    @staticmethod
    def load(folder, files):
        parts = files.split(",")
        kernels = np.load(folder + parts[0] + ".npy")
        bias = np.load(folder + parts[1] + ".npy")
        return Conv2d(kernels.shape[0], kernels.shape[2],
                      kernels.shape[1], kernels, bias)