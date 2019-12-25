import numpy as np

from plain.layers.base_layer import BaseLayer


class MaxPooling(BaseLayer):
    def __init__(self, input_shape: tuple, size, stride):
        self.input_shape = input_shape
        self.size = size
        self.stride = stride
        self.mask = np.zeros(input_shape)

    def forward(self, image):
        if image.ndim != 3:
            raise RuntimeError('Image must be 3 dimensional')
        if image.shape != self.input_shape:
            raise RuntimeError('Image must match configured input shape')
        self.mask = np.zeros(image.shape)
        dims = image.shape
        result_width = int((dims[1] - self.size) / self.stride + 1)
        result_height = int((dims[2] - self.size) / self.stride + 1)
        result = np.zeros((dims[0], result_width, result_height))
        self.mask = np.zeros((result.shape[0], result.shape[1], result.shape[2], 2), dtype=int)
        for c in range(0, dims[0]):
            for i in range(0, result_width):
                for j in range(result_height):
                    start_w = i * self.stride
                    start_h = j * self.stride
                    segment = image[c, start_w:start_w + self.size, start_h:start_h + self.size]
                    indexes = np.unravel_index(np.argmax(segment, axis=None), segment.shape)
                    self.mask[c, i, j] = (start_w + indexes[0], start_h + indexes[1])
                    result[c, i, j] = segment[indexes]
        return result

    def back(self, activation_theta):
        dims = activation_theta.shape
        result = np.zeros(self.input_shape)
        for c in range(0, dims[0]):
            for i in range(0, dims[1]):
                for j in range(dims[2]):
                    indexes = self.mask[c, i, j]
                    result[c, indexes[0], indexes[1]] = activation_theta[c, i, j]
        return result

    def update_weights(self, layer_cache, learning_rate):
        pass

    def __str__(self) -> str:
        return '%d x %d x %d' % self.input_shape
