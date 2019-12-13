import numpy as np


class MaxPooling:
    def __init__(self, input_shape, size, stride):
        self.input_shape = input_shape
        self.size = size
        self.stride = stride
        self.mask = np.zeros(input_shape)

    def forward(self, image):
        if image.ndim != 3:
            raise RuntimeError('Image must be 3 dimensional')
        self.mask = np.zeros(image.shape)
        dims = image.shape
        result_width = int((dims[1] - self.size) / self.stride + 1)
        result_height = int((dims[2] - self.size) / self.stride + 1)
        result = np.zeros((dims[0], result_width, result_height))
        for c in range(0, dims[0]):
            for i in range(0, result_width):
                for j in range(result_height):
                    start_w = i * self.stride
                    start_h = j * self.stride
                    segment = image[c, start_w:start_w + self.size, start_h:start_h + self.size]
                    indexes = np.unravel_index(np.argmax(segment, axis=None), segment.shape)
                    self.mask[c, start_w + indexes[1], start_h + indexes[2]] = 1
                    result[c, i, j] = segment[indexes]
        return result


    def back(self, activation_theta):
        dims = activation_theta.shape
        result = np.zeros(self.mask.shape)
        for c in range(0, dims[0]):
            for i in range(0, dims[1]):
                for j in range(dims[2]):
                    start_w = i * self.stride
                    start_h = j * self.stride
                    segment = self.mask[c, start_w:start_w + self.size, start_h:start_h + self.size]
                    indexes = np.unravel_index(np.argmax(segment, axis=None), segment.shape)
                    result[c, start_w + indexes[1], start_h + indexes[2]] = activation_theta[c, i, j]
                    result[c, i, j] = segment[indexes]
        return result