import numpy as np


class MaxPooling:
    def __init__(self, size):
        self.size = size

    def forward(self, image):
        dims = image.shape
        result_width = dims[0] - self.size + 1
        result_height = dims[1] - self.size + 1
        result = np.zeros((result_width, result_height))
        for i in range(0, result_width):
            for j in range(0, result_height):
                result[i, j] = np.amax(image[i:i + self.size, j:j + self.size])
        return result
