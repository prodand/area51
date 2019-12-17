import numpy as np


class Softmax:

    def forward(self, input):
        input_shift = input - np.max(input)
        exp = np.exp(input_shift)
        return exp / np.sum(exp)

    def back(self, theta):
        return theta
