import numpy as np

class Softmax:

    def forward(self, input):
        exp = np.exp(input)
        return exp / np.sum(exp)

    def back(self, theta):
        return []