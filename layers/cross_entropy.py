import numpy as np


class CrossEntropy:

    def loss(self, probs, expected):
        return np.multiply(np.log(probs), expected)

    def delta(self, probs: np.array, expected: np.array) -> np.array:
        return probs - expected
