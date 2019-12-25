from unittest import TestCase

import numpy as np

from layers.softmax import Softmax


class TestSoftmax(TestCase):

    def test_forward(self):
        input = np.array([1, 2, -1, 3])
        layer = Softmax()
        res = layer.forward(input)
        print(np.exp(74600.0))
        sum = np.exp(-2) + np.exp(-1) + np.exp(-4) + np.exp(0)
        exp = [np.exp(-2) / sum, np.exp(-1) / sum, np.exp(-4) / sum, np.exp(0) / sum]
        np.testing.assert_allclose(exp, res)

    def test_back(self):
        pass