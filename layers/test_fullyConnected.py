from unittest import TestCase

import numpy as np

from layers.fully_connected import FullyConnected


class TestFullyConnected(TestCase):
    def test_forward(self):
        subject = FullyConnected(3, 2)
        subject.weights = np.array(([2, 1, 3], [0, 1, -1]))
        subject.bias = np.array([0.5, 0.5]).T
        res = subject.forward(np.array([2, 3, 4]).T)
        exp = np.array([4 + 3 + 12 + 0.5, 3 - 4 + 0.5]).T
        np.testing.assert_array_equal(res, exp)
