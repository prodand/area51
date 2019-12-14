from unittest import TestCase

import numpy as np

from layers.maxPooling import MaxPooling


class TestMaxPool(TestCase):

    def test_forward(self):
        image = np.array(([1, 4, 7, 5,
                           2, 5, 3, 8,
                           -9, 1, 1, 1,
                           8, 3, 1, 1])).reshape((1, 4, 4))
        layer = MaxPooling((1, 4, 4), 2, 2)
        res = layer.forward(image)
        exp = np.array([5, 8, 8, 1]).reshape((1, 2, 2))
        np.testing.assert_array_equal(res, exp)

    def test_back(self):
        image = np.array(([1, 4, 7, 5,
                           2, 5, 3, 8,
                           -9, 1, 1, 1,
                           8, 3, 1, 1])).reshape((1, 4, 4))
        theta = np.array([5, 8, 8, 1]).reshape((1, 2, 2))

        layer = MaxPooling((1, 4, 4), 2, 2)
        layer.forward(image)

        res = layer.back(theta)
        exp = np.array(([
            0, 0, 0, 0,
            0, 5, 0, 8,
            0, 0, 1, 0,
            8, 0, 0, 0])).reshape((1, 4, 4))
        np.testing.assert_allclose(exp, res)
