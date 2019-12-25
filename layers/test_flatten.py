from unittest import TestCase

import numpy as np

from layers.flatten import Flatten


class TestFlatten(TestCase):

    def setUp(self) -> None:
        self.image = np.arange(1, 13).reshape((3, 2, 2))
        self.flatten = np.array([np.array([i]) for i in range(1, 13)])

    def test_forward(self):
        layer = Flatten()
        flatten = layer.forward(self.image)
        np.testing.assert_array_equal(flatten, self.flatten)

    def test_back(self):
        layer = Flatten()
        image_theta = layer.back(self.flatten)
        np.testing.assert_array_equal(image_theta, self.image)
