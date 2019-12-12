from unittest import TestCase

import numpy as np

from layers.maxPooling import MaxPooling


class TestMaxPool(TestCase):

    def test_execute(self):
        image = np.array(([1, 4, 7], [2, 5, 8], [3, 6, 9]))
        layer = MaxPooling(2)
        res = layer.execute(image)
        np.testing.assert_array_equal(res, np.array(([5, 8], [6, 9])))
