from unittest import TestCase

import numpy as np

from plain.layers.cross_entropy import CrossEntropy


class TestCrossEntropy(TestCase):

    def test_loss(self):
        loss = CrossEntropy()
        res = loss.loss(np.array([0.1, 0.2, 0.3]).reshape((3, 1)),
                        np.array([0, 1, 0]).reshape((3, 1)))
        self.assertEqual(np.log(0.2), res)
