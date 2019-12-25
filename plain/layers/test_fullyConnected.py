from unittest import TestCase

import numpy as np

from plain.layers.fully_connected import FullyConnected


class TestFullyConnected(TestCase):

    def test_forward(self):
        subject = FullyConnected(3, 2)
        subject.weights = np.array(([2, 1, 3], [0, 1, -1]))
        subject.bias = np.array([0.5, 0.5]).T
        res = subject.forward(np.array([2, 3, 4]).T)
        exp = np.array([4 + 3 + 12 + 0.5, 3 - 4 + 0.5]).T
        np.testing.assert_array_equal(res, exp)

    def test_back(self):
        subject = FullyConnected(3, 2)
        subject.weights = np.array(([2, 1, 3], [0, 1, -1]))
        subject.bias = np.array([0.5, 0.5]).T
        res = subject.back(np.array([2, 1]))
        exp = np.array([4, 3, 5])
        np.testing.assert_array_equal(res, exp)

    def test_update_weights(self):
        subject = FullyConnected(3, 2)
        subject.weights = np.array(([2, 1, 0], [0, 1, 1]))
        subject.bias = np.array([[0.5, 0.5]]).T

        subject.update_weights(list([
            (np.array([[2, 1, 4]]).reshape((3, 1)), np.array([[2, 1]]).reshape((2, 1))),
            (np.array([[1, 3, 1]]).reshape((3, 1)), np.array([[3, -1]]).reshape((2, 1)))
        ]), 0.1)
        exp_weights = np.array((
            [2 - 0.1 * 3.5, 1 - 0.1 * 5.5, -0.1 * 5.5],
            [-0.1 * 0.5, 1 - 0.1 * -1, 1 - 0.1 * 1.5])
        )
        exp_bias = np.array([[0.5 - 0.1 * 2.5, 0.5]]).T
        np.testing.assert_array_equal(subject.weights, exp_weights)
        np.testing.assert_array_equal(subject.bias, exp_bias)
