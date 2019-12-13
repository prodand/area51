from unittest import TestCase

import numpy as np

from layers.conv2d import Conv2d


class TestConv2d(TestCase):

    def test_init(self):
        test = Conv2d(2, 3, 5)
        self.assertEqual(test.kernel.shape, (2, 5, 3, 3))

    def test_relu(self):
        layer = Conv2d(1, 2, 1, [1, -1, 1, 2])
        input = np.array(([-1, 3], [2, -1]))
        exp = np.array(([0, 3], [2, 0]))
        np.testing.assert_array_equal(layer.relu(input), exp)

    def test_convolve(self):
        layer = Conv2d(2, 2, 1,
                       [1, -1, 1, 2, 1, 0, 0, 2])
        image = np.arange(1, 10).reshape((1, 3, 3))
        print(image)
        print(layer.kernel)
        res = layer.convolve(image)
        exp = np.array(([
            [[13., 16.], [22., 25.]],
            [[11., 14.], [20., 23.]]
        ]))
        np.testing.assert_array_equal(res, exp)

    def test_convolve_multi_channels(self):
        layer = Conv2d(3, 1, 2,
                       [0.5, 0.5, 2, 2, 1.5, 1.5])
        image = np.arange(1, 9).reshape((2, 2, 2))
        res = layer.convolve(image)
        exp = np.array(([
            [[3., 4.], [5., 6.]],
            [[12., 16.], [20., 24.]],
            [[9., 12.], [15., 18.]]
        ]))
        np.testing.assert_array_equal(res, exp)

    def test_calculate_prev_layer_error(self):
        layer = Conv2d(1, 2, 1, [1, -1, 1, 2])
        theta = np.array((0.5, 0.3, 0.2, 0.7)).reshape((1, 2, 2))
        res = layer.calculate_prev_layer_error(theta)
        exp = np.array([0.5, -0.2, -0.3, 0.7, 1.8, -0.1, 0.2, 1.1, 1.4]).reshape((1, 3, 3))
        np.testing.assert_allclose(res, exp)

    def test_calculate_prev_layer_error_multi_feature(self):
        layer = Conv2d(2, 2, 1, [1, -1, 1, 2, 1, 2, 1, 1])

        theta = np.array((1., 0., 1., 0., 2., 0., 2., 0.)).reshape((2, 2, 2))
        res = layer.calculate_prev_layer_error(theta)
        # 1 -1 0 , 2 1 0, 1 2 0
        # 2  4 0 , 4 6 0, 2 2 0
        exp = np.array([3, 3, 0, 6, 7, 0, 3, 4, 0]).reshape((1, 3, 3))
        np.testing.assert_allclose(res, exp)

    def test_calculate_prev_layer_error_multi_feature_multi_channel(self):
        layer = Conv2d(2, 2, 2, [
            1, -1, 1, 2, 1, 2, 1, 1,
            1, -1, 1, 1, 2, -2, 1, 1
        ])

        theta = np.array((1., 0., 1., 0., 2., 0., 2., 0.)).reshape((2, 2, 2))
        res = layer.calculate_prev_layer_error(theta)
        exp = np.array([
            3, -3, 0, 6, 1, 0, 3, 4, 0,
            5, -2, 0, 8, 1, 0, 3, 3, 0
        ]).reshape((2, 3, 3))
        np.testing.assert_allclose(res, exp)

    def test_update_weights(self):
        layer = Conv2d(1, 2, 1, [1, -1, 1, 2])
        image = np.array(([1, 2, 1], [2, 3, 1], [2, 1, 1])).reshape((1, 3, 3))
        layer.forward(image)

        theta = np.array(([2, 1, 1, 0])).reshape((1, 2, 2)).astype(np.float64)
        res = layer.update_weights(theta, 0.1)
        exp = np.array(([[1.0 - 6. * 0.1, -1.8], [1.0 - 0.9, 1.2]])).reshape((1, 1, 2, 2))
        np.testing.assert_array_equal(res, exp)

    def test_update_weights_multichannel(self):
        layer = Conv2d(1, 2, 2, [
            1, 1, 1, 1,
            1, 1, 1, 1
        ])
        image = np.array((
            1, 2, 1, 2, 3, 1, 2, 1, 1,
            1, 2, 1, 0, 2, 2, 1, 1, 1
        )).reshape((2, 3, 3))
        layer.forward(image)

        theta = np.array(([1, 0], [1, 0])).reshape((1, 2, 2))
        res = layer.update_weights(theta, 0.1)
        exp = np.array([
            1 - 0.3, 1 - 0.5, 1 - 0.4, 1 - 0.4,
            1 - 0.1, 1 - 0.4, 1 - 0.1, 1 - 0.3
        ]).reshape((1, 2, 2, 2))
        np.testing.assert_allclose(res, exp)

    def test_update_weights_multiple_features(self):
        layer = Conv2d(2, 2, 1, [
            1, 1, 1, 1,
            1, 1, 1, 1
        ])
        image = np.array((
            1, 2, 1, 2, 3, 1, 2, 1, 1
        )).reshape((1, 3, 3))
        layer.forward(image)

        theta = np.array(([
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ])).reshape((2, 2, 2))
        res = layer.update_weights(theta, 0.1)
        exp = np.array([
            1 - 0.3, 1 - 0.5, 1 - 0.4, 1 - 0.4,
            1 - 0.5, 1 - 0.2, 1 - 0.4, 1 - 0.2
        ]).reshape((2, 1, 2, 2))
        np.testing.assert_allclose(res, exp)
