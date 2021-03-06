from unittest import TestCase

import numpy as np

from vectorized.layers.conv2d import Conv2d


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
                       [1, -1, 1, 2, 1, 0, 0, 2], [2, 3])
        image = np.arange(1, 10).reshape((1, 3, 3))
        res = layer.convolve(image)
        exp = np.array(([
            [[15., 18.], [24., 27.]],
            [[14., 17.], [23., 26.]]
        ]))
        np.testing.assert_array_equal(res, exp)

    def test_convolve_multi_channels(self):
        layer = Conv2d(3, 1, 2,
                       [0.5, 0.5, 2, 2, 1.5, 1.5], [1, 1, 1])
        image = np.arange(1, 9).reshape((2, 2, 2))
        res = layer.convolve(image)
        exp = np.array(([
            [[3 + 1., 4. + 1], [5. + 1, 6. + 1]],
            [[12. + 1, 16. + 1], [20. + 1, 24. + 1]],
            [[9. + 1, 12. + 1], [15. + 1, 18. + 1]]
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

    def test_calculate_average_weights_derivative(self):
        layer = Conv2d(1, 2, 1, [1, -1, 1, 2])
        image = np.array(([1, 2, 1], [2, 3, 1], [2, 1, 1])).reshape((1, 3, 3))

        theta = np.array(([2, 1, 1, 0])).reshape((1, 2, 2)).astype(np.float64)
        cache_values = [
            (image, theta),
        ]
        res = layer.calculate_average_weights_derivative(cache_values)
        exp = np.array(([[6., 8.], [9., 8.]])).reshape((1, 1, 2, 2))
        np.testing.assert_array_equal(res, exp)

    def test_calculate_average_weights_derivative_multichannel(self):
        layer = Conv2d(1, 2, 2, [
            1, 1, 1, 1,
            1, 1, 1, 1
        ])
        image = np.array((
            1, 2, 1, 2, 3, 1, 2, 1, 1,
            1, 2, 1, 0, 2, 2, 1, 1, 1
        )).reshape((2, 3, 3))

        theta = np.array(([1, 0], [1, 0])).reshape((1, 2, 2))

        cache_values = [
            (image, theta),
        ]
        res = layer.calculate_average_weights_derivative(cache_values)
        exp = np.array([
            3., 5., 4., 4.,
            1., 4., 1., 3.
        ]).reshape((1, 2, 2, 2))
        np.testing.assert_allclose(res, exp)

    def test_calculate_average_weights_derivative_multiple_features(self):
        layer = Conv2d(2, 2, 1, [
            1, 1, 1, 1,
            1, 1, 1, 1
        ])
        image = np.array((
            1, 2, 1, 2, 3, 1, 2, 1, 1
        )).reshape((1, 3, 3))

        theta = np.array(([
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ])).reshape((2, 2, 2))

        cache_values = [
            (image, theta)
        ]
        res = layer.calculate_average_weights_derivative(cache_values)
        exp = np.array([
            3., 5., 4., 4.,
            5., 2., 4., 2.
        ]).reshape((2, 1, 2, 2))
        np.testing.assert_allclose(res, exp)

    def test_calculate_average_weights_derivative_multiple_features_batch(self):
        layer = Conv2d(2, 2, 1, [
            1, 1, 1, 1,
            1, 1, 1, 1
        ])
        image1 = np.array((
            1, 2, 1, 2, 3, 1, 2, 1, 1
        )).reshape((1, 3, 3))
        image2 = np.array((
            2, 1, 1, 2, 1, 1, 2, 2, 1
        )).reshape((1, 3, 3))

        theta1 = np.array(([
            [[1, 0], [1, 0]],
            [[0, 1], [0, 1]],
        ])).reshape((2, 2, 2))

        theta2 = np.array(([
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]],
        ])).reshape((2, 2, 2))

        cache_values = [
            (image1, theta1),
            (image2, theta2)
        ]
        res = layer.calculate_average_weights_derivative(cache_values)
        exp = np.array([
            2.5, 3.5, 3.5, 3.,
            4.5, 2., 4., 2.5
        ]).reshape((2, 1, 2, 2))
        np.testing.assert_allclose(res, exp)

    def test_calculate_average_biases(self):
        layer = Conv2d(2, 2, 1, [
            1, 1, 1, 1,
            1, 1, 1, 1
        ])
        image1 = np.array((
            1, 2, 1, 2, 3, 1, 2, 1, 1
        )).reshape((1, 3, 3))
        image2 = np.array((
            2, 1, 1, 2, 1, 1, 2, 2, 1
        )).reshape((1, 3, 3))

        theta1 = np.array(([
            [[1, 2], [1, 3]],
            [[-1, 4], [2, 1]],
        ])).reshape((2, 2, 2))

        theta2 = np.array(([
            [[2, 1], [5, 1]],
            [[1, 2], [5, 1]],
        ])).reshape((2, 2, 2))

        cache_values = [
            (image1, theta1),
            (image2, theta2)
        ]
        res = layer.calculate_average_biases(cache_values)
        exp_bias = np.array([8.0, 7.5]).reshape((2, 1))
        np.testing.assert_allclose(res, exp_bias)

    def test_img2vec(self):
        layer = Conv2d(1, 2, 2)
        image = np.arange(1, 33).reshape((2, 4, 4))
        res = layer.img2vec(image)
