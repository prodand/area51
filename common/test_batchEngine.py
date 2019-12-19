from unittest import TestCase
from unittest.mock import MagicMock, Mock

import numpy as np

from common.batch_engine import BatchEngine
from layers.conv2d import Conv2d
from layers.cross_entropy import CrossEntropy


class TestBatchEngine(TestCase):

    def setUp(self) -> None:
        layers = []
        for i in range(1, 4):
            layer = Conv2d(1, 1, 1)
            layer.forward = self.mock_method([
                np.array([i, i, i]),
                np.array([i * 10, i * 10, i * 10])
            ])
            layer.back = self.mock_method([
                np.array([-i, -i, -i]),
                np.array([-i * 10, -i * 10, -i * 10])
            ])
            layer.update_weights = MagicMock()
            layers.append(layer)

        self.loss_function = CrossEntropy()
        self.loss_function.loss = self.mock_method([0.1, 0.2])
        self.loss_function.delta = self.mock_method([
            [0.2, -0.3, 0.1],
            [0.5, -0.6, 0.1]
        ])
        self.images = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        self.labels = np.array([
            [1, 0, 0],
            [0, 0, 1],
        ])
        self.layers = layers

    def test_run_cache_check(self):
        engine = BatchEngine(self.layers, self.loss_function)
        engine.run(self.images, self.labels)

        cache_exp = list([
            list([
                (np.array([1, 2, 3]), np.array([-2, -2, -2])),
                (np.array([4, 5, 6]), np.array([-20, -20, -20]))
            ]),
            list([
                (np.array([1, 1]), np.array([-3, -3, -3])),
                (np.array([10, 10]), np.array([-30, -30, -30]))
            ]),
            list([
                (np.array([2, 2, 2]), np.array([0.2, -0.3, 0.1])),
                (np.array([20, 20]), np.array([0.5, -0.6, 0.1]))
            ])
        ])
        np.testing.assert_array_equal(cache_exp[0][0], engine.cache[0][0])

    def test_run_cache_forward_check(self):
        engine = BatchEngine(self.layers, self.loss_function)
        engine.run(self.images, self.labels)

        np.testing.assert_array_equal(np.array([1, 2, 3]),
                                      self.get_forward_arg(self.layers[0], 0))
        np.testing.assert_array_equal(np.array([4, 5, 6]),
                                      self.get_forward_arg(self.layers[0], 1))
        np.testing.assert_array_equal(np.array([1, 1, 1]),
                                      self.get_forward_arg(self.layers[1], 0))
        np.testing.assert_array_equal(np.array([10, 10, 10]),
                                      self.get_forward_arg(self.layers[1], 1))
        np.testing.assert_array_equal(np.array([2, 2, 2]),
                                      self.get_forward_arg(self.layers[2], 0))
        np.testing.assert_array_equal(np.array([20, 20, 20]),
                                      self.get_forward_arg(self.layers[2], 1))

    def test_run_cache_back_check(self):
        engine = BatchEngine(self.layers, self.loss_function)
        engine.run(self.images, self.labels)

        np.testing.assert_array_equal(np.array([0.2, -0.3, 0.1]),
                                      self.get_back_arg(self.layers[2], 0))
        np.testing.assert_array_equal(np.array([0.5, -0.6, 0.1]),
                                      self.get_back_arg(self.layers[2], 1))
        np.testing.assert_array_equal(np.array([-3, -3, -3]),
                                      self.get_back_arg(self.layers[1], 0))
        np.testing.assert_array_equal(np.array([-30, -30, -30]),
                                      self.get_back_arg(self.layers[1], 1))
        self.assertFalse(self.layers[0].back.called)

    # TODO: test loss function

    def test_run_cache_update_weights_check(self):
        engine = BatchEngine(self.layers, self.loss_function)
        engine.run(self.images, self.labels)

        np.testing.assert_array_equal(list([
                (np.array([1, 2, 3]), np.array([-2, -2, -2])),
                (np.array([4, 5, 6]), np.array([-20, -20, -20]))
            ]),
            self.get_update_weights_arg(self.layers[0])
        )
        np.testing.assert_array_equal(
            [
                (np.array([1, 1, 1]), np.array([-3, -3, -3])),
                (np.array([10, 10, 10]), np.array([-30, -30, -30]))
            ],
            self.get_update_weights_arg(self.layers[1])
        )
        np.testing.assert_array_equal(
            [
                (np.array([2, 2, 2]), np.array([0.2, -0.3, 0.1])),
                (np.array([20, 20, 20]), np.array([0.5, -0.6, 0.1]))
            ],
            self.get_update_weights_arg(self.layers[2])
        )

    def mock_method(self, results):
        return MagicMock(side_effect=results)

    @staticmethod
    def get_forward_arg(layer, index):
        args, kwargs = layer.forward.call_args_list[index]
        return args[0]

    @staticmethod
    def get_back_arg(layer, index):
        args, kwargs = layer.back.call_args_list[index]
        return args[0]

    @staticmethod
    def get_update_weights_arg(layer):
        args, kwargs = layer.update_weights.call_args
        return args[0]
