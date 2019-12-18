from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from common.batch_engine import BatchEngine
from layers.conv2d import Conv2d
from layers.cross_entropy import CrossEntropy


class TestBatchEngine(TestCase):

    def test_run(self):
        layers = []
        for i in range(1, 4):
            layer = Conv2d(1, 1, 1)
            layer.forward = self.mock_method([
                np.array([i, i]),
                np.array([i * 10, i * 10])
            ])
            layer.back = self.mock_method([
                np.array([-i, -i]),
                np.array([-i * 10, -i * 10])
            ])
            layers.append(layer)

        loss_function = CrossEntropy()
        loss_function.loss = self.mock_method([0.1, 0.2])
        loss_function.delta = self.mock_method([
            [0.2, -0.3],
            [0.5, -0.6]
        ])
        engine = BatchEngine(layers, loss_function)

        images = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        labels = np.array([
            [1, 0, 0],
            [0, 0, 1],
        ])
        engine.run(images, labels)

        cache_exp = np.array([
            [
                ([1, 2, 3], [-1, -1]),
                ([4, 5, 6], [-10, -10])
            ],
            [
                ([1, 1], [-2, -2]),
                ([10, 10], [-20, -20])
            ],
            [
                ([2, 2], [-3, -3]),
                ([20, 20], [-30, -30])
            ]
        ])
        np.testing.assert_array_equal(engine.cache, cache_exp)

    def mock_method(self, results):
        magic_mock = MagicMock()
        magic_mock.side_effects = results
        return magic_mock