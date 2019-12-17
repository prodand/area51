from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from layers.conv2d import Conv2d


class TestBatchEngine(TestCase):

    def test_run(self):
        layer1 = Conv2d(1, 1, 1)
        layer1.forward = MagicMock(return_value=np.array([1]))
        layer1.forward(np.array([1]))
        layer1.forward.assert_called_once_with(np.array([1]))
        pass
