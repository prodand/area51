from unittest import TestCase

import numpy as np

from cnn import Cnn


class TestCnn(TestCase):

    def setUp(self) -> None:
        self.subject = Cnn()
