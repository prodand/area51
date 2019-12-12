import numpy as np


class Cnn:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)
