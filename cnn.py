import numpy as np

from common.batch_engine import BatchEngine
from layers.cross_entropy import CrossEntropy


class Cnn:

    loss_function = CrossEntropy()

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def learn(self, images, labels):
        batch_engine = BatchEngine(self.layers, self.loss_function)
        batch_engine.run(images, labels)

    def predict(self, image):
        for layer in self.layers:
            image = layer.forward(image)

        return np.argmax(image)