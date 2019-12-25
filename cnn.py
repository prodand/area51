import numpy as np

from common.batch_engine import BatchEngine
from layers.cross_entropy import CrossEntropy


class Cnn:

    loss_function = CrossEntropy()

    def __init__(self, batch_size):
        self.layers = []
        self.batch_size = batch_size

    def add_layer(self, layer):
        self.layers.append(layer)

    # images as (image_index x channels x width x heights)
    def learn(self, images, labels):
        batch_engine = BatchEngine(self.layers, self.loss_function, self.batch_size)
        batch_engine.run(images, labels)

    def predict(self, image):
        for layer in self.layers:
            image = layer.forward(image)

        return np.argmax(image)