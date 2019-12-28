import os
from datetime import datetime

import numpy as np

from plain.common.batch_engine import BatchEngine
from plain.layers.cross_entropy import CrossEntropy
from plain.layers.flatten import Flatten
from plain.layers.maxPooling import MaxPooling
from plain.layers.softmax import Softmax
from vectorized.layers.conv2d import Conv2d
from vectorized.layers.fully_connected import FullyConnected


class Cnn:
    loss_function = CrossEntropy()

    def __init__(self, batch_size):
        self.layers = []
        self.batch_size = batch_size

    def add_layer(self, layer):
        self.layers.append(layer)

    def learn(self, images, labels):
        """
        :param images: data as (image_index x channels x width x heights)
        :param labels: expected result in vector form
        :return:
        """
        batch_engine = BatchEngine(self.layers, self.loss_function, 0.07, self.batch_size)
        batch_engine.run(images, labels)

    def predict(self, image):
        for layer in self.layers:
            image = layer.forward(image)

        return np.argmax(image)

    def save(self):
        print("Here I should be saved")
        folder = "saved/cnn_%s" % datetime.today().strftime("%m-%d-%Y_%H-%M")
        os.makedirs(folder)
        file = open("%s/main.txt" % folder, "w")
        file.write("\n")
        for layer in self.layers:
            array_file = layer.save(folder)
            file.write(type(layer).__name__ + ":" + array_file)
            file.write("\n")
        file.close()

    @staticmethod
    def load(folder):
        path = "%s/main.txt" % folder
        file = open(path, "r")
        line = file.readline()
        cnn = Cnn(300)
        while not line is None:
            line = file.readline()
            if len(line) < 3:
                break
            cnn.add_layer(Cnn.str2layer(folder, line))
        return cnn

    @staticmethod
    def str2layer(directory, line):
        switcher = {
            "Conv2d": lambda folder, args: Conv2d.load(folder, args),
            "FullyConnected": lambda folder, args: FullyConnected.load(folder, args),
            "MaxPooling": lambda folder, args: MaxPooling.load(folder, args),
            "Flatten": lambda folder, args: Flatten.load(folder, args),
            "Softmax": lambda folder, args: Softmax()
        }
        parts = line.rstrip().split(":")
        func = switcher.get(parts[0])
        return func(directory, parts[1])
