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

    def __init__(self, batch_size, learning_rate, learning_rate_decay=0.005):
        self.layers = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.folds_number = 10

    def add_layer(self, layer):
        self.layers.append(layer)

    def learn(self, images, labels):
        """
        :param images: data as (image_index x channels x width x heights)
        :param labels: expected result in vector form
        :return:
        """
        batch_engine = BatchEngine(self.layers, self.loss_function, self.batch_size)

        learned = False
        epoch = 1
        excluded_fold = 0
        while not learned:
            train_images, train_labels = self.extract_train_data(images, labels, excluded_fold)
            train_loss = batch_engine.run(train_images,
                                          train_labels,
                                          self.learning_rate - epoch * self.learning_rate_decay,
                                          epoch)

            validation_images, validation_labels = self.extract_validate_data(images, labels, excluded_fold)
            validation_loss, percent = batch_engine.validate(validation_images, validation_labels)

            print("Val: %s; Train: %s; Epoch: %s; Percent: %s" % (validation_loss, train_loss, epoch, percent))

            excluded_fold = excluded_fold + 1 if excluded_fold < (self.folds_number - 1) else 0
            epoch += 1
            learned = train_loss < 0.01
        self.save()

    def predict(self, image):
        for layer in self.layers:
            image = layer.forward(image)

        return np.argmax(image)

    def extract_train_data(self, images, labels, excluded_fold):
        fold_size = int(len(images) / self.folds_number)
        fold_start = excluded_fold * fold_size
        fold_end = (excluded_fold + 1) * fold_size
        if excluded_fold == 0:
            return images[fold_end:], labels[fold_end:]
        if excluded_fold == (self.folds_number - 1):
            return images[0:fold_start], labels[0:fold_start]
        return np.concatenate((images[0:fold_start], images[fold_end:])), np.concatenate(
            (labels[0:fold_start], labels[fold_end:]))

    def extract_validate_data(self, images, labels, fold: int):
        fold_size = int(len(images) / self.folds_number)
        fold_start = fold * fold_size
        fold_end = (fold + 1) * fold_size
        return images[fold_start:fold_end], labels[fold_start:fold_end]

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
        cnn = Cnn(300, 0.07)
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
