import time

import matplotlib.pyplot as plt
import numpy as np


class BatchEngine:

    def __init__(self, layers, loss_function, batch_size=32):
        self.loss_fn = loss_function
        self.layers = layers
        self.batch_size = batch_size
        self.cache = list()
        self.train_plt_loss = []
        self.validation_plt_loss = []

    def run(self, images, labels, learning_rate, epoch):
        print('New Round')
        train_loss = 0
        batches_count = int(len(images) / self.batch_size)
        for batch_index in range(0, batches_count):
            start = batch_index * self.batch_size
            end = (batch_index + 1) * self.batch_size
            start_time = time.time()
            loss = self.run_batch(images[start:end], labels[start:end], learning_rate)
            train_loss += loss
            print('%s Loss: %s [%s]' % (epoch, str(loss / self.batch_size), (time.time() - start_time)))

        return train_loss / len(images)

    def validate(self, images, labels):
        total_loss = 0
        correct_answers = 0
        for index, image in enumerate(images, start=0):
            convolved_image = image
            for layer in self.layers:
                convolved_image = layer.forward(convolved_image)
            actual = np.argmax(convolved_image)
            correct_answers += labels[index][actual]
            error = self.loss_fn.loss(convolved_image, labels[index])
            total_loss += error

        return total_loss / len(images), correct_answers / len(images)

    def run_batch(self, images, labels, learning_rate):
        total_loss = 0
        layers_cache = list()
        for layer in self.layers:
            layers_cache.append(list())

        for index, image in enumerate(images, start=0):
            forward_activations = [image]
            convolved_image = image
            for layer in self.layers:
                convolved_image = layer.forward(convolved_image)
                forward_activations.append(convolved_image)

            last = forward_activations.pop()
            error = self.loss_fn.loss(last, labels[index])
            theta = self.loss_fn.delta(last, labels[index])

            total_loss += error

            activation_thetas = [theta]
            for layer in reversed(self.layers[1:]):
                theta = layer.back(theta)
                activation_thetas.append(theta)

            layer_index = 0
            for (saved_image, theta) in zip(forward_activations, reversed(activation_thetas)):
                layers_cache[layer_index].append((saved_image, theta))
                layer_index += 1

        for (layer, cache) in zip(self.layers, layers_cache):
            layer.update_weights(cache, learning_rate)

        return total_loss

    def plot(self, train_loss, validation_loss):
        self.train_plt_loss.append(train_loss)
        self.validation_plt_loss.append(validation_loss)
        epochs = [i for i in range(1, len(self.train_plt_loss) + 1)]
        plt.plot(epochs, self.train_plt_loss, label="Train loss")
        plt.plot(epochs, self.validation_plt_loss, label="Validation loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.title("Simple Plot")

        plt.legend()

        plt.show()
