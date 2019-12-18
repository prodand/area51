import numpy as np


class BatchEngine:

    def __init__(self, layers, loss_function):
        self.loss_fn = loss_function
        self.cache = np.array([])
        self.layers = layers

    def run(self, images, labels):
        total_loss = 0
        for index, image in enumerate(images, start=0):
            forward_activations = [image]
            for layer in self.layers:
                convolved_image = layer.forward(image)
                forward_activations.append(convolved_image)

            error = self.loss_fn.loss(forward_activations[-1], labels[index])
            theta = self.loss_fn.delta(forward_activations[-1], labels[index])

            total_loss += error

            # TODO: need to align with forwards
            activation_thetas = []
            for layer in reversed(self.layers):
                theta_error = layer.back(theta)
                activation_thetas.append(theta_error)
            layer_index = 0
            for (image, theta) in forward_activations, reversed(activation_thetas):
                layer_cache =
                self.cache[layer_index].append
