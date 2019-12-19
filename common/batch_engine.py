class BatchEngine:

    def __init__(self, layers, loss_function):
        self.loss_fn = loss_function
        self.layers = layers
        self.cache = list()

    def run(self, images, labels):
        total_loss = 0
        for layer in self.layers:
            self.cache.append(list())

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
                self.cache[layer_index].append((saved_image, theta))
                layer_index += 1

        for (layer, cache) in zip(self.layers, self.cache):
            layer.update_weights(cache, 0.1)
