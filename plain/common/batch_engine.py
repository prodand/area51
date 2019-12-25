class BatchEngine:

    def __init__(self, layers, loss_function, batch_size=32):
        self.loss_fn = loss_function
        self.layers = layers
        self.batch_size = batch_size
        self.cache = list()

    def run(self, images, labels):
        batches_count = int(len(images) / self.batch_size)
        learned = False
        while not learned:
            print('New Round')
            for batch_index in range(0, batches_count):
                start = batch_index * self.batch_size
                end = (batch_index + 1) * self.batch_size
                learned, loss = self.run_batch(images[start:end], labels[start:end])
                print('Loss: ' + str(loss))
                if learned:
                    break

    def run_batch(self, images, labels):
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

        if total_loss < 0.01:
            return True, total_loss

        for (layer, cache) in zip(self.layers, self.cache):
            layer.update_weights(cache, 0.3)

        return False, total_loss
