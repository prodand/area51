import matplotlib.pyplot as plt


class BatchEngine:

    def __init__(self, layers, loss_function, batch_size=32):
        self.loss_fn = loss_function
        self.layers = layers
        self.batch_size = batch_size
        self.cache = list()
        self.folds_number = 10
        self.train_plt_loss = []
        self.validation_plt_loss = []

    def run(self, images, labels):
        learned = False
        epoch = 1
        excluded_fold = 0
        while not learned:
            print('New Round')
            train_loss = 0
            for fold in range(0, self.folds_number):
                if fold == excluded_fold:
                    continue
                train_images = self.extract_fold(images, fold)
                train_labels = self.extract_fold(labels, fold)
                batches_count = int(len(train_images) / self.batch_size)
                for batch_index in range(0, batches_count):
                    start = batch_index * self.batch_size
                    end = (batch_index + 1) * self.batch_size
                    learned, loss = self.run_batch(train_images[start:end], train_labels[start:end])
                    train_loss += loss
                    print('%s Loss: %s' % (epoch, str(loss)))
                    if learned:
                        break

            train_loss = self.average_train_loss(images, train_loss)
            validation_loss = self.validate(self.extract_fold(images, excluded_fold),
                                            self.extract_fold(labels, excluded_fold))
            self.plot(train_loss, validation_loss)
            excluded_fold = excluded_fold + 1 if excluded_fold < self.folds_number else 0
            epoch += 1

    def validate(self, images, labels):
        total_loss = 0
        for index, image in enumerate(images, start=0):
            convolved_image = image
            for layer in self.layers:
                convolved_image = layer.forward(convolved_image)
            error = self.loss_fn.loss(convolved_image, labels[index])
            total_loss += error

        return total_loss / len(images)

    def run_batch(self, images, labels):
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

        if total_loss < 0.01:
            return True, total_loss

        for (layer, cache) in zip(self.layers, layers_cache):
            layer.update_weights(cache, 0.07)

        return False, total_loss

    def extract_fold(self, images, fold: int):
        fold_size = int(len(images) / self.folds_number)
        fold_start = fold * fold_size
        fold_end = (fold + 1) * fold_size
        return images[fold_start:fold_end]

    def average_train_loss(self, images, loss):
        images_count = len(images) * (self.folds_number - 1) / self.folds_number
        return loss / images_count

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
