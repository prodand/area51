from plain.layers.base_layer import BaseLayer


class Flatten(BaseLayer):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, image):
        if image.shape != self.shape:
            raise RuntimeError('Image shape does not match')

        return image.reshape((self.shape[0] * self.shape[1] * self.shape[2], 1))

    def back(self, activation_theta):
        return activation_theta.reshape(self.shape)

    def update_weights(self, layer_cache, learning_rate):
        pass
