from layers.base_layer import BaseLayer


class BatchEngine:

    def run(self, images, layers: list[BaseLayer]):
        for image in images:
            for layer in layers:
                convolved_image = layer.forward(image)
