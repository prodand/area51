from abc import abstractmethod, ABC


class BaseLayer(ABC):

    @abstractmethod
    def forward(self, image):
        ...

    @abstractmethod
    def back(self):
        ...