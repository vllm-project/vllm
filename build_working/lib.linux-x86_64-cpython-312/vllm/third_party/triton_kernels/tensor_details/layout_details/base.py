from abc import ABC, abstractmethod


class Layout(ABC):

    def __init__(self, shape) -> None:
        self.initial_shape = shape

    @abstractmethod
    def swizzle_data(self, data):
        pass

    @abstractmethod
    def unswizzle_data(self, data):
        pass

    @abstractmethod
    def swizzle_block_shape(self, block_shape):
        pass
