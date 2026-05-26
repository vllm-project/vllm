from .base import Layout


class StridedLayout(Layout):
    name: str = None

    def __init__(self, shape) -> None:
        super().__init__(shape)

    def swizzle_data(self, data):
        return data

    def unswizzle_data(self, data):
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape
