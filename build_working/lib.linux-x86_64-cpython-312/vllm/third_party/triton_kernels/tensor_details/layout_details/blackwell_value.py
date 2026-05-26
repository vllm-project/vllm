import torch
from .base import Layout


class BlackwellMXValueLayout(Layout):
    name: str = "BLACKWELL_VALUE"

    def __init__(self, shape) -> None:
        super().__init__(shape)
        self.shape = shape

    def swizzle_data(self, data):
        # permutation needed to make `data` row major
        to_row_major = sorted(range(data.ndim), key=lambda d: (data.stride(d), d))[::-1]
        # permutation  needed to retrieve original order
        inv = [0] * data.ndim
        for i, d in enumerate(to_row_major):
            inv[d] = i
        # leading dimension must be padded to be aligned to 128
        align_dim = lambda x: (x + 128 - 1) // 128 * 128
        major_dim = data.stride().index(1)
        pad = align_dim(data.shape[major_dim]) - data.shape[major_dim]
        data = torch.nn.functional.pad(data.permute(to_row_major), (0, pad)).permute(inv)
        return data

    def unswizzle_data(self, data: torch.Tensor):
        # Trim padding along all dims back to the original shape recorded at init.
        assert data.ndim == len(self.shape), "Rank mismatch between data and recorded shape"
        sizes = [min(data.size(i), self.shape[i]) for i in range(data.ndim)]
        return data[tuple(slice(0, s) for s in sizes)]

    def swizzle_block_shape(self, block_shape):
        return block_shape
