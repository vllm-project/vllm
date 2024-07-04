from typing import Optional, Union

import torch


class DeferredTensor:

    def __init__(self, slice_view, slices=None, narrow_count=0):
        self.slice_view = slice_view
        self.shape = slice_view.get_shape()
        if slices is None:
            slices = tuple(slice(None, None, None) for x in self.shape)
        self.slices = slices
        self.narrow_count = narrow_count
        self.materialized_tensor: Optional[torch.Tensor] = None

    def __hash__(self):
        return id(self)

    def narrow(self, dim, start_idx, length) -> 'DeferredTensor':
        if self.narrow_count > 0:
            raise ValueError(
                "Cannot narrow a tensor that has already been narrowed")
        slices = list(self.slices)
        slices[dim] = slice(start_idx, start_idx + length)
        return DeferredTensor(self.slice_view, tuple(slices), 1)

    def materialize(self) -> torch.Tensor:
        if self.materialized_tensor is None:
            self.materialized_tensor = self.slice_view[self.slices]
        return self.materialized_tensor


def convert_like(x: Union[DeferredTensor, torch.Tensor],
                 param: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DeferredTensor):
        x = x.materialize()
    assert isinstance(x, torch.Tensor)
    return x.to(device=param.device)
