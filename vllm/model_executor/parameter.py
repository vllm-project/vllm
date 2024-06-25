from torch.nn import Parameter
from typing import Optional, Callable, Tuple, Union
import torch
from functools import partial

__all__ = ["vLLMParameter", "PackedParameter"]


class vLLMParameter(Parameter):

    def __new__(cls,
                data: torch.Tensor,
                requires_grad: Optional[bool] = False,
                **kwargs):

        return super().__new__(cls, data=data, requires_grad=requires_grad)

    def __init__(self,
                 data: torch.Tensor,
                 requires_grad: Optional[bool] = False,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 weight_loader: Optional[Callable] = None):

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weight_loader = weight_loader
        self._use_default_loading = True  # when output_dim is not None or input_dim is not none
        self._use_shard_splitting = False
        self._is_metadata = False

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def weight_loader(self):
        return self._weight_loader

    @property
    def use_default_loading(self):
        return self._use_default_loading

    @use_default_loading.setter
    def use_default_loading(self, value: bool):
        self._use_default_loading = value

    @property
    def use_shard_splitting(self):
        return self._use_shard_splitting

    @use_shard_splitting.setter
    def use_shard_splitting(self, value: bool):
        self._use_shard_splitting = value

    @property
    def is_metadata(self):
        return self._is_metadata

    @is_metadata.setter
    def is_metadata(self, value: bool):
        self._is_metadata = value

    def shard_splitter(self, *args, **kwargs):
        return NotImplementedError()


class PackedParameter(vLLMParameter):

    def __init__(self,
                 packed_factor: int,
                 packed_dim: int,
                 marlin_tile: Optional[int] = None,
                 use_bitsandbytes: Optional[int] = False,
                 **kwargs):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile = marlin_tile
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile(self):
        return self._marlin_tile

    def _adjust_marlin_shard(self, shard_size, shard_offset):
        return shard_size * self.marlin_tile, shard_offset * self.marlin_tile

    def adjust_packed_shard(self, shard_size, shard_offset):
        shard_size = shard_size // self.packed_factor
        shard_offset = shard_offset // self.packed_factor
        if self.marlin_tile is not None:
            return self._adjust_marlin_shard(shard_size, shard_offset)
        return shard_size, shard_offset
