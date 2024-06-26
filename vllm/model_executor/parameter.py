from torch.nn import Parameter
from typing import Optional, Callable
import torch
from vllm.logger import init_logger

__all__ = ["vLLMParameter", "PackedvLLMParameter"]

logger = init_logger(__name__)


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
                 weight_loader: Optional[Callable] = None,
                 packed: Optional[bool] = False,
                 ignore_warnings: Optional[bool] = True):

        self._ignore_warnings = True
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weight_loader = weight_loader
        self._is_packed = packed
        self._use_column_loading = True if self._output_dim is not None else False
        self._use_row_loading = True if self._input_dim is not None else False
        self._use_row_shard_splitting = False
        self._use_col_shard_splitting = False
        self._use_metadata_loading = False
        self._use_bits_and_bytes = False

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
    def is_packed(self):
        return self._is_packed

    @property
    def use_row_loading(self):
        return self._use_row_loading

    @property
    def use_column_loading(self):
        return self._use_column_loading

    @property
    def use_row_shard_splitting(self):
        return self._use_row_shard_splitting

    @use_row_shard_splitting.setter
    def use_row_shard_splitting(self, value: bool):
        if self.use_row_loading and value:
            raise ValueError(
                "Can only use one of row shard splitting or row default loading"
            )
        self._use_row_shard_splitting = value

    @property
    def use_col_shard_splitting(self):
        return self._use_col_shard_splitting

    @use_col_shard_splitting.setter
    def use_col_shard_splitting(self, value: bool):
        if self._use_column_loading and value:
            raise ValueError(
                "Can only use one of column shard splitting or column default loading"
            )
        if not self._use_column_loading and not value:
            logger.warning(
                "Loading a weight without using default column loading "
                "or column shard splitting, assume the weight is the same "
                "for all partitions.")
        self._use_col_shard_splitting = value

    @property
    def use_metadata_loading(self):
        return self._use_metadata_loading

    @use_metadata_loading.setter
    def use_metadata_loading(self, value: bool):
        self._use_metadata_loading = value

    @property
    def use_bits_and_bytes(self):
        return self._use_bits_and_bytes

    @use_bits_and_bytes.setter
    def use_bits_and_bytes(self, value: bool):
        self._use_bits_and_bytes = value

    # Should shard splitting params be in a another param?
    # Packed Parameters which are sharded?
    def row_shard_splitter(self, *args, **kwargs):
        return NotImplementedError(
            "A row shard splitter is not defined for this param")

    def col_shard_splitter(self, *args, **kwargs):
        return NotImplementedError(
            "A column shard splitter is not defined for this param")


class PackedvLLMParameter(vLLMParameter):

    def __init__(self,
                 packed_factor: int,
                 packed_dim: int,
                 marlin_tile_size: Optional[int] = None,
                 use_bitsandbytes: Optional[int] = False,
                 **kwargs):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile = marlin_tile_size
        super().__init__(**kwargs, packed=True)

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
