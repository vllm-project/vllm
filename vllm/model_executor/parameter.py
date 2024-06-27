from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.nn import Parameter

from vllm.logger import init_logger

__all__ = [
    "vLLMParameter", "PackedvLLMParameter", "ScalerToArrayvLLMParameter"
]

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
                 use_row_loading: Optional[bool] = False,
                 use_col_loading: Optional[bool] = False,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 weight_loader: Optional[Callable] = None,
                 packed: Optional[bool] = False,
                 use_col_shard_split: Optional[bool] = False,
                 ignore_warnings: Optional[bool] = True):

        self._ignore_warnings = True
        self._use_row_loading = use_row_loading
        self._use_col_loading = use_col_loading

        if self._use_row_loading and input_dim is None:
            raise ValueError(
                "In order to use row loading, an input dim must be set")
        if self._use_col_loading and output_dim is None:
            raise ValueError(
                "In order to use col loading, an output dim must be set")

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weight_loader = weight_loader
        self._is_packed = packed
        self._use_col_shard_split = use_col_shard_split
        if not self._use_col_loading and not self._use_col_shard_split:
            logger.warning(
                "Loading a weight without using default column loading "
                "or column shard splitting, assume the weight is the same "
                "for all partitions.")
        self._use_metadata_loading = False

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
    def use_col_shard_split(self):
        return self._use_col_shard_split

    @property
    def use_column_loading(self):
        return self._use_col_loading

    @property
    def use_row_loading(self):
        return self._use_row_loading

    # TODO: should be part of ScalerToArrayvLLMParameter logic?
    @property
    def use_metadata_loading(self):
        return self._use_metadata_loading

    @use_metadata_loading.setter
    def use_metadata_loading(self, value: bool):
        self._use_metadata_loading = value


class ScalerToArrayvLLMParameter(vLLMParameter):

    def __init__(self, logical_widths: List[int], **kwargs):
        self.logical_widths = logical_widths
        self.qkv_idxs = {"q": 0, "k": 1, "v": 2}

        super().__init__(**kwargs, use_col_shard_split=True)

        if self.use_column_loading:
            raise ValueError("Can only use one of column shard splitting "
                             "or column default loading")

    def _shard_id_as_int(self, shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id

        assert isinstance(shard_id, str)
        assert shard_id in self.qkv_idxs
        return self.qkv_idxs[shard_id]

    def col_shard_splitter(
            self, param_data: torch.Tensor, loaded_weight: torch.Tensor,
            shard_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_id = self._shard_id_as_int(shard_id)
        offset = sum(self.logical_widths[:shard_id])
        size = self.logical_widths[shard_id]
        # update loaded weight with copies for broadcast.
        loaded_weight = loaded_weight.repeat(size)
        return param_data[offset:offset + size], loaded_weight


class PackedvLLMParameter(vLLMParameter):

    def __init__(self,
                 packed_factor: int,
                 packed_dim: int,
                 marlin_tile_size: Optional[int] = None,
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
