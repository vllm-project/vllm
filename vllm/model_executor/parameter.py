from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.nn import Parameter

from vllm.logger import init_logger

__all__ = [
    "vLLMParameter", "PackedvLLMParameter", "ScalerToArrayvLLMParameter",
    "ModelWeightParameter", "ChannelQuantScaleParameter",
    "GroupQuantScaleParameter"
]

logger = init_logger(__name__)


class vLLMParameter(Parameter):

    def __new__(cls, data: torch.Tensor, **kwargs):

        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self,
                 data: torch.Tensor,
                 weight_loader: Callable,
                 use_column_loading: Optional[bool] = False,
                 use_row_loading: Optional[bool] = False,
                 ignore_warnings: Optional[bool] = True):

        self._ignore_warnings = True
        self._weight_loader = weight_loader
        self.use_column_loading = use_column_loading
        self.use_row_loading = use_row_loading
        """
        logger.warning(
            "Loading a weight without using default column loading "
            "or column shard splitting, assume the weight is the same "
            "for all partitions.")
        """
        self._use_metadata_loading = False

    @property
    def weight_loader(self):
        return self._weight_loader

    # TODO: should be part of ScalerToArrayvLLMParameter logic?
    @property
    def use_metadata_loading(self):
        return self._use_metadata_loading

    @use_metadata_loading.setter
    def use_metadata_loading(self, value: bool):
        self._use_metadata_loading = value


# uses row loading and column loading
class ModelWeightParameter(vLLMParameter):

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        self._input_dim = input_dim
        self._output_dim = output_dim
        # TODO: log using row loading and col loading
        super().__init__(**kwargs,
                         use_row_loading=True,
                         use_column_loading=True)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim


class GroupQuantScaleParameter(ModelWeightParameter):
    pass


# use col loading, now row loading
class ChannelQuantScaleParameter(vLLMParameter):

    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        # TODO: log using col loading
        super().__init__(**kwargs, use_column_loading=True)

    @property
    def output_dim(self):
        return self._output_dim


class ScalerToArrayvLLMParameter(vLLMParameter):

    def __init__(self, logical_widths: List[int], **kwargs):
        self.logical_widths = logical_widths
        self.qkv_idxs = {"q": 0, "k": 1, "v": 2}

        super().__init__(**kwargs)

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


class PackedvLLMParameter(ModelWeightParameter):

    def __init__(self,
                 packed_factor: int,
                 packed_dim: int,
                 marlin_tile_size: Optional[int] = None,
                 **kwargs):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile = marlin_tile_size
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
