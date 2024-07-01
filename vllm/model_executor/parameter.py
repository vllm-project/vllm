from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.nn import Parameter

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

__all__ = [
    "vLLMParameter", "PackedvLLMParameter", "PerTensorScaleParameter",
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
                 ignore_warnings: Optional[bool] = True):

        self._ignore_warnings = True
        self._weight_loader = weight_loader

    @property
    def weight_loader(self):
        return self._weight_loader


class _ColumnvLLMParameter(vLLMParameter):

    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self):
        return self._output_dim

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor,
                                    **kwargs):
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(self.output_dim,
                                             tp_rank * shard_size, shard_size)

    def load_merged_column_weight(self, param_data: torch.Tensor,
                                  loaded_weight: torch.Tensor,
                                  shard_offset: int, shard_size: int,
                                  **kwargs):

        if isinstance(
                self,
                PackedvLLMParameter) and self.packed_dim == self.output_dim:
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size)

        tp_rank = get_tensor_model_parallel_rank()
        param_data = param_data.narrow(self.output_dim, shard_offset,
                                       shard_size)
        loaded_weight = loaded_weight.narrow(self.output_dim,
                                             tp_rank * shard_size, shard_size)
        return param_data, loaded_weight

    def load_qkv_weight(self, param_data: torch.Tensor,
                        loaded_weight: torch.Tensor, num_heads: int,
                        shard_offset: int, shard_size: int, shard_id: str,
                        **kwargs):
        if isinstance(
                self,
                PackedvLLMParameter) and self.output_dim == self.packed_dim:
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size)

        tp_rank = get_tensor_model_parallel_rank()
        shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads
        param_data = param_data.narrow(self.output_dim, shard_offset,
                                       shard_size)
        loaded_weight = loaded_weight.narrow(self.output_dim,
                                             shard_id * shard_size, shard_size)

        return param_data, loaded_weight


# has column and row dims
class ModelWeightParameter(_ColumnvLLMParameter):

    def __init__(self, input_dim: int, **kwargs):
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self):
        return self._input_dim

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = self.data.shape[self.input_dim]
        loaded_weight = loaded_weight.narrow(self.input_dim,
                                             tp_rank * shard_size, shard_size)
        return loaded_weight


# group scales are loaded the same as our weights; have column and row
class GroupQuantScaleParameter(ModelWeightParameter):
    pass


# channel scales are loaded with column dim only; same as _ColumnvLLMParameter
class ChannelQuantScaleParameter(_ColumnvLLMParameter):
    pass


class PerTensorScaleParameter(vLLMParameter):

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

    def load_merged_column_weight(self, *args, **kwargs):
        return self._col_shard_splitter(*args, **kwargs)

    def load_qkv_weight(self, *args, **kwargs):
        return self._col_shard_splitter(*args, **kwargs)

    def load_column_parallel_weight(self, *args, **kwargs):
        return self._col_shard_splitter(*args, **kwargs)

    def _col_shard_splitter(self, param_data: torch.Tensor,
                            loaded_weight: torch.Tensor, shard_id: Union[str,
                                                                         int],
                            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _adjust_shard_indexes_for_marlin(self, shard_size, shard_offset):
        return shard_size * self.marlin_tile, shard_offset * self.marlin_tile

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        shard_size = shard_size // self.packed_factor
        shard_offset = shard_offset // self.packed_factor
        if self.marlin_tile is not None:
            return self._adjust_shard_indexes_for_marlin(
                shard_size, shard_offset)
        return shard_size, shard_offset
