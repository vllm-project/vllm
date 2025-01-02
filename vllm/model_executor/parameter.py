from enum import Enum
from fractions import Fraction
from typing import Callable, Optional, Union

import torch
from torch.nn import Parameter

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger

__all__ = [
    "BasevLLMParameter", "PackedvLLMParameter", "PerTensorScaleParameter",
    "ModelWeightParameter", "ChannelQuantScaleParameter",
    "GroupQuantScaleParameter", "PackedColumnParameter", "RowvLLMParameter"
]

logger = init_logger(__name__)


class Features(Enum):
    """
    Enum for parameter features.
    """
    Base = 0
    ModelWeight = 1
    PerTensorScale = 2
    ChannelQuantScale = 3
    GroupQuantScale = 4
    PackedColumn = 5
    Packed = 6
    Row = 7
    Column = 8
    BlockQuantScale = 9
    HQQEmpty = 10
    HQQWeight = 11
    HQQZeroScale = 12


def add_param_feature(obj, feature_name):
    """
    Add a feature to a parameter object.
    """
    if not hasattr(obj, 'param_features'):
        obj.param_features = []
    obj.param_features.append(feature_name)


def has_any_param_feature(obj, feature_name_or_list):
    """
    Check if a parameter object has any of the specified features.
    """
    if not hasattr(obj, 'param_features'):
        return False
    if isinstance(feature_name_or_list, Features):
        return feature_name_or_list in obj.param_features
    elif isinstance(feature_name_or_list, list):
        for feature in feature_name_or_list:
            assert isinstance(feature, Features)
            if feature in obj.param_features:
                return True
        return False
    return False


def BasevLLMParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    return param


def wrap_base_vllm_parameter(param: Parameter, weight_loader: Callable,
                             **kwargs):
    """
    Add basic functionality for vLLM linear layer parameters.
    """

    def _assert_and_load(param: Parameter, loaded_weight: torch.Tensor):
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    param.weight_loader = weight_loader
    param.load_column_parallel_weight = lambda loaded_weight: _assert_and_load(
        param, loaded_weight)
    param.load_row_parallel_weight = lambda loaded_weight: _assert_and_load(
        param, loaded_weight)
    param.load_merged_column_weight = \
        lambda loaded_weight, **kwargs: _assert_and_load(
        param, loaded_weight)
    param.load_qkv_weight = lambda loaded_weight, **kwargs: _assert_and_load(
        param, loaded_weight)
    add_param_feature(param, Features.Base)


def wrap_column_vllm_parameter(param: Parameter, output_dim: int,
                               **kwargs) -> None:
    """
    Add functionality to the parameter for loading weights into
    linear layers with column parallelism. This includes QKV and MLP
    layers which are not already fused on disk. Requires an output
    dimension to be defined. Called within the weight loader of each
    of the column parallel linear layers.
    """

    def load_column_parallel_weight(param: Parameter,
                                    loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[param.output_dim]
        loaded_weight = loaded_weight.narrow(param.output_dim,
                                             tp_rank * shard_size, shard_size)
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    def load_merged_column_weight(param: Parameter,
                                  loaded_weight: torch.Tensor, **kwargs):
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        if (has_any_param_feature(param,
                                  [Features.PackedColumn, Features.Packed])
                and param.output_dim == param.packed_dim):
            shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size)

        param_data = param.data

        tp_rank = get_tensor_model_parallel_rank()
        param_data = param_data.narrow(param.output_dim, shard_offset,
                                       shard_size)
        loaded_weight = loaded_weight.narrow(param.output_dim,
                                             tp_rank * shard_size, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(param: Parameter, loaded_weight: torch.Tensor,
                        **kwargs):
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")

        if (has_any_param_feature(param,
                                  [Features.PackedColumn, Features.Packed])
                and output_dim == param.packed_dim):
            shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size)

        param_data = param.data
        tp_rank = get_tensor_model_parallel_rank()
        shard_id = tp_rank if shard_id == "q" else tp_rank // num_heads
        param_data = param_data.narrow(param.output_dim, shard_offset,
                                       shard_size)
        loaded_weight = loaded_weight.narrow(param.output_dim,
                                             shard_id * shard_size, shard_size)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    param.output_dim = output_dim
    param.load_column_parallel_weight = lambda loaded_weight: (
        load_column_parallel_weight(param, loaded_weight))
    param.load_merged_column_weight = lambda loaded_weight, **kwargs: (
        load_merged_column_weight(param, loaded_weight, **kwargs))
    param.load_qkv_weight = lambda loaded_weight, **kwargs: (load_qkv_weight(
        param, loaded_weight, **kwargs))
    add_param_feature(param, Features.Column)


def RowvLLMParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    return param


def wrap_row_vllm_parameter(param: Parameter, input_dim: int,
                            **kwargs) -> None:
    """
    Add functionality to the parameter for loading weights into
    linear layers with row parallelism. This includes layers
    which are fused on disk. Requires an input dimension to be
    defined. Called within the weight loader of each of the
    row parallel linear layers.
    """

    def load_row_parallel_weight(param: Parameter,
                                 loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[input_dim]
        loaded_weight = loaded_weight.narrow(input_dim, tp_rank * shard_size,
                                             shard_size)
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    param.input_dim = input_dim
    param.load_row_parallel_weight = lambda loaded_weight: (
        load_row_parallel_weight(param, loaded_weight))
    add_param_feature(param, Features.Row)


def ModelWeightParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    wrap_model_weight_parameter(param, **kwargs)
    return param


def wrap_model_weight_parameter(param: Parameter, **kwargs) -> None:
    add_param_feature(param, Features.ModelWeight)


def GroupQuantScaleParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    wrap_group_quant_scale_parameter(param, **kwargs)
    return param


def wrap_group_quant_scale_parameter(param: Parameter, **kwargs) -> None:
    add_param_feature(param, Features.GroupQuantScale)


def ChannelQuantScaleParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_channel_quant_scale_parameter(param, **kwargs)
    return param


def wrap_channel_quant_scale_parameter(param: Parameter, **kwargs) -> None:
    add_param_feature(param, Features.ChannelQuantScale)


def PerTensorScaleParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_per_tensor_scale_parameter(param, **kwargs)
    return param


def wrap_per_tensor_scale_parameter(param: Parameter, **kwargs) -> None:
    """
    Add functionality for scales where the number of scales is
    equivalent to the number of logical matrices in fused linear
    layers (e.g. for QKV, there are 3 scales loaded from disk).
    This is relevant to weights with per-tensor quantization.
    Adds functionality to map the scalers to a shard during
    weight loading.

    Note: additional parameter manipulation may be handled
    for each quantization config specifically, within
    process_weights_after_loading
    """

    def shard_id_as_int(shard_id: Union[str, int]) -> int:
        if isinstance(shard_id, int):
            return shard_id

        # if not int, assume shard_id for qkv
        # map to int and return
        assert isinstance(shard_id, str)
        assert shard_id in param.qkv_idxs
        return param.qkv_idxs[shard_id]

    def load_into_shard_id(param: Parameter, loaded_weight: torch.Tensor,
                           shard_id: int, **kwargs):
        param_data = param.data
        shard_id = param.shard_id_as_int(shard_id)
        # AutoFP8 scales do not have a shape
        # compressed-tensors scales do have a shape
        if len(loaded_weight.shape) != 0:
            assert loaded_weight.shape[0] == 1
            loaded_weight = loaded_weight[0]
        param_data = param_data[shard_id]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_row_parallel_weight(param: Parameter,
                                 loaded_weight: torch.Tensor):
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    def load_merged_column_weight(param: Parameter, **kwargs):
        param.load_into_shard_id(param, **kwargs)

    def load_qkv_weight(param: Parameter, **kwargs):
        param.load_into_shard_id(param, **kwargs)

    def load_column_parallel_weight(param: Parameter,
                                    loaded_weight: torch.Tensor):
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    param.qkv_idxs = {"q": 0, "k": 1, "v": 2}
    param.shard_id_as_int = shard_id_as_int
    param.load_into_shard_id = load_into_shard_id
    param.load_row_parallel_weight = lambda loaded_weight: (
        load_row_parallel_weight(param, loaded_weight))
    param.load_merged_column_weight = lambda **kwargs: (
        load_merged_column_weight(param, **kwargs))
    param.load_qkv_weight = lambda **kwargs: (load_qkv_weight(param, **kwargs))
    param.load_column_parallel_weight = lambda loaded_weight: (
        load_column_parallel_weight(param, loaded_weight))
    add_param_feature(param, Features.PerTensorScale)


def PackedColumnParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_packed_column_parameter(param, **kwargs)
    return param


def wrap_packed_column_parameter(param: Parameter,
                                 packed_factor: Union[int, Fraction],
                                 packed_dim: int,
                                 marlin_tile_size: Optional[int] = None,
                                 **kwargs) -> None:
    """
    Add properties and methods for parameters which are packed on disk
    and support column parallelism only. See PackedvLLMParameter
    for more details on the packed properties.
    """

    def adjust_shard_indexes_for_packing(shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=packed_factor,
            marlin_tile_size=marlin_tile_size)

    param.packed_factor = packed_factor
    param.packed_dim = packed_dim
    param.marlin_tile_size = marlin_tile_size
    param.adjust_shard_indexes_for_packing = adjust_shard_indexes_for_packing
    add_param_feature(param, Features.PackedColumn)


def PackedvLLMParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    wrap_packed_vllm_parameter(param, **kwargs)
    return param


def wrap_packed_vllm_parameter(param: Parameter,
                               packed_factor: Union[int, Fraction],
                               packed_dim: int,
                               marlin_tile_size: Optional[int] = None,
                               **kwargs) -> None:
    """
    Add properties and methods for parameters which are packed on disk.
    Example: GPTQ Marlin weights are int4 or int8, packed into int32.
    Extends the ModelWeightParameter to take in the
    packed factor, the packed dimension, and optionally, marlin
    tile size for marlin kernels. Adjusts the shard_size and
    shard_offset for fused linear layers model weight loading
    by accounting for packing and optionally, marlin tile size.
    """

    def adjust_shard_indexes_for_packing(shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=packed_factor,
            marlin_tile_size=marlin_tile_size)

    param.packed_factor = packed_factor
    param.packed_dim = packed_dim
    param.marlin_tile_size = marlin_tile_size
    param.adjust_shard_indexes_for_packing = adjust_shard_indexes_for_packing
    add_param_feature(param, Features.Packed)


def BlockQuantScaleParameter(data: torch.Tensor, **kwargs) -> Parameter:
    param = Parameter(data, requires_grad=False)
    wrap_base_vllm_parameter(param, **kwargs)
    wrap_column_vllm_parameter(param, **kwargs)
    wrap_row_vllm_parameter(param, **kwargs)
    add_param_feature(param, Features.BlockQuantScale)
    return param


def permute_param_layout_(param: Parameter, input_dim: int, output_dim: int,
                          **kwargs) -> Parameter:
    """
    Permute a parameter's layout to the specified input and output dimensions, 
    useful for forcing the parameter into a known layout, for example, if I need
    a packed (quantized) weight matrix to be in the layout 
        {input_dim = 0, output_dim = 1, packed_dim = 0}
    then I can call:
        permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
    to ensure x is in the correct layout (permuting it to the correct layout if 
    required, asserting if it cannot get it to the correct layout)
    """

    curr_input_dim = getattr(param, "input_dim", None)
    curr_output_dim = getattr(param, "output_dim", None)

    if curr_input_dim is None or curr_output_dim is None:
        assert param.data.dim() == 2,\
            "permute_param_layout_ only supports 2D parameters when either "\
            "input_dim or output_dim is not set"

    # if one of the dimensions is not set, set it to the opposite of the other
    #  we can only do this since we asserted the parameter is 2D above
    if curr_input_dim is None:
        assert curr_output_dim is not None,\
            "either input or output dim must be set"
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        assert curr_input_dim is not None,\
            "either input or output dim must be set"
        curr_output_dim = (curr_input_dim + 1) % 2

    # create permutation from the current layout to the layout with
    # self.input_dim at input_dim and self.output_dim at output_dim preserving
    # other dimensions
    perm = [
        i for i in range(param.data.dim())
        if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    if "packed_dim" in kwargs:
        assert hasattr(param, "packed_dim") and\
            param.packed_dim == perm[kwargs["packed_dim"]],\
            "permute_param_layout_ currently doesn't support repacking"

    param.data = param.data.permute(*perm)
    if hasattr(param, "_input_dim"):
        param._input_dim = input_dim
    if hasattr(param, "_output_dim"):
        param._output_dim = output_dim
    if "packed_dim" in kwargs and hasattr(param, "_packed_dim"):
        param._packed_dim = kwargs["packed_dim"]

    return param


def _adjust_shard_indexes_for_marlin(shard_size, shard_offset,
                                     marlin_tile_size):
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def _adjust_shard_indexes_for_packing(shard_size, shard_offset, packed_factor,
                                      marlin_tile_size):
    shard_size = shard_size // packed_factor
    shard_offset = shard_offset // packed_factor
    if marlin_tile_size is not None:
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size)
    return shard_size, shard_offset
