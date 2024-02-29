import contextlib
import contextvars
import functools
import threading
import typing
from types import MethodType
from typing import Optional

import torch
from torch import nn

from vllm.model_executor.layers.activation import ScaledActivation
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear, \
    QKVParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.logger import init_logger

logger = init_logger(__name__)


## Monkey patch for Parameter to ensure `requires_grad=False`
from torch.nn.parameter import Parameter

# Save the original __init__ method for later use
original_new = Parameter.__new__

def _new(cls, data, requires_grad=False):
    return original_new(cls, data, requires_grad=requires_grad)

# Replace the original __init__ method with our new one
Parameter.__new__ = _new

def tensorizer_loader(params_dict):
    return _TensorizerWeightsLoaderImpl(params_dict).context_manager()

def qkv_weight_loader(self,
                  param: Parameter,
                  loaded_weight: torch.Tensor,
                  loaded_shard_id: Optional[str] = None):
    param_data = param.data
    output_dim = getattr(param, "output_dim", None)
    if output_dim is None:
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
        return

    assert loaded_shard_id in ["q", "k", "v"]
    if output_dim is not None:
        if loaded_shard_id == "q":
            shard_offset = 0
            shard_size = self.total_num_heads * self.head_size
        elif loaded_shard_id == "k":
            shard_offset = self.total_num_heads * self.head_size
            shard_size = self.total_num_kv_heads * self.head_size
        elif loaded_shard_id == "v":
            shard_offset = (self.total_num_heads +
                            self.total_num_kv_heads) * self.head_size
            shard_size = self.total_num_kv_heads * self.head_size

    else:
        ignore_warning = getattr(param, "ignore_warning", False)
        if not ignore_warning:
            logger.warning(
                "Loading a weight without `output_dim` attribute in "
                "QKVParallelLinear, assume the weight is the same "
                "for all partitions.")
    param_data[shard_offset: shard_offset + shard_size].copy_(loaded_weight)

def tensorizer_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: Optional[int] = None):
    if param.nelement() == 0:
        param.set_(loaded_weight)
    else:
        # This is unavoidable for concatenating layers like QKVParallelLinear and certain buffers
        param.copy_(loaded_weight)

class _TensorizerWeightsLoaderImpl:
    is_active = contextvars.ContextVar("_TensorizerWeightsLoaderImpl.is_active", default=False)

    def __init__(self, params_dict):
        self.params_dict = params_dict
        self._original_loader = {}
        for param_name, param in self.params_dict.items():
            if hasattr(param, "weight_loader"):
                self._original_loader[param_name] = param.weight_loader


    @contextlib.contextmanager
    def context_manager(self):
        if self.is_active.get():
            yield
            return

        for param_name, param in self.params_dict.items():
            if not hasattr(param, "weight_loader"):
                continue
            else:
                layer_type = param.weight_loader.__self__
                if not (isinstance(layer_type, QKVParallelLinear) | isinstance(layer_type, MergedColumnParallelLinear)):
                    param.weight_loader = tensorizer_weight_loader
                else:
                    ## For QKVParallelLinear, and MergedColumnParallelLinear
                    param.resize_(param.shape[:-1])

        reset_token = self.is_active.set(True)

        try:
            yield
        finally:
            self.is_active.reset(reset_token)
            for param_name, param in self.params_dict.items():
                if hasattr(param, "weight_loader"):
                    param.weight_loader = self._original_loader[param_name]



@contextlib.contextmanager
def zero_length_init() -> typing.ContextManager:
    """
    Suppress the initialization of weights while loading a model.
    Appends a zero dimension (i.e. (...) -> (..., 0)) to the ``size``
    parameter of all calls to ``torch.empty`` while active.
    """
    global _active_count
    with _active_count_lock:
        _active_count += 1
        reset_token = _zero_length_init_active.set(True)
        if _active_count == 1:
            torch.empty = _torch_empty_substitute
            torch.ones = _torch_empty_substitute
    try:
        yield
    finally:
        with _active_count_lock:
            _active_count -= 1
            if _active_count == 0:
                torch.empty = _torch_empty
                torch.ones = _torch_empty ## TODO: Fix this, as this likely will cause issues with non-persistent buffers
            _zero_length_init_active.reset(reset_token)


_zero_length_init_active = contextvars.ContextVar(
    "_zero_length_init_active", default=False
)
_active_count: int = 0
_active_count_lock = threading.Lock()
_torch_empty: typing.Callable = torch.empty
_torch_ones: typing.Callable = torch.ones


@functools.wraps(_torch_empty)
def _torch_empty_substitute(*args, **kwargs):
    if _zero_length_init_active.get():
        if "size" in kwargs:
            kwargs["size"] = (*kwargs["size"], 0)
        elif len(args) > 1:
            # Varargs
            args = (*args, 0)
        elif len(args) == 1:
            # Either a single int or a single sequence
            dimension: typing.Union[
                typing.Sequence[int], typing.SupportsIndex
            ] = args[0]
            try:
                args = torch.Size((dimension, 0))
            except TypeError:
                # Single sequence argument
                args = ((*dimension, 0),)
    return _torch_empty(device = "cuda", requires_grad = False, *args, **kwargs)

# def vpe_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
#     param_data = param.data
#     if self.input_is_parallel:
#         tp_rank = get_tensor_model_parallel_rank()
#         shard_size = param_data.shape[0]
#         start_idx = tp_rank * shard_size
#         loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
#     assert param_data.shape == loaded_weight.shape
#     param_data.copy_(loaded_weight)
#
# def new_vpe_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
#     return loaded_weight
#
# def cpl_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
#     tp_rank = get_tensor_model_parallel_rank()
#     output_dim = getattr(param, "output_dim", None)
#     param_data = param.data
#     if output_dim is not None:
#         shard_size = param_data.shape[output_dim]
#         start_idx = tp_rank * shard_size
#         loaded_weight = loaded_weight.narrow(output_dim, start_idx,
#                                              shard_size)
#     assert param_data.shape == loaded_weight.shape
#     param_data.copy_(loaded_weight)
#
# def new_cpl_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
#     return loaded_weight
#
# def mcpl_weight_loader(self,
#                   param: Parameter,
#                   loaded_weight: torch.Tensor,
#                   loaded_shard_id: Optional[int] = None):
#     param_data = param.data
#     output_dim = getattr(param, "output_dim", None)
#     if loaded_shard_id is None:
#         # Loaded weight is already packed.
#         if output_dim is None:
#             assert param_data.shape == loaded_weight.shape
#             param_data.copy_(loaded_weight)
#             return
#         current_shard_offset = 0
#         shard_offsets = []
#         for i, output_size in enumerate(self.output_sizes):
#             shard_offsets.append((i, current_shard_offset, output_size))
#             current_shard_offset += output_size
#         packed_dim = getattr(param, "packed_dim", None)
#         for shard_id, shard_offset, shard_size in shard_offsets:
#             # If quantized, we need to adjust the offset and size to account
#             # for the packing.
#             if packed_dim == output_dim:
#                 shard_size = shard_size // param.pack_factor
#                 shard_offset = shard_offset // param.pack_factor
#             loaded_weight_shard = loaded_weight.narrow(
#                 output_dim, shard_offset, shard_size)
#             self.weight_loader(param, loaded_weight_shard, shard_id)
#         return
#
# def new_mcpl_weight_loader(self,
#                   param: Parameter,
#                   loaded_weight: torch.Tensor,
#                   loaded_shard_id: Optional[int] = None):
#     return loaded_weight
#
# def rpl_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
#     tp_rank = get_tensor_model_parallel_rank()
#     input_dim = getattr(param, "input_dim", None)
#     param_data = param.data
#     if input_dim is not None:
#         shard_size = param_data.shape[input_dim]
#         start_idx = tp_rank * shard_size
#         loaded_weight = loaded_weight.narrow(input_dim, start_idx,
#                                              shard_size)
#     assert param_data.shape == loaded_weight.shape
#     param_data.copy_(loaded_weight)
#
# def new_rpl_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
#     return loaded_weight
#
# def qkvpl_weight_loader(self,
#                   param: Parameter,
#                   loaded_weight: torch.Tensor,
#                   loaded_shard_id: Optional[str] = None):
#     param_data = param.data
#     output_dim = getattr(param, "output_dim", None)
#     if loaded_shard_id is None:
#         # Loaded weight is already packed.
#         if output_dim is None:
#             assert param_data.shape == loaded_weight.shape
#             param_data.copy_(loaded_weight)
#             return
#         shard_offsets = [
#             # (shard_id, shard_offset, shard_size)
#             ("q", 0, self.total_num_heads * self.head_size),
#             ("k", self.total_num_heads * self.head_size,
#              self.total_num_kv_heads * self.head_size),
#             ("v", (self.total_num_heads + self.total_num_kv_heads) *
#              self.head_size, self.total_num_kv_heads * self.head_size),
#         ]
#         packed_dim = getattr(param, "packed_dim", None)
#         for shard_id, shard_offset, shard_size in shard_offsets:
#             # If quantized, we need to adjust the offset and size to account
#             # for the packing.
#             if packed_dim == output_dim:
#                 shard_size = shard_size // param.pack_factor
#                 shard_offset = shard_offset // param.pack_factor
#             loaded_weight_shard = loaded_weight.narrow(
#                 output_dim, shard_offset, shard_size)
#             self.weight_loader(param, loaded_weight_shard, shard_id)
#         return
#
#     tp_rank = get_tensor_model_parallel_rank()
#     assert loaded_shard_id in ["q", "k", "v"]
#     if output_dim is not None:
#         if loaded_shard_id == "q":
#             shard_offset = 0
#             shard_size = self.num_heads * self.head_size
#         elif loaded_shard_id == "k":
#             shard_offset = self.num_heads * self.head_size
#             shard_size = self.num_kv_heads * self.head_size
#         elif loaded_shard_id == "v":
#             shard_offset = (self.num_heads +
#                             self.num_kv_heads) * self.head_size
#             shard_size = self.num_kv_heads * self.head_size
#         # If quantized, we need to adjust the offset and size to account
#         # for the packing.
#         packed_dim = getattr(param, "packed_dim", None)
#         if packed_dim == output_dim:
#             shard_size = shard_size // param.pack_factor
#             shard_offset = shard_offset // param.pack_factor
#         param_data = param_data.narrow(output_dim, shard_offset,
#                                        shard_size)
#         if loaded_shard_id == "q":
#             shard_id = tp_rank
#         else:
#             shard_id = tp_rank // self.num_kv_head_replicas
#         start_idx = shard_id * shard_size
#         loaded_weight = loaded_weight.narrow(output_dim, start_idx,
#                                              shard_size)
#     else:
#         ignore_warning = getattr(param, "ignore_warning", False)
#         if not ignore_warning:
#             logger.warning(
#                 "Loading a weight without `output_dim` attribute in "
#                 "QKVParallelLinear, assume the weight is the same "
#                 "for all partitions.")
#     assert param_data.shape == loaded_weight.shape
#     param_data.copy_(loaded_weight)
#
# def new_qkvpl_weight_loader(self,
#                   param: Parameter,
#                   loaded_weight: torch.Tensor,
#                   loaded_shard_id: Optional[str] = None):
#
#     return loaded_weight
#
# def sa_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
#     param_data = param.data
#     if self.input_is_parallel:
#         tp_rank = get_tensor_model_parallel_rank()
#         shard_size = param_data.shape[0]
#         start_idx = tp_rank * shard_size
#         loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
#     assert param_data.shape == loaded_weight.shape
#     param_data.copy_(loaded_weight)
#
# def new_sa_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
#     return loaded_weight