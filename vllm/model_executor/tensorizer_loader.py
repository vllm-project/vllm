import contextlib
import contextvars
import dataclasses
import functools
import threading
import time
import typing
from typing import Optional
from typing import Type, Union, Any, Callable
import io
import os
import argparse


import torch
from dataclasses import dataclass
from tensorizer import TensorDeserializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear

logger = init_logger(__name__)

def load_with_tensorizer(model_cls: Type[nn.Module], model_config: ModelConfig) -> nn.Module:
    tensorizer = TensorizerAgent(model_cls, model_config)
    return tensorizer.deserialize()

def _is_vllm_model(model_config: ModelConfig) -> bool:
    return "vllm" in model_config.tensorizer_args.tensorizer_uri

def _make_model_contiguous(model: nn.Module):
    # Ensure tensors are saved in memory contiguously
    for param in model.parameters():
        param.data = param.data.contiguous()


@dataclass
class TensorizerArgs:
    tensorizer_uri: Union[
        io.BufferedIOBase,
        io.RawIOBase,
        typing.BinaryIO,
        str,
        bytes,
        os.PathLike,
        int,
    ]
    device: Optional[Union[torch.device, str]] = None
    dtype: Optional[torch.dtype] = None
    ## Commenting out serializer_encryption until I work out how I want to implement it
    # serializer_encryption: Optional[bool] = False
    lazy_load: bool = False
    plaid_mode_buffers: Optional[int] = None
    verify_hash: bool = False
    filter_func: Optional[Callable[[str], Union[bool, Any]]] = None
    deserializer_encryption_key: Optional[str] = None

    def __post_init__(self):
        self.file_obj = self.tensorizer_uri
        self.s3_access_key_id = os.environ.get("S3_ACCESS_KEY_ID") or None
        self.s3_secret_access_key = os.environ.get("S3_SECRET_ACCESS_KEY") or None
        self.s3_endpoint = os.environ.get("S3_ENDPOINT_URL") or None

        self.credentials = {
            "s3_access_key_id": self.s3_access_key_id,
            "s3_secret_access_key": self.s3_secret_access_key,
            "s3_endpoint": self.s3_endpoint,
        }
        self.serializer_params = {
            # Placeholder for now
        }


        # Omitting self.dtype and self.device as this behaves weirdly
        self.deserializer_params = {
            "filter_func": self.filter_func,
            "lazy_load": self.lazy_load,
            "plaid_mode": True if not self.device == "cpu" else False,
            "plaid_mode_buffers": self.plaid_mode_buffers,
            "verify_hash": self.verify_hash,
            "encryption": self.deserializer_encryption_key,
            # "dtype":self.dtype,
            # "device":self.device,
        }

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Tensorizer CLI arguments"""
        # TODO: Add support for encryption -- CLI args can be base64 encoded
        #       key/password for --serializer-encryption. Need to revist
        parser.add_argument(
            "--serializer-encryption",
            action="store_true",
            help="An `EncryptionParams` object holding a password or key"
            "to use for encryption. If None, no encryption will be used.",
        )
        parser.add_argument(
            "--lazy-load",
            action="store_true",
            help="If True, tensors will be loaded and cached when keys are"
            "accessed. If False, all tensors will be loaded into memory up"
            "front.",
        )
        parser.add_argument(
            "--tensorizer-uri",
            help="Path to serialized model tensors. Can be a local file path"
                 "or a S3 URI.",
        )
        parser.add_argument(
            "--plaid-mode-buffers",
            default=None,
            help="The number of buffers to use in plaid mode."
            "This is only used if ``plaid_mode=True``. These buffers"
            "are used to pipeline the loading and processing of tensors.",
        )
        parser.add_argument(
            "--verify-hash",
            action="store_true",
            help="If True, the hashes of each tensor will be verified"
            "against the hashes stored in the metadata. A `HashMismatchError`"
            "will be raised if any of the hashes do not match.",
        )
        parser.add_argument(
            "--deserializer-encryption-key",
            default=None,
            help="A `DecryptionParams` object holding a password or key"
            "to use for decryption. ``None`` (the default) means no decryption.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "TensorizerArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        tensorizer_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return tensorizer_args



class TensorizerAgent:
    def __init__(self, model_cls: Type[nn.Module],
                 model_config: ModelConfig,
                 ):
        self.model_cls = model_cls
        self.model_config = model_config
        self.tensorizer_args = self.model_config.tensorizer_args
        self.serialize_model = not self._verify_path_reachable()
        self.model = self._init_model()

    def _init_model(self):
        model_args = self.model_config.hf_config
        model_args.torch_dtype = self.model_config.dtype
        model = no_init_or_tensor(lambda: self.model_cls(*[model_args]))
        return model

    def _verify_path_reachable(self):
        if not self.tensorizer_args.tensorizer_uri.endswith(".tensors"):
            raise ValueError(f"download_dir {self.tensorizer_args.tensorizer_uri} must specify a .tensors "
                             f"file when load_format = tensorizer")

    def deserialize(self):
        before_mem = get_mem_usage()
        # Lazy load the tensors from S3 into the model.
        start = time.time()
        stream = stream_io.open_stream(self.tensorizer_args.tensorizer_uri, mode="rb", **self.tensorizer_args.credentials)
        deserializer = TensorDeserializer(stream, **self.deserialize_args)
        deserializer.load_into_module(self.model)
        self.model = self.model.to(dtype=self.model_config.dtype)
        end = time.time()

        # Brag about how fast we are.
        total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
        duration = end - start
        per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
        after_mem = get_mem_usage()
        deserializer.close()
        logger.info(
            f"Deserialized {total_bytes_str} in {end - start:0.2f}s, {per_second}/s"
        )
        logger.info(f"Memory usage before: {before_mem}")
        logger.info(f"Memory usage after: {after_mem}")

        return self.model.eval()

    # def serialize(self):
    #     with torch.device("cuda"):
    #         model = self.model_cls(self.model_config.hf_config)
    #     self.model_config.load_format = "auto"
    #     model.load_weights(
    #         self.model_config.model,
    #         self.model_config.download_dir,
    #         self.model_config.load_format,
    #         self.model_config.revision,
    #     )
    #     _make_model_contiguous(model)
    #     stream = stream_io.open_stream(self.tensorizer_args.download_dir, "wb", **self.credentials)
    #     serializer = TensorSerializer(stream, **self.serialize_args)
    #     logger.info(
    #         f"Serializing model tensors {self.model_config.model} to {self.tensorizer_args.download_dir}."
    #     )
    #     serializer.write_module(model)
    #     serializer.close()
    #     logger.info(
    #         f"Serialization complete. Running the previous command will deserialize the saved model weights."
    #     )
    #     return model.eval()


## Monkey patch for Parameter to ensure `requires_grad=False`
from torch.nn.parameter import Parameter

# Save the original __init__ method for later use
#original_new = Parameter.__new__

def _new(cls, data, requires_grad=False):
    return original_new(cls, data, requires_grad=requires_grad)

# Replace the original __init__ method with our new one
#Parameter.__new__ = _new

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