import argparse
import dataclasses
import io
import os
import time
import typing
from typing import Optional
from typing import Type, Union, Any, Callable

import torch
from dataclasses import dataclass
from tensorizer import TensorDeserializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

def load_with_tensorizer(model_cls: Type[nn.Module], model_config: ModelConfig) -> nn.Module:
    tensorizer = TensorizerAgent(model_cls, model_config)
    return tensorizer.deserialize()

def _is_vllm_model(model_config: ModelConfig = None,
                   file_uri: Optional[str] = None) -> bool:
    if file_uri:
        return "vllm" in file_uri
    else:
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
            "plaid_mode": True if _is_vllm_model(file_uri=self.file_obj) else False,
            "plaid_mode_buffers": self.plaid_mode_buffers,
            "verify_hash": self.verify_hash,
            "encryption": self.deserializer_encryption_key,
            "force_http": True
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
        #self.serialize_model = not self._verify_path_reachable()
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
        deserializer = TensorDeserializer(stream, **self.tensorizer_args.deserializer_params)
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