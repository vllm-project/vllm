import argparse
import dataclasses
import io
import os
import time
import typing
import warnings
from dataclasses import dataclass
from typing import Optional, Type, Union

from tensorizer import DecryptionParams, TensorDeserializer, stream_io
from tensorizer.utils import convert_bytes, get_mem_usage, no_init_or_tensor
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def load_with_tensorizer(model_cls: Type[nn.Module],
                         model_config: ModelConfig) -> nn.Module:
    tensorizer = TensorizerAgent(model_cls, model_config)
    return tensorizer.deserialize()


def _is_vllm_model(model_config: ModelConfig = None,
                   file_uri: Optional[str] = None) -> bool:
    if file_uri:
        return "vllm" in file_uri
    else:
        return "vllm" in model_config.tensorizer_args.tensorizer_uri


class ParameterizedLoadFormat(str):
    __slots__ = "params"


class PerformanceWarning(UserWarning):

    def __str__(self):
        return (f"{super().__str__()}"
                " (set the VLLM_SILENCE_PERFORMANCE_WARNINGS"
                " environment variable to hide this)")


if (os.getenv("VLLM_SILENCE_PERFORMANCE_WARNINGS", "").lower()
        not in ("", "0", "n", "no", "off", "disable")):
    warnings.simplefilter("ignore", category=PerformanceWarning)


@dataclass
class TensorizerArgs:
    tensorizer_uri: Union[io.BufferedIOBase, io.RawIOBase, typing.BinaryIO,
                          str, bytes, os.PathLike, int]
    verify_hash: bool = False
    encryption_keyfile: Optional[str] = None
    num_readers: Optional[int] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_endpoint: Optional[str] = None
    """
  Args for the TensorizerAgent class. These are used to configure the behavior 
  of the TensorDeserializer when loading tensors from a serialized model.
  
  Args:
      tensorizer_uri: Path to serialized model tensors. Can be a local file 
          path or a S3 URI.
      verify_hash: If True, the hashes of each tensor will be verified against 
          the hashes stored in the metadata. A `HashMismatchError` will be 
          raised if any of the hashes do not match.
      encryption_keyfile: File path to a binary file containing a  
          binary key to use for decryption. `None` (the default) means 
          no decryption. See the example script in 
          examples/tensorize_vllm_model.py. 
      num_readers: Controls how many threads are allowed to read concurrently
          from the source file. Default is 1. This greatly increases
          performance.
      s3_access_key_id: The access key for the S3 bucket. Can also be set via
          the S3_ACCESS_KEY_ID environment variable.
      s3_secret_access_key: The secret access key for the S3 bucket. Can also
          be set via the S3_SECRET_ACCESS_KEY environment variable.
      s3_endpoint: The endpoint for the S3 bucket. Can also be set via the
          S3_ENDPOINT_URL environment variable.
        
  """

    def __post_init__(self):
        self.file_obj = self.tensorizer_uri
        self.s3_access_key_id = (self.s3_access_key_id
                                 or os.environ.get("S3_ACCESS_KEY_ID")) or None
        self.s3_secret_access_key = (
            self.s3_secret_access_key
            or os.environ.get("S3_SECRET_ACCESS_KEY")) or None
        self.s3_endpoint = (self.s3_endpoint
                            or os.environ.get("S3_ENDPOINT_URL")) or None
        self.stream_params = {
            "s3_access_key_id": self.s3_access_key_id,
            "s3_secret_access_key": self.s3_secret_access_key,
            "s3_endpoint": self.s3_endpoint,
        }

        # Omitting self.dtype and self.device as this behaves weirdly
        self.deserializer_params = {
            "verify_hash": self.verify_hash,
            "encryption": self.encryption_keyfile,
            "num_readers": self.num_readers
        }
        if self.encryption_keyfile:
            with stream_io.open_stream(
                    self.encryption_keyfile,
                    **self.stream_params,
            ) as stream:
                key = stream.read()
                decryption_params = DecryptionParams.from_key(key)
                self.deserializer_params['encryption'] = decryption_params

    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Tensorizer CLI arguments"""

        # Create the argument group
        group = parser.add_argument_group(
            'tensorizer options',
            description=('Options for configuring the behavior of the'
                         ' tensorizer deserializer when '
                         '--load-format=tensorizer'))

        group.add_argument(
            "--tensorizer-uri",
            help="Path to serialized model tensors. Can be a local file path,"
            " or an HTTP(S) or S3 URI.",
        )
        group.add_argument(
            "--verify-hash",
            action="store_true",
            help="If enabled, the hashes of each tensor will be verified"
            " against the hashes stored in the file metadata. An exception"
            " will be raised if any of the hashes do not match.",
        )
        group.add_argument(
            "--encryption-keyfile",
            default=None,
            help="The file path to a binary file containing a binary key to "
            "use for decryption. Can be a file path or S3 network URI.")
        group.add_argument(
            "--num-readers",
            default=1,
            type=int,
            help="Controls how many threads are allowed to read concurrently "
                 "from the source file.")
        group.add_argument(
            "--s3-access-key-id",
            default=None,
            help="The access key for the S3 bucket. Can also be set via the "
            "S3_ACCESS_KEY_ID environment variable.",
        )
        group.add_argument(
            "--s3-secret-access-key",
            default=None,
            help="The secret access key for the S3 bucket. Can also be set via "
            "the S3_SECRET_ACCESS_KEY environment variable.",
        )
        group.add_argument(
            "--s3-endpoint",
            default=None,
            help="The endpoint for the S3 bucket. Can also be set via the "
            "S3_ENDPOINT_URL environment variable.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "TensorizerArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        tensorizer_args = cls(**{
            attr: getattr(args, attr)
            for attr in attrs if hasattr(args, attr)
        })
        return tensorizer_args


class TensorizerAgent:
    """
    A class for performing tensorizer deserializations specifically for
    vLLM models using plaid_mode. Uses TensorizerArgs to configure the
    behavior of the TensorDeserializer when loading tensors from a serialized
    model. For deserializations of HuggingFace models, TensorDeserializer is
    instead used as an iterator directly in the func hf_model_weights_iterator
    in vllm/model_executor/weight_utils.py
    """

    def __init__(
        self,
        model_cls: Type[nn.Module],
        model_config: ModelConfig,
    ):
        self.model_cls = model_cls
        self.model_config = model_config
        self.tensorizer_args = self.model_config.tensorizer_args
        self.model = self._init_model()

    def _init_model(self):
        model_args = self.model_config.hf_config
        model_args.torch_dtype = self.model_config.dtype
        with no_init_or_tensor():
            return self.model_cls(model_args)

    def deserialize(self):
        """
        Deserialize the model using the TensorDeserializer. This method is
        specifically for vLLM models using tensorizer's plaid_mode.

        The deserializer makes use of tensorizer_args.stream_params
        to configure the behavior of the stream when loading tensors from a
        serialized model. The deserializer_params are used to configure the
        behavior of the TensorDeserializer when loading tensors themselves.
        Documentation on these params can be found in TensorizerArgs

        Returns:
            nn.Module: The deserialized model.
        """
        before_mem = get_mem_usage()
        # Lazy load the tensors from S3 into the model.
        start = time.perf_counter()
        with stream_io.open_stream(
                self.tensorizer_args.tensorizer_uri,
                mode="rb",
                **self.tensorizer_args.stream_params,
        ) as stream, TensorDeserializer(
                stream,
                dtype=self.model_config.dtype,
                **self.tensorizer_args.deserializer_params) as deserializer:
            deserializer.load_into_module(self.model)
            end = time.perf_counter()

        total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
        duration = end - start
        per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
        after_mem = get_mem_usage()
        deserializer.close()
        logger.info(f"Deserialized {total_bytes_str} in "
                    f"{end - start:0.2f}s, {per_second}/s")
        logger.info(f"Memory usage before: {before_mem}")
        logger.info(f"Memory usage after: {after_mem}")

        return self.model.eval()
