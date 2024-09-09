import argparse
import dataclasses
import io
import os
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import BinaryIO, Generator, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config import ModelConfig, ParallelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.utils import FlexibleArgumentParser

tensorizer_error_msg = None

try:
    from tensorizer import (DecryptionParams, EncryptionParams,
                            TensorDeserializer, TensorSerializer)
    from tensorizer.stream_io import open_stream
    from tensorizer.utils import (convert_bytes, get_mem_usage,
                                  no_init_or_tensor)

    _read_stream, _write_stream = (partial(
        open_stream,
        mode=mode,
    ) for mode in ("rb", "wb+"))
except ImportError as e:
    tensorizer_error_msg = str(e)

__all__ = [
    'EncryptionParams', 'DecryptionParams', 'TensorDeserializer',
    'TensorSerializer', 'open_stream', 'convert_bytes', 'get_mem_usage',
    'no_init_or_tensor', 'TensorizerConfig'
]

logger = init_logger(__name__)


@dataclass
class TensorizerConfig:
    tensorizer_uri: str
    vllm_tensorized: Optional[bool] = False
    verify_hash: Optional[bool] = False
    num_readers: Optional[int] = None
    encryption_keyfile: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_endpoint: Optional[str] = None
    model_class: Optional[Type[torch.nn.Module]] = None
    hf_config: Optional[PretrainedConfig] = None
    dtype: Optional[Union[str, torch.dtype]] = None
    _is_sharded: bool = False

    def __post_init__(self):
        # check if the configuration is for a sharded vLLM model
        self._is_sharded = isinstance(self.tensorizer_uri, str) \
            and re.search(r'%0\dd', self.tensorizer_uri) is not None

    def _construct_tensorizer_args(self) -> "TensorizerArgs":
        tensorizer_args = {
            "tensorizer_uri": self.tensorizer_uri,
            "vllm_tensorized": self.vllm_tensorized,
            "verify_hash": self.verify_hash,
            "num_readers": self.num_readers,
            "encryption_keyfile": self.encryption_keyfile,
            "s3_access_key_id": self.s3_access_key_id,
            "s3_secret_access_key": self.s3_secret_access_key,
            "s3_endpoint": self.s3_endpoint,
        }
        return TensorizerArgs(**tensorizer_args)  # type: ignore

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        if parallel_config.tensor_parallel_size > 1 \
            and not self._is_sharded:
            raise ValueError(
                "For a sharded model, tensorizer_uri should include a"
                " string format template like '%04d' to be formatted"
                " with the rank of the shard")

    def verify_with_model_config(self, model_config: "ModelConfig") -> None:
        if (model_config.quantization is not None
                and self.tensorizer_uri is not None):
            logger.warning(
                "Loading a model using Tensorizer with quantization on vLLM"
                " is unstable and may lead to errors.")

    def open_stream(self, tensorizer_args: Optional["TensorizerArgs"] = None):
        if tensorizer_args is None:
            tensorizer_args = self._construct_tensorizer_args()

        return open_stream(self.tensorizer_uri,
                           **tensorizer_args.stream_params)


def load_with_tensorizer(tensorizer_config: TensorizerConfig,
                         **extra_kwargs) -> nn.Module:
    tensorizer = TensorizerAgent(tensorizer_config, **extra_kwargs)
    return tensorizer.deserialize()


@dataclass
class TensorizerArgs:
    tensorizer_uri: Union[io.BufferedIOBase, io.RawIOBase, BinaryIO, str,
                          bytes, os.PathLike, int]
    vllm_tensorized: Optional[bool] = False
    verify_hash: Optional[bool] = False
    num_readers: Optional[int] = None
    encryption_keyfile: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_endpoint: Optional[str] = None
    """
  Args for the TensorizerAgent class. These are used to configure the behavior 
  of the TensorDeserializer when loading tensors from a serialized model.
  
  Args:
      tensorizer_uri: Path to serialized model tensors. Can be a local file 
          path or a S3 URI.
      vllm_tensorized: If True, indicates that the serialized model is a 
          vLLM model. This is used to determine the behavior of the 
          TensorDeserializer when loading tensors from a serialized model.
          It is far faster to deserialize a vLLM model as it utilizes
          tensorizer's optimized GPU loading. Note that this is now
          deprecated, as serialized vLLM models are now automatically
          inferred as vLLM models.
      verify_hash: If True, the hashes of each tensor will be verified against 
          the hashes stored in the metadata. A `HashMismatchError` will be 
          raised if any of the hashes do not match.
      num_readers: Controls how many threads are allowed to read concurrently
          from the source file. Default is `None`, which will dynamically set
          the number of readers based on the number of available 
          resources and model size. This greatly increases performance.
      encryption_keyfile: File path to a binary file containing a  
          binary key to use for decryption. `None` (the default) means 
          no decryption. See the example script in 
          examples/tensorize_vllm_model.py. 
      s3_access_key_id: The access key for the S3 bucket. Can also be set via
          the S3_ACCESS_KEY_ID environment variable.
      s3_secret_access_key: The secret access key for the S3 bucket. Can also
          be set via the S3_SECRET_ACCESS_KEY environment variable.
      s3_endpoint: The endpoint for the S3 bucket. Can also be set via the
          S3_ENDPOINT_URL environment variable.
  """

    def __post_init__(self):
        self.file_obj = self.tensorizer_uri
        self.s3_access_key_id = self.s3_access_key_id or envs.S3_ACCESS_KEY_ID
        self.s3_secret_access_key = (self.s3_secret_access_key
                                     or envs.S3_SECRET_ACCESS_KEY)
        self.s3_endpoint = self.s3_endpoint or envs.S3_ENDPOINT_URL
        self.stream_params = {
            "s3_access_key_id": self.s3_access_key_id,
            "s3_secret_access_key": self.s3_secret_access_key,
            "s3_endpoint": self.s3_endpoint,
        }

        self.deserializer_params = {
            "verify_hash": self.verify_hash,
            "encryption": self.encryption_keyfile,
            "num_readers": self.num_readers
        }

        if self.encryption_keyfile:
            with open_stream(
                    self.encryption_keyfile,
                    **self.stream_params,
            ) as stream:
                key = stream.read()
                decryption_params = DecryptionParams.from_key(key)
                self.deserializer_params['encryption'] = decryption_params

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Tensorizer CLI arguments"""

        # Tensorizer options arg group
        group = parser.add_argument_group(
            'tensorizer options',
            description=('Options for configuring the behavior of the'
                         ' tensorizer deserializer when '
                         'load_format=tensorizer is specified when '
                         'initializing an LLMEngine, either via the CLI '
                         'when running the vLLM OpenAI inference server '
                         'with a JSON string passed to '
                         '--model-loader-extra-config or as arguments given '
                         'to TensorizerConfig when passed to '
                         'model_loader_extra_config in the constructor '
                         'for LLMEngine.'))

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
            default=None,
            type=int,
            help="Controls how many threads are allowed to read concurrently "
            "from the source file. Default is `None`, which will dynamically "
            "set the number of readers based on the available resources "
            "and model size. This greatly increases performance.")
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
        attrs = [attr.name for attr in dataclasses.fields(cls)]
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
    in vllm/model_executor/model_loader/weight_utils.py
    """

    def __init__(self, tensorizer_config: TensorizerConfig,
                 quant_config: QuantizationConfig, **extra_kwargs):
        if tensorizer_error_msg is not None:
            raise ImportError(
                "Tensorizer is not installed. Please install tensorizer "
                "to use this feature with `pip install vllm[tensorizer]`. "
                "Error message: {}".format(tensorizer_error_msg))

        self.tensorizer_config = tensorizer_config
        self.tensorizer_args = (
            self.tensorizer_config._construct_tensorizer_args())
        self.extra_kwargs = extra_kwargs
        if extra_kwargs.get("quant_config", None) is not None:
            self.quant_config = extra_kwargs["quant_config"]
        else:
            self.quant_config = quant_config
        self.model = self._init_model()

    def _init_model(self):
        assert self.tensorizer_config.hf_config is not None
        model_args = self.tensorizer_config.hf_config
        model_args.torch_dtype = self.tensorizer_config.dtype
        assert self.tensorizer_config.model_class is not None
        with no_init_or_tensor():
            return self.tensorizer_config.model_class(
                config=model_args,
                quant_config=self.quant_config,
                **self.extra_kwargs)

    def _resize_lora_embeddings(self):
        """Modify LoRA embedding layers to use bigger tensors
        to allow for adapter added tokens."""
        for child in self.model.modules():
            if (isinstance(child, VocabParallelEmbedding)
                    and child.weight.shape[0] <
                    child.num_embeddings_per_partition):
                new_weight = torch.empty(child.num_embeddings_per_partition,
                                         child.embedding_dim,
                                         dtype=child.weight.dtype,
                                         device=child.weight.device)
                new_weight[:child.weight.shape[0]].copy_(child.weight.data)
                new_weight[child.weight.shape[0]:].fill_(0)
                child.weight.data = new_weight

    def _check_tensors_on_meta_device(self):
        for tensor in self.model.state_dict().values():
            if tensor.device.type == 'meta':
                raise ValueError(
                    "The serialized model contains tensors on the meta device,"
                    " indicating that some tensors were not loaded properly."
                    " Please check that the parameters of the model being"
                    " specified match that of the serialized model, such as"
                    " its quantization.")

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
        start = time.perf_counter()
        with _read_stream(
                self.tensorizer_config.tensorizer_uri,
                **self.tensorizer_args.stream_params
        ) as stream, TensorDeserializer(
                stream,
                dtype=self.tensorizer_config.dtype,
                device=f'cuda:{torch.cuda.current_device()}',
                **self.tensorizer_args.deserializer_params) as deserializer:
            deserializer.load_into_module(self.model)
            end = time.perf_counter()

        total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
        duration = end - start
        per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
        after_mem = get_mem_usage()
        deserializer.close()
        logger.info("Deserialized %s in %0.2fs, %s/s", total_bytes_str,
                    end - start, per_second)
        logger.info("Memory usage before: %s", before_mem)
        logger.info("Memory usage after: %s", after_mem)

        self._check_tensors_on_meta_device()
        self._resize_lora_embeddings()
        del self.model.vllm_tensorized_marker
        return self.model.eval()


def tensorizer_weights_iterator(
    tensorizer_args: "TensorizerArgs"
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    logger.warning(
        "Deserializing HuggingFace models is not optimized for "
        "loading on vLLM, as tensorizer is forced to load to CPU. "
        "Consider deserializing a vLLM model instead for faster "
        "load times. See the examples/tensorize_vllm_model.py example "
        "script for serializing vLLM models.")

    deserializer_args = tensorizer_args.deserializer_params
    stream_params = tensorizer_args.stream_params
    stream = open_stream(tensorizer_args.tensorizer_uri, **stream_params)
    with TensorDeserializer(stream, **deserializer_args,
                            device="cpu") as state:
        for name, param in state.items():
            yield name, param
    del state


def is_vllm_tensorized(tensorizer_config: "TensorizerConfig") -> bool:
    """
    Infer if the model is a vLLM model by checking the weights for
    a vLLM tensorized marker.

    Args:
        tensorizer_config: The TensorizerConfig object containing the
            tensorizer_uri to the serialized model.

    Returns:
        bool: True if the model is a vLLM model, False otherwise.
    """
    tensorizer_args = tensorizer_config._construct_tensorizer_args()
    deserializer = TensorDeserializer(open_stream(
        tensorizer_args.tensorizer_uri, **tensorizer_args.stream_params),
                                      **tensorizer_args.deserializer_params,
                                      lazy_load=True)
    if tensorizer_config.vllm_tensorized:
        logger.warning(
            "Please note that newly serialized vLLM models are automatically "
            "inferred as vLLM models, so setting vllm_tensorized=True is "
            "only necessary for models serialized prior to this change.")
        return True
    if (".vllm_tensorized_marker" in deserializer):
        return True
    return False


def serialize_vllm_model(
    model: nn.Module,
    tensorizer_config: TensorizerConfig,
) -> nn.Module:
    model.register_parameter(
        "vllm_tensorized_marker",
        nn.Parameter(torch.tensor((1, ), device="meta"), requires_grad=False))
    tensorizer_args = tensorizer_config._construct_tensorizer_args()

    encryption_params = None
    if (keyfile := tensorizer_config.encryption_keyfile) is not None:
        with open(keyfile, "rb") as f:
            key = f.read()
        encryption_params = EncryptionParams(key=key)

    output_file = tensorizer_args.tensorizer_uri
    if tensorizer_config._is_sharded:
        from vllm.distributed import get_tensor_model_parallel_rank
        output_file = output_file % get_tensor_model_parallel_rank()

    with _write_stream(output_file, **tensorizer_args.stream_params) as stream:
        serializer = TensorSerializer(stream, encryption=encryption_params)
        serializer.write_module(model)
        serializer.close()
    logger.info("Successfully serialized model to %s", str(output_file))
    return model


def tensorize_vllm_model(engine_args: EngineArgs,
                         tensorizer_config: TensorizerConfig,
                         generate_keyfile: bool = True):
    """Utility to load a model and then serialize it with Tensorizer

       Intended to be used separately from running a vLLM server since it
       creates its own Engine instance.
    """
    engine_config = engine_args.create_engine_config()
    tensorizer_config.verify_with_model_config(engine_config.model_config)
    tensorizer_config.verify_with_parallel_config(
        engine_config.parallel_config)

    # generate the encryption key before creating the engine to support sharding
    if generate_keyfile and (keyfile :=
                             tensorizer_config.encryption_keyfile) is not None:
        encryption_params = EncryptionParams.random()
        with _write_stream(
                keyfile,
                s3_access_key_id=tensorizer_config.s3_access_key_id,
                s3_secret_access_key=tensorizer_config.s3_secret_access_key,
                s3_endpoint=tensorizer_config.s3_endpoint,
        ) as stream:
            stream.write(encryption_params.key)

    engine = LLMEngine.from_engine_args(engine_args)
    if tensorizer_config._is_sharded:
        # if the engine is a distributed engine (for tensor parallel) then each
        # worker shard needs to serialize its part of the model.
        engine.model_executor._run_workers(
            "save_tensorized_model",
            tensorizer_config=tensorizer_config,
        )
    else:
        # with a single worker, we can get to the underlying model directly
        serialize_vllm_model(
            engine.model_executor.driver_worker.model_runner.model,
            tensorizer_config,
        )
