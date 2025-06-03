# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import contextlib
import contextvars
import dataclasses
import io
import json
import os
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from typing import Any, BinaryIO, Optional, Union

import regex as re
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config import (ModelConfig, ParallelConfig, VllmConfig,
                         set_current_vllm_config)
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.utils import FlexibleArgumentParser, PlaceholderModule

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
except ImportError:
    tensorizer = PlaceholderModule("tensorizer")
    DecryptionParams = tensorizer.placeholder_attr("DecryptionParams")
    EncryptionParams = tensorizer.placeholder_attr("EncryptionParams")
    TensorDeserializer = tensorizer.placeholder_attr("TensorDeserializer")
    TensorSerializer = tensorizer.placeholder_attr("TensorSerializer")
    open_stream = tensorizer.placeholder_attr("stream_io.open_stream")
    convert_bytes = tensorizer.placeholder_attr("utils.convert_bytes")
    get_mem_usage = tensorizer.placeholder_attr("utils.get_mem_usage")
    no_init_or_tensor = tensorizer.placeholder_attr("utils.no_init_or_tensor")

    _read_stream = tensorizer.placeholder_attr("_read_stream")
    _write_stream = tensorizer.placeholder_attr("_write_stream")

__all__ = [
    'EncryptionParams', 'DecryptionParams', 'TensorDeserializer',
    'TensorSerializer', 'open_stream', 'convert_bytes', 'get_mem_usage',
    'no_init_or_tensor', 'TensorizerConfig'
]

logger = init_logger(__name__)


class MetaTensorMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if func._schema.name == "aten::empty" and "device" not in kwargs:
            kwargs["device"] = "meta"

        return func(*args, **kwargs)


def meta_tensor_mode(loading_code=None, ):

    if loading_code is None:
        return _NoInitOrTensorImpl.context_manager()
    elif callable(loading_code):
        with _NoInitOrTensorImpl.context_manager():
            return loading_code()
    else:
        raise TypeError(
            "expected a callable to evaluate,"
            " or None if being used as a context manager;"
            f' got an object of type "{type(loading_code).__name__}" instead.')


class _NoInitOrTensorImpl:
    _MODULES = (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)
    _MODULE_ORIGINALS = tuple((m, m.reset_parameters) for m in _MODULES)

    is_active = contextvars.ContextVar("_NoInitOrTensorImpl.is_active",
                                       default=False)
    _count_active: int = 0
    _count_active_lock = threading.Lock()

    @classmethod
    @contextlib.contextmanager
    def context_manager(cls):
        if cls.is_active.get():
            yield
            return

        with cls._count_active_lock:
            cls._count_active += 1
            if cls._count_active == 1:
                for mod in cls._MODULES:
                    mod.reset_parameters = cls._disable(mod.reset_parameters)

        reset_token = cls.is_active.set(True)

        try:
            with MetaTensorMode():
                yield
        finally:
            cls.is_active.reset(reset_token)
            with cls._count_active_lock:
                cls._count_active -= 1
                if cls._count_active == 0:
                    for mod, original in cls._MODULE_ORIGINALS:
                        mod.reset_parameters = original

    @staticmethod
    def _disable(func):

        def wrapper(*args, **kwargs):
            if not _NoInitOrTensorImpl.is_active.get():
                return func(*args, **kwargs)

        return wrapper


@dataclass
class TensorizerConfig:
    tensorizer_uri: Union[str, None] = None
    vllm_tensorized: Optional[bool] = False
    verify_hash: Optional[bool] = False
    num_readers: Optional[int] = None
    encryption_keyfile: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_endpoint: Optional[str] = None
    model_class: Optional[type[torch.nn.Module]] = None
    hf_config: Optional[PretrainedConfig] = None
    dtype: Optional[Union[str, torch.dtype]] = None
    lora_dir: Optional[str] = None
    _is_sharded: bool = False

    def __post_init__(self):
        # check if the configuration is for a sharded vLLM model
        self._is_sharded = isinstance(self.tensorizer_uri, str) \
            and re.search(r'%0\dd', self.tensorizer_uri) is not None
        if not self.tensorizer_uri and not self.lora_dir:
            raise ValueError("tensorizer_uri must be provided.")
        if not self.tensorizer_uri and self.lora_dir:
            self.tensorizer_uri = f"{self.lora_dir}/adapter_model.tensors"
        assert self.tensorizer_uri is not None, ("tensorizer_uri must be "
                                                 "provided.")
        self.tensorizer_dir = os.path.dirname(self.tensorizer_uri)
        self.lora_dir = self.tensorizer_dir

    @classmethod
    def as_dict(cls, *args, **kwargs) -> dict[str, Any]:
        cfg = TensorizerConfig(*args, **kwargs)
        return dataclasses.asdict(cfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

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
          path or a S3 URI. This is a required field unless lora_dir is 
          provided and the config is meant to be used for the
          `tensorize_lora_adapter` function.
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
          examples/others/tensorize_vllm_model.py. 
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
            type=str,
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
            type=str,
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
            type=str,
            default=None,
            help="The access key for the S3 bucket. Can also be set via the "
            "S3_ACCESS_KEY_ID environment variable.",
        )
        group.add_argument(
            "--s3-secret-access-key",
            type=str,
            default=None,
            help="The secret access key for the S3 bucket. Can also be set via "
            "the S3_SECRET_ACCESS_KEY environment variable.",
        )
        group.add_argument(
            "--s3-endpoint",
            type=str,
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


def _check_tensors_on_meta_device(model: nn.Module) -> None:
    for tensor in model.state_dict().values():
        if tensor.device.type == 'meta':
            raise ValueError(
                "The serialized model contains tensors on the meta device,"
                " indicating that some tensors were not loaded properly."
                " Please check that the parameters of the model being"
                " specified match that of the serialized model, such as"
                " its quantization.")


def _resize_lora_embeddings(model: nn.Module):
    """Modify LoRA embedding layers to use bigger tensors
    to allow for adapter added tokens."""
    for child in model.modules():
        if (isinstance(child, VocabParallelEmbedding) and child.weight.shape[0]
                < child.num_embeddings_per_partition):
            new_weight = torch.empty(child.num_embeddings_per_partition,
                                     child.embedding_dim,
                                     dtype=child.weight.dtype,
                                     device=child.weight.device)
            new_weight[:child.weight.shape[0]].copy_(child.weight.data)
            new_weight[child.weight.shape[0]:].fill_(0)
            child.weight.data = new_weight


def init_tensorizer_model(tensorizer_config: TensorizerConfig,
                          vllm_config: VllmConfig) -> nn.Module:
    assert tensorizer_config.hf_config is not None
    model_args = tensorizer_config.hf_config
    model_args.torch_dtype = tensorizer_config.dtype
    assert tensorizer_config.model_class is not None
    # TODO: Do we need to consider old-style model class?
    with meta_tensor_mode(), set_current_vllm_config(vllm_config,
                                                     check_compile=True):
        return tensorizer_config.model_class(vllm_config=vllm_config)


def deserialize_tensorizer_model(model: nn.Module,
                                 tensorizer_config: TensorizerConfig) -> None:
    tensorizer_args = tensorizer_config._construct_tensorizer_args()
    before_mem = get_mem_usage()
    start = time.perf_counter()
    with _read_stream(
            tensorizer_config.tensorizer_uri,
            **tensorizer_args.stream_params) as stream, TensorDeserializer(
                stream,
                dtype=tensorizer_config.dtype,
                device=f'cuda:{torch.cuda.current_device()}',
                **tensorizer_args.deserializer_params) as deserializer:
        deserializer.load_into_module(model)
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

    _check_tensors_on_meta_device(model)
    _resize_lora_embeddings(model)
    del model.vllm_tensorized_marker


def tensorizer_weights_iterator(
    tensorizer_args: "TensorizerArgs"
) -> Generator[tuple[str, torch.Tensor], None, None]:
    logger.warning("Deserializing HuggingFace models is not optimized for "
                   "loading on vLLM, as tensorizer is forced to load to CPU. "
                   "Consider deserializing a vLLM model instead for faster "
                   "load times. See the "
                   "examples/others/tensorize_vllm_model.py example script "
                   "for serializing vLLM models.")

    deserializer_args = tensorizer_args.deserializer_params
    stream_params = tensorizer_args.stream_params
    stream = open_stream(tensorizer_args.tensorizer_uri, **stream_params)
    with TensorDeserializer(stream, **deserializer_args,
                            device="cpu") as state:
        yield from state.items()
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
    return ".vllm_tensorized_marker" in deserializer


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

    from vllm import LLMEngine
    from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

    if not envs.VLLM_USE_V1:
        engine = LLMEngine.from_engine_args(engine_args)
        engine.model_executor.collective_rpc(
            "save_tensorized_model",
            kwargs=dict(tensorizer_config=tensorizer_config),
        )
    else:
        engine = V1LLMEngine.from_vllm_config(engine_config)
        engine.collective_rpc(
            "save_tensorized_model",
            kwargs=dict(tensorizer_config=tensorizer_config),
        )


def tensorize_lora_adapter(lora_path: str,
                           tensorizer_config: TensorizerConfig):
    """
    Uses tensorizer to serialize a LoRA adapter. Assumes that the files
    needed to load a LoRA adapter are a safetensors-format file called
    adapter_model.safetensors and a json config file called adapter_config.json.

    Serializes the files in the tensorizer_config.lora_dir
    """
    import safetensors

    from vllm.lora.utils import get_adapter_absolute_path

    lora_dir = get_adapter_absolute_path(lora_path)

    tensor_path = config_path = ""

    for file in os.listdir(lora_dir):
        if file.startswith("adapter_model"):
            tensor_path = lora_dir + "/" + file
        if file.startswith("adapter_config"):
            config_path = lora_dir + "/" + file
        if tensor_path and config_path:
            break

    if tensor_path.endswith(".safetensors"):
        tensors = safetensors.torch.load_file(tensor_path)
    elif tensor_path.endswith(".bin"):
        tensors = torch.load(tensor_path)
    else:
        raise ValueError("Unsupported file: %s", tensor_path)

    with open(config_path) as f:
        config = json.load(f)

    tensorizer_args = tensorizer_config._construct_tensorizer_args()

    with open_stream(f"{tensorizer_config.lora_dir}/adapter_config.json",
                     mode="wb+",
                     **tensorizer_args.stream_params) as f:

        f.write(json.dumps(config).encode("utf-8"))

    lora_uri = (f"{tensorizer_config.lora_dir}"
                f"/adapter_model.tensors")
    with open_stream(lora_uri, mode="wb+",
                     **tensorizer_args.stream_params) as f:
        serializer = TensorSerializer(f)
        serializer.write_state_dict(tensors)
        serializer.close()

    logger.info("Successfully serialized LoRA files to %s",
                str(tensorizer_config.lora_dir))
