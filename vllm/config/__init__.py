# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa: F401
import ast
import copy
import enum
import hashlib
import inspect
import json
import os
import textwrap
import warnings
from contextlib import contextmanager
from dataclasses import InitVar, field, fields, is_dataclass, replace
from functools import cached_property, lru_cache
from importlib.util import find_spec
from typing import (TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol,
                    TypeVar, Union, cast, get_args)

import regex as re
import torch
from pydantic import (ConfigDict, SkipValidation, field_validator,
                      model_validator)
from pydantic.dataclasses import dataclass
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from typing_extensions import assert_never, runtime_checkable

import vllm.envs as envs
from vllm import version
from vllm.config.cache import (BlockSize, CacheConfig, CacheDType, MambaDType,
                               PrefixCachingHashAlgo)
from vllm.config.compilation import (CompilationConfig, CompilationLevel,
                                     CUDAGraphMode, PassConfig)
from vllm.config.kv_events import KVEventsConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.load import LoadConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import (MMCacheType, MMEncoderTPMode,
                                    MultiModalConfig)
from vllm.config.parallel import (DistributedExecutorBackend, EPLBConfig,
                                  ParallelConfig)
from vllm.config.scheduler import SchedulerConfig, SchedulerPolicy
from vllm.config.speculative import SpeculativeConfig
from vllm.config.utils import ConfigType, config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.transformers_utils.config import (
    ConfigFormat, get_config, get_hf_image_processor_config,
    get_hf_text_config, get_pooling_config,
    get_sentence_transformer_tokenizer_config, is_encoder_decoder,
    is_interleaved, maybe_override_with_speculators_target_model,
    try_get_generation_config, try_get_safetensors_metadata,
    try_get_tokenizer_config, uses_mrope)
from vllm.transformers_utils.runai_utils import (ObjectStorageModel,
                                                 is_runai_obj_uri)
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils import (DEFAULT_MAX_NUM_BATCHED_TOKENS,
                        STR_DUAL_CHUNK_FLASH_ATTN_VAL, LayerBlockType,
                        LazyLoader, common_broadcastable_dtype, random_uuid)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from transformers.configuration_utils import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
    import vllm.model_executor.models as me_models
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig)
    from vllm.v1.sample.logits_processor import LogitsProcessor

    HfOverrides = Union[dict, Callable[[type], type]]
else:
    DataclassInstance = Any
    PretrainedConfig = Any
    QuantizationConfig = Any
    QuantizationMethods = Any
    BaseModelLoader = Any
    LogitsProcessor = Any
    HfOverrides = Union[dict[str, Any], Callable[[type], type]]

    me_quant = LazyLoader("model_executor", globals(),
                          "vllm.model_executor.layers.quantization")
    me_models = LazyLoader("model_executor", globals(),
                           "vllm.model_executor.models")

logger = init_logger(__name__)
DataclassInstanceT = TypeVar("DataclassInstanceT", bound=DataclassInstance)

TaskOption = Literal["auto", "generate", "embedding", "embed", "classify",
                     "score", "reward", "transcription", "draft"]

_ResolvedTask = Literal["generate", "transcription", "encode", "embed",
                        "classify", "reward", "draft"]

RunnerOption = Literal["auto", "generate", "pooling", "draft"]

RunnerType = Literal["generate", "pooling", "draft"]

ConvertOption = Literal["auto", "none", "embed", "classify", "reward"]

ConvertType = Literal["none", "embed", "classify", "reward"]

_RUNNER_TASKS: dict[RunnerType, list[TaskOption]] = {
    "generate": ["generate", "transcription"],
    "pooling": ["embedding", "embed", "classify", "score", "reward"],
    "draft": ["draft"],
}

_RUNNER_CONVERTS: dict[RunnerType, list[ConvertType]] = {
    "generate": [],
    "pooling": ["embed", "classify", "reward"],
    "draft": [],
}

# Some model suffixes are based on auto classes from Transformers:
# https://huggingface.co/docs/transformers/en/model_doc/auto
# NOTE: Items higher on this list priority over lower ones
_SUFFIX_TO_DEFAULTS: list[tuple[str, tuple[RunnerType, ConvertType]]] = [
    ("ForCausalLM", ("generate", "none")),
    ("ForConditionalGeneration", ("generate", "none")),
    ("ChatModel", ("generate", "none")),
    ("LMHeadModel", ("generate", "none")),
    ("ForTextEncoding", ("pooling", "embed")),
    ("EmbeddingModel", ("pooling", "embed")),
    ("ForSequenceClassification", ("pooling", "classify")),
    ("ForAudioClassification", ("pooling", "classify")),
    ("ForImageClassification", ("pooling", "classify")),
    ("ForVideoClassification", ("pooling", "classify")),
    ("ClassificationModel", ("pooling", "classify")),
    ("ForRewardModeling", ("pooling", "reward")),
    ("RewardModel", ("pooling", "reward")),
    # Let other `*Model`s take priority
    ("Model", ("pooling", "embed")),
]


def iter_architecture_defaults():
    yield from _SUFFIX_TO_DEFAULTS


def try_match_architecture_defaults(
    architecture: str,
    *,
    runner_type: Optional[RunnerType] = None,
    convert_type: Optional[ConvertType] = None,
) -> Optional[tuple[str, tuple[RunnerType, ConvertType]]]:
    for suffix, (default_runner_type,
                 default_convert_type) in iter_architecture_defaults():
        if ((runner_type is None or runner_type == default_runner_type) and
            (convert_type is None or convert_type == default_convert_type)
                and architecture.endswith(suffix)):
            return suffix, (default_runner_type, default_convert_type)

    return None


@runtime_checkable
class SupportsHash(Protocol):

    def compute_hash(self) -> str:
        ...


class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> dict[str, str]:
        ...


class ModelImpl(str, enum.Enum):
    AUTO = "auto"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    TERRATORCH = "terratorch"


def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    """

    def pairwise(iterable):
        """
        Manually implement https://docs.python.org/3/library/itertools.html#itertools.pairwise

        Can be removed when Python 3.9 support is dropped.
        """
        iterator = iter(iterable)
        a = next(iterator, None)

        for b in iterator:
            yield a, b
            a = b

    try:
        cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
    except (OSError, KeyError, TypeError):
        # HACK: Python 3.13+ workaround - set missing __firstlineno__
        # Workaround can be removed after we upgrade to pydantic==2.12.0
        with open(inspect.getfile(cls)) as f:
            for i, line in enumerate(f):
                if f"class {cls.__name__}" in line and ":" in line:
                    cls.__firstlineno__ = i + 1
                    break
        cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (not isinstance(a, (ast.Assign, ast.AnnAssign))
                or not isinstance(b, ast.Expr)
                or not isinstance(b.value, ast.Constant)
                or not isinstance(b.value.value, str)):
            continue

        doc = inspect.cleandoc(b.value.value)

        # An assignment can have multiple targets (a = b = v), but an
        # annotated assignment only has one target.
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out


def is_init_field(cls: ConfigType, name: str) -> bool:
    return next(f for f in fields(cls) if f.name == name).init


TokenizerMode = Literal["auto", "slow", "mistral", "custom"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]


class LogprobsMode(enum.Enum):
    RAW_LOGITS = "raw_logits"
    RAW_LOGPROBS = "raw_logprobs"
    PROCESSED_LOGITS = "processed_logits"
    PROCESSED_LOGPROBS = "processed_logprobs"


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """Configuration for the model."""

    model: str = "Qwen/Qwen3-0.6B"
    """Name or path of the Hugging Face model to use. It is also used as the
    content for `model_name` tag in metrics output when `served_model_name` is
    not specified."""
    runner: RunnerOption = "auto"
    """The type of model runner to use. Each vLLM instance only supports one
    model runner, even if the same model can be used for multiple types."""
    convert: ConvertOption = "auto"
    """Convert the model using adapters defined in
    [vllm.model_executor.models.adapters][]. The most common use case is to
    adapt a text generation model to be used for pooling tasks."""
    task: Optional[TaskOption] = None
    """[DEPRECATED] The task to use the model for. If the model supports more
    than one model runner, this is used to select which model runner to run.

    Note that the model may support other tasks using the same model runner.
    """
    tokenizer: SkipValidation[str] = None  # type: ignore
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode = "auto"
    """Tokenizer mode:\n
    - "auto" will use the fast tokenizer if available.\n
    - "slow" will always use the slow tokenizer.\n
    - "mistral" will always use the tokenizer from `mistral_common`.\n
    - "custom" will use --tokenizer to select the preregistered tokenizer."""
    trust_remote_code: bool = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model
    and tokenizer."""
    dtype: Union[ModelDType, torch.dtype] = "auto"
    """Data type for model weights and activations:\n
    - "auto" will use FP16 precision for FP32 and FP16 models, and BF16
    precision for BF16 models.\n
    - "half" for FP16. Recommended for AWQ quantization.\n
    - "float16" is the same as "half".\n
    - "bfloat16" for a balance between precision and range.\n
    - "float" is shorthand for FP32 precision.\n
    - "float32" for FP32 precision."""
    seed: Optional[int] = None
    """Random seed for reproducibility. Initialized to None in V0, but
    initialized to 0 in V1."""
    hf_config_path: Optional[str] = None
    """Name or path of the Hugging Face config to use. If unspecified, model
    name or path will be used."""
    allowed_local_media_path: str = ""
    """Allowing API requests to read local images or videos from directories
    specified by the server file system. This is a security risk. Should only
    be enabled in trusted environments."""
    revision: Optional[str] = None
    """The specific model version to use. It can be a branch name, a tag name,
    or a commit id. If unspecified, will use the default version."""
    code_revision: Optional[str] = None
    """The specific revision to use for the model code on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    rope_scaling: dict[str, Any] = field(default_factory=dict)
    """RoPE scaling configuration. For example,
    `{"rope_type":"dynamic","factor":2.0}`."""
    rope_theta: Optional[float] = None
    """RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE
    theta improves the performance of the scaled model."""
    tokenizer_revision: Optional[str] = None
    """The specific revision to use for the tokenizer on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    max_model_len: SkipValidation[int] = None  # type: ignore
    """Model context length (prompt and output). If unspecified, will be
    automatically derived from the model config.

    When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable
    format. Examples:\n
    - 1k -> 1000\n
    - 1K -> 1024\n
    - 25.6k -> 25,600"""
    spec_target_max_model_len: Optional[int] = None
    """Specify the maximum length for spec decoding draft models."""
    quantization: SkipValidation[Optional[QuantizationMethods]] = None
    """Method used to quantize the weights. If `None`, we first check the
    `quantization_config` attribute in the model config file. If that is
    `None`, we assume the model weights are not quantized and use `dtype` to
    determine the data type of the weights."""
    enforce_eager: bool = False
    """Whether to always use eager-mode PyTorch. If True, we will disable CUDA
    graph and always execute the model in eager mode. If False, we will use
    CUDA graph and eager execution in hybrid for maximal performance and
    flexibility."""
    max_seq_len_to_capture: int = 8192
    """Maximum sequence len covered by CUDA graphs. When a sequence has context
    length larger than this, we fall back to eager mode. Additionally for
    encoder-decoder models, if the sequence length of the encoder input is
    larger than this, we fall back to the eager mode."""
    max_logprobs: int = 20
    """Maximum number of log probabilities to return when `logprobs` is
    specified in `SamplingParams`. The default value comes the default for the
    OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length *
    vocab_size) logprobs are allowed to be returned and it may cause OOM."""
    logprobs_mode: LogprobsMode = LogprobsMode.RAW_LOGPROBS
    """Indicates the content returned in the logprobs and prompt_logprobs.
    Supported mode:
    1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    Raw means the values before applying any logit processors, like bad words.
    Processed means the values after applying all processors, including
    temperature and top_k/top_p.
    """
    disable_sliding_window: bool = False
    """Whether to disable sliding window. If True, we will disable the sliding
    window functionality of the model, capping to sliding window size. If the
    model does not support sliding window, this argument is ignored."""
    disable_cascade_attn: bool = False
    """Disable cascade attention for V1. While cascade attention does not
    change the mathematical correctness, disabling it could be useful for
    preventing potential numerical issues. Note that even if this is set to
    False, cascade attention will be only used when the heuristic tells that
    it's beneficial."""
    skip_tokenizer_init: bool = False
    """Skip initialization of tokenizer and detokenizer. Expects valid
    `prompt_token_ids` and `None` for prompt from the input. The generated
    output will contain token ids."""
    enable_prompt_embeds: bool = False
    """If `True`, enables passing text embeddings as inputs via the
    `prompt_embeds` key. Note that enabling this will double the time required
    for graph compilation."""
    served_model_name: Optional[Union[str, list[str]]] = None
    """The model name(s) used in the API. If multiple names are provided, the
    server will respond to any of the provided names. The model name in the
    model field of a response will be the first name in this list. If not
    specified, the model name will be the same as the `--model` argument. Noted
    that this name(s) will also be used in `model_name` tag content of
    prometheus metrics, if multiple names provided, metrics tag will take the
    first one."""
    use_async_output_proc: bool = True
    """Whether to use async output processor."""
    config_format: Union[str, ConfigFormat] = "auto"
    """The format of the model config to load:\n
    - "auto" will try to load the config in hf format if available else it
    will try to load in mistral format.\n
    - "hf" will load the config in hf format.\n
    - "mistral" will load the config in mistral format."""
    hf_token: Optional[Union[bool, str]] = None
    """The token to use as HTTP bearer authorization for remote files . If
    `True`, will use the token generated when running `huggingface-cli login`
    (stored in `~/.huggingface`)."""
    hf_overrides: HfOverrides = field(default_factory=dict)
    """If a dictionary, contains arguments to be forwarded to the Hugging Face
    config. If a callable, it is called to update the HuggingFace config."""
    pooler_config: Optional["PoolerConfig"] = field(init=False)
    """Pooler config which controls the behaviour of output pooling in pooling
    models."""
    override_pooler_config: Optional[Union[dict, "PoolerConfig"]] = None
    """Initialize non-default pooling config or override default pooling config
    for the pooling model. e.g. `{"pooling_type": "mean", "normalize": false}`.
    """
    logits_processor_pattern: Optional[str] = None
    """Optional regex pattern specifying valid logits processor qualified names
    that can be passed with the `logits_processors` extra completion argument.
    Defaults to `None`, which allows no processors."""
    generation_config: str = "auto"
    """The folder path to the generation config. Defaults to `"auto"`, the
    generation config will be loaded from model path. If set to `"vllm"`, no
    generation config is loaded, vLLM defaults will be used. If set to a folder
    path, the generation config will be loaded from the specified folder path.
    If `max_new_tokens` is specified in generation config, then it sets a
    server-wide limit on the number of output tokens for all requests."""
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    """Overrides or sets generation config. e.g. `{"temperature": 0.5}`. If
    used with `--generation-config auto`, the override parameters will be
    merged with the default config from the model. If used with
    `--generation-config vllm`, only the override parameters are used."""
    enable_sleep_mode: bool = False
    """Enable sleep mode for the engine (only cuda platform is supported)."""
    model_impl: Union[str, ModelImpl] = ModelImpl.AUTO.value
    """Which implementation of the model to use:\n
    - "auto" will try to use the vLLM implementation, if it exists, and fall
    back to the Transformers implementation if no vLLM implementation is
    available.\n
    - "vllm" will use the vLLM model implementation.\n
    - "transformers" will use the Transformers model implementation.\n
    - "terratorch" will use the TerraTorch model implementation.
    """
    override_attention_dtype: Optional[str] = None
    """Override dtype for attention"""
    logits_processors: Optional[list[Union[str, type[LogitsProcessor]]]] = None
    """One or more logits processors' fully-qualified class names or class
    definitions"""
    io_processor_plugin: Optional[str] = None
    """IOProcessor plugin name to load at model startup"""

    # Multimodal config and init vars
    multimodal_config: Optional[MultiModalConfig] = None
    limit_mm_per_prompt: InitVar[Optional[dict[str, int]]] = None
    media_io_kwargs: InitVar[Optional[dict[str, dict[str, Any]]]] = None
    mm_processor_kwargs: InitVar[Optional[dict[str, Any]]] = None
    mm_processor_cache_gb: InitVar[Optional[float]] = None
    mm_processor_cache_type: InitVar[Optional[MMCacheType]] = None
    mm_shm_cache_max_object_size_mb: InitVar[Optional[int]] = None
    mm_encoder_tp_mode: InitVar[Optional[MMEncoderTPMode]] = None
    interleave_mm_strings: InitVar[Optional[bool]] = None
    skip_mm_profiling: InitVar[Optional[bool]] = None

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.model)
        factors.append(self.dtype)
        factors.append(self.quantization)
        factors.append(self.revision)
        factors.append(self.code_revision)
        factors.append(self.max_model_len)
        factors.append(self.max_logprobs)
        factors.append(self.disable_sliding_window)
        factors.append(self.trust_remote_code)
        factors.append(self.generation_config)
        factors.append(self.model_impl)
        factors.append(self.override_generation_config)
        factors.append(self.rope_scaling)
        factors.append(self.rope_theta)
        # hf_config can control how the model looks!
        factors.append(self.hf_config.to_json_string())
        str_factors = str(factors)
        assert_hashable(str_factors)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(
            self,
            # Multimodal config init vars
            limit_mm_per_prompt: Optional[dict[str, int]],
            media_io_kwargs: Optional[dict[str, dict[str, Any]]],
            mm_processor_kwargs: Optional[dict[str, Any]],
            mm_processor_cache_gb: Optional[float],
            mm_processor_cache_type: Optional[MMCacheType],
            mm_shm_cache_max_object_size_mb: Optional[int],
            mm_encoder_tp_mode: Optional[MMEncoderTPMode],
            interleave_mm_strings: Optional[bool],
            skip_mm_profiling: Optional[bool]) -> None:
        # Set the default seed to 0 in V1.
        # NOTE(woosuk): In V0, we set the default seed to None because the
        # driver worker shares the same process as the user process, and thus
        # setting a seed affects the user process as well.
        # In V1, we use separate processes for workers (unless
        # VLLM_ENABLE_V1_MULTIPROCESSING=0), so setting a seed here
        # doesn't affect the user process. However, without a consistent seed,
        # different tensor parallel workers would sample different tokens,
        # leading to inconsistent results.
        if envs.VLLM_USE_V1 and self.seed is None:
            self.seed = 0
            if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
                logger.warning(
                    "The global random seed is set to %d. Since "
                    "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                    "affect the random state of the Python process that "
                    "launched vLLM.", self.seed)

        # Keep set served_model_name before maybe_model_redirect(self.model)
        self.served_model_name = get_served_model_name(self.model,
                                                       self.served_model_name)
        self.model = maybe_model_redirect(self.model)
        # The tokenizer is consistent with the model by default.
        if self.tokenizer is None:
            self.tokenizer = self.model
        if self.tokenizer_revision is None:
            self.tokenizer_revision = self.revision
        self.tokenizer = maybe_model_redirect(self.tokenizer)

        if isinstance(self.hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(self.hf_config_path)

        if callable(self.hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = self.hf_overrides
        else:
            hf_overrides_kw = self.hf_overrides
            hf_overrides_fn = None

        if self.rope_scaling:
            hf_override: dict[str, Any] = {"rope_scaling": self.rope_scaling}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides_kw)
            msg = (
                "`--rope-scaling` will be removed in a future release. "
                f"'Please instead use `--hf-overrides '{hf_overrides_str}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        if self.rope_theta is not None:
            hf_override = {"rope_theta": self.rope_theta}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides_kw)
            msg = (
                "`--rope-theta` will be removed in a future release. "
                f"'Please instead use `--hf-overrides '{hf_overrides_str}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)

        self.maybe_pull_model_tokenizer_for_runai(self.model, self.tokenizer)

        if self.runner != "draft":
            # If we're not running the draft model, check for speculators config
            # If speculators config, set model / tokenizer to be target model
            self.model, self.tokenizer = maybe_override_with_speculators_target_model(  # noqa: E501
                model=self.model,
                tokenizer=self.tokenizer,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code)

        if (backend := envs.VLLM_ATTENTION_BACKEND
            ) and backend == "FLASHINFER" and find_spec("flashinfer") is None:
            raise ValueError(
                "VLLM_ATTENTION_BACKEND is set to FLASHINFER, but flashinfer "
                "module was not found. See "
                "https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile "  # noqa: E501
                "for instructions on how to install it.")

        from vllm.platforms import current_platform

        if (self.override_attention_dtype is not None
                and not current_platform.is_rocm()):
            warnings.warn(
                "override-attention-dtype is set but not using ROCm platform",
                stacklevel=2)

        if (self.enable_sleep_mode
                and not current_platform.is_sleep_mode_available()):
            raise ValueError(
                "Sleep mode is not supported on current platform.")

        hf_config = get_config(self.hf_config_path or self.model,
                               self.trust_remote_code,
                               self.revision,
                               self.code_revision,
                               self.config_format,
                               hf_overrides_kw=hf_overrides_kw,
                               hf_overrides_fn=hf_overrides_fn)

        self.hf_config = hf_config
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.attention_chunk_size = getattr(self.hf_text_config,
                                            "attention_chunk_size", None)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, hf_token=self.hf_token, revision=self.revision)

        architectures = self.architectures
        registry = self.registry
        is_generative_model = registry.is_text_generation_model(
            architectures, self)
        is_pooling_model = registry.is_pooling_model(architectures, self)

        def _task_to_convert(task: TaskOption) -> ConvertType:
            if task == "embedding" or task == "embed":
                return "embed"
            if task == "classify":
                return "classify"
            if task == "reward":
                return "reward"
            if task == "score":
                new_task = self._get_default_pooling_task(architectures)
                return "classify" if new_task == "classify" else "embed"

            return "none"

        if self.task is not None:
            runner: RunnerOption = "auto"
            convert: ConvertOption = "auto"
            msg_prefix = ("The 'task' option has been deprecated and will be "
                          "removed in v0.13.0 or v1.0, whichever comes first.")
            msg_hint = "Please remove this option."

            is_generative_task = self.task in _RUNNER_TASKS["generate"]
            is_pooling_task = self.task in _RUNNER_TASKS["pooling"]

            if is_generative_model and is_pooling_model:
                if is_generative_task:
                    runner = "generate"
                    convert = "auto"
                    msg_hint = ("Please replace this option with `--runner "
                                "generate` to continue using this model "
                                "as a generative model.")
                elif is_pooling_task:
                    runner = "pooling"
                    convert = "auto"
                    msg_hint = ("Please replace this option with `--runner "
                                "pooling` to continue using this model "
                                "as a pooling model.")
                else:  # task == "auto"
                    pass
            elif is_generative_model or is_pooling_model:
                if is_generative_task:
                    runner = "generate"
                    convert = "auto"
                    msg_hint = "Please remove this option"
                elif is_pooling_task:
                    runner = "pooling"
                    convert = _task_to_convert(self.task)
                    msg_hint = ("Please replace this option with `--convert "
                                f"{convert}` to continue using this model "
                                "as a pooling model.")
                else:  # task == "auto"
                    pass
            else:
                raise AssertionError("The model should be a generative or "
                                     "pooling model when task is set to "
                                     f"{self.task!r}.")

            self.runner = runner
            self.convert = convert

            msg = f"{msg_prefix} {msg_hint}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        self.runner_type = self._get_runner_type(architectures, self.runner)
        self.convert_type = self._get_convert_type(architectures,
                                                   self.runner_type,
                                                   self.convert)

        if self.runner_type == "generate" and not is_generative_model:
            generate_converts = _RUNNER_CONVERTS["generate"]
            if self.convert_type not in generate_converts:
                # Currently we don't have any converters for generative models
                raise ValueError(
                    "This model does not support `--runner generate`.")
        if self.runner_type == "pooling" and not is_pooling_model:
            pooling_converts = _RUNNER_CONVERTS["pooling"]
            if self.convert_type not in pooling_converts:
                convert_option = "<" + "|".join(pooling_converts) + ">"
                raise ValueError(
                    "This model does not support `--runner pooling`. "
                    f"You can pass `--convert {convert_option} to adapt "
                    "it into a pooling model.")

        self.supported_tasks = self._get_supported_tasks(
            architectures, self.runner_type, self.convert_type)

        # Note: Initialize these attributes early because transformers fallback
        # may fail to load dynamic modules in child processes
        model_info, arch = registry.inspect_model_cls(architectures, self)
        self._model_info = model_info
        self._architecture = arch
        logger.info("Resolved architecture: %s", arch)

        self.pooler_config = self._init_pooler_config()

        self.dtype: torch.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            is_pooling_model=self.runner_type == "pooling",
            revision=self.revision,
        )

        # Interleaved attention is not supported by some backends in V0
        if (not self.disable_sliding_window
                and is_interleaved(self.hf_text_config)
                and not envs.VLLM_USE_V1
                and (backend := envs.VLLM_ATTENTION_BACKEND)
                in ("XFORMERS", "FLASHINFER")):
            logger.warning_once(
                "%s has interleaved attention, which is currently not "
                "supported by the %s backend. Disabling sliding window and "
                "capping the max length to the sliding window size (%d).",
                self.hf_text_config.model_type,
                backend,
                self.hf_text_config.sliding_window,
            )
            self.disable_sliding_window = True

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
        # Init multimodal config if needed
        if self._model_info.supports_multimodal:
            if (mm_encoder_tp_mode == "data" and
                    not self._model_info.supports_multimodal_encoder_tp_data):
                logger.warning_once(
                    "This model does not support `--mm-encoder-tp-mode data`. "
                    "Falling back to `--mm-encoder-tp-mode weights`.")
                mm_encoder_tp_mode = "weights"

            mm_config_kwargs = dict(
                limit_per_prompt=limit_mm_per_prompt,
                media_io_kwargs=media_io_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                mm_processor_cache_gb=mm_processor_cache_gb,
                mm_processor_cache_type=mm_processor_cache_type,
                mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
                mm_encoder_tp_mode=mm_encoder_tp_mode,
                interleave_mm_strings=interleave_mm_strings,
                skip_mm_profiling=skip_mm_profiling,
            )

            mm_config_kwargs = {
                k: v
                for k, v in mm_config_kwargs.items() if v is not None
            }

            self.multimodal_config = MultiModalConfig(**mm_config_kwargs)

        if self.disable_sliding_window:
            # Set after get_and_verify_max_len to ensure that max_model_len
            # can be correctly capped to sliding window size
            self.hf_text_config.sliding_window = None

        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        # Avoid running try_verify_and_update_config multiple times
        self.config_updated = False

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

    @field_validator("quantization", mode="before")
    @classmethod
    def validate_quantization_before(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def validate_model_config_after(self: "ModelConfig") -> "ModelConfig":
        if not isinstance(self.tokenizer, str):
            raise ValueError("tokenizer must be a string after __post_init__.")
        if not isinstance(self.max_model_len, int):
            raise ValueError(
                "max_model_len must be an integer after __post_init__.")
        return self

    def _get_transformers_backend_cls(self) -> str:
        """Determine which Transformers backend class will be used if
        `model_impl` is set to `transformers` or `auto`."""
        if getattr(self, "runner_type", self.runner) == "pooling":
            return "TransformersModel"
        if self.hf_config != self.hf_text_config:
            # If 'hf_text_config' is the same as 'hf_config'. If not, it is
            # probably a composite config, i.e. multimodal
            return "TransformersForMultimodalLM"
        return "TransformersForCausalLM"

    def using_transformers_backend(self) -> bool:
        """Check if the model is using the Transformers backend class."""
        return self.architecture == self._get_transformers_backend_cls()

    @property
    def registry(self):
        return me_models.ModelRegistry

    @property
    def architectures(self) -> list[str]:
        return getattr(self.hf_config, "architectures", [])

    @property
    def architecture(self) -> str:
        """The architecture vllm actually used."""
        return self._architecture

    def maybe_pull_model_tokenizer_for_runai(self, model: str,
                                             tokenizer: str) -> None:
        """Pull model/tokenizer from Object Storage to temporary
        directory when needed.

        Args:
            model: Model name or path
            tokenizer: Tokenizer name or path
        """
        if not (is_runai_obj_uri(model) or is_runai_obj_uri(tokenizer)):
            return

        if is_runai_obj_uri(model):
            object_storage_model = ObjectStorageModel()
            object_storage_model.pull_files(
                model, allow_pattern=["*.model", "*.py", "*.json"])
            self.model_weights = model
            self.model = object_storage_model.dir

            # If tokenizer is same as model, download to same directory
            if model == tokenizer:
                object_storage_model.pull_files(model,
                                                ignore_pattern=[
                                                    "*.pt", "*.safetensors",
                                                    "*.bin", "*.tensors"
                                                ])
                self.tokenizer = object_storage_model.dir
                return

        # Only download tokenizer if needed and not already handled
        if is_runai_obj_uri(tokenizer):
            object_storage_tokenizer = ObjectStorageModel()
            object_storage_tokenizer.pull_files(
                model,
                ignore_pattern=["*.pt", "*.safetensors", "*.bin", "*.tensors"])
            self.tokenizer = object_storage_tokenizer.dir

    def _get_encoder_config(self):
        return get_sentence_transformer_tokenizer_config(
            self.model, self.revision)

    def _init_pooler_config(self) -> Optional["PoolerConfig"]:
        if self.runner_type == "pooling":
            if isinstance(self.override_pooler_config, dict):
                self.override_pooler_config = PoolerConfig(
                    **self.override_pooler_config)

            pooler_config = self.override_pooler_config or PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                for k, v in base_config.items():
                    if getattr(pooler_config, k) is None:
                        setattr(pooler_config, k, v)

            default_pooling_type = self._model_info.default_pooling_type
            if pooler_config.pooling_type is None:
                pooler_config.pooling_type = default_pooling_type

            return pooler_config

        return None

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = cast(TokenizerMode, self.tokenizer_mode.lower())
        if tokenizer_mode not in get_args(TokenizerMode):
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                f"one of {get_args(TokenizerMode)}.")
        self.tokenizer_mode = tokenizer_mode

    def _get_default_runner_type(
        self,
        architectures: list[str],
    ) -> RunnerType:
        registry = self.registry

        # Some Sentence Transformers models use *ForCausalLM archs
        if get_pooling_config(self.model, self.revision):
            return "pooling"

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if registry.is_pooling_model(architectures, self):
                    return "pooling"
                if registry.is_text_generation_model(architectures, self):
                    return "generate"

            match = try_match_architecture_defaults(arch)
            if match:
                _, (runner_type, _) = match
                return runner_type

        return "generate"

    def _get_runner_type(
        self,
        architectures: list[str],
        runner: RunnerOption,
    ) -> RunnerType:
        if runner != "auto":
            return runner

        runner_type = self._get_default_runner_type(architectures)

        # Don't log the most common case
        if runner_type != "generate":
            logger.info(
                "Resolved `--runner auto` to `--runner %s`. "
                "Pass the value explicitly to silence this message.",
                runner_type)

        return runner_type

    def _get_default_convert_type(
        self,
        architectures: list[str],
        runner_type: RunnerType,
    ) -> ConvertType:
        registry = self.registry

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if (runner_type == "generate"
                        and registry.is_text_generation_model(
                            architectures, self)):
                    return "none"
                if (runner_type == "pooling"
                        and registry.is_pooling_model(architectures, self)):
                    return "none"

            match = try_match_architecture_defaults(arch,
                                                    runner_type=runner_type)
            if match:
                _, (_, convert_type) = match
                return convert_type

        # This is to handle Sentence Transformers models that use *ForCausalLM
        # and also multi-modal pooling models which are not defined as
        # Sentence Transformers models
        if runner_type == "pooling":
            return "embed"

        return "none"

    def _get_convert_type(
        self,
        architectures: list[str],
        runner_type: RunnerType,
        convert: ConvertOption,
    ) -> ConvertType:
        if convert != "auto":
            return convert

        convert_type = self._get_default_convert_type(architectures,
                                                      runner_type)

        # Don't log the most common case
        if convert_type != "none":
            logger.info(
                "Resolved `--convert auto` to `--convert %s`. "
                "Pass the value explicitly to silence this message.",
                convert_type)

        return convert_type

    def _get_supported_generation_tasks(
        self,
        architectures: list[str],
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        registry = self.registry

        if registry.is_transcription_only_model(architectures, self):
            return ["transcription"]

        # TODO: Use get_supported_generation_tasks once V0 is removed
        supported_tasks = list[_ResolvedTask]()
        if (registry.is_text_generation_model(architectures, self)
                or convert_type in _RUNNER_CONVERTS["generate"]):
            supported_tasks.append("generate")

        if registry.is_transcription_model(architectures, self):
            supported_tasks.append("transcription")

        return supported_tasks

    def _get_default_pooling_task(
        self,
        architectures: list[str],
    ) -> Literal["embed", "classify", "reward"]:
        if self.registry.is_cross_encoder_model(architectures, self):
            return "classify"

        for arch in architectures:
            match = try_match_architecture_defaults(arch,
                                                    runner_type="pooling")
            if match:
                _, (_, convert_type) = match
                assert convert_type != "none"
                return convert_type

        return "embed"

    def _get_supported_pooling_tasks(
        self,
        architectures: list[str],
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        registry = self.registry

        # TODO: Use get_supported_pooling_tasks once V0 is removed
        supported_tasks = list[_ResolvedTask]()
        if (registry.is_pooling_model(architectures, self)
                or convert_type in _RUNNER_CONVERTS["pooling"]):
            supported_tasks.append("encode")

            extra_task = (self._get_default_pooling_task(architectures)
                          if convert_type == "none" else convert_type)
            supported_tasks.append(extra_task)

        return supported_tasks

    def _get_supported_tasks(
        self,
        architectures: list[str],
        runner_type: RunnerType,
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        if runner_type == "generate":
            return self._get_supported_generation_tasks(
                architectures, convert_type)
        if runner_type == "pooling":
            return self._get_supported_pooling_tasks(architectures,
                                                     convert_type)
        if runner_type == "draft":
            return ["draft"]

        assert_never(runner_type)

    def _parse_quant_hf_config(self, hf_config: PretrainedConfig):
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(hf_config, "compression_config", None)

        else:
            # Set quant_method for ModelOpt models.
            producer_name = quant_cfg.get("producer", {}).get("name")
            if producer_name == "modelopt":
                quant_algo = quant_cfg.get("quantization",
                                           {}).get("quant_algo")
                if quant_algo == "FP8":
                    quant_cfg["quant_method"] = "modelopt"
                elif quant_algo == "NVFP4":
                    quant_cfg["quant_method"] = "modelopt_fp4"
                elif quant_algo is not None:
                    raise ValueError(
                        f"Unknown ModelOpt quant algo: {quant_algo}")

        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = me_quant.QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8",
            "modelopt",
            "gptq_marlin_24",
            "gptq_marlin",
            "awq_marlin",
            "fbgemm_fp8",
            "compressed-tensors",
            "experts_int8",
            "quark",
            "modelopt_fp4",
            "bitblas",
            "gptq_bitblas",
            "inc",
            "petit_nvfp4",
        ]
        if self.quantization is not None:
            self.quantization = cast(me_quant.QuantizationMethods,
                                     self.quantization)

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config(self.hf_config)
        if quant_cfg is None and (text_config := getattr(
                self.hf_config, "text_config", None)):
            # Check the text config as well for multi-modal models.
            quant_cfg = self._parse_quant_hf_config(text_config)

        if quant_cfg is not None:
            # Use the community standard 'quant_method'
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Normalize library names
            quant_method = quant_method.replace("compressed_tensors",
                                                "compressed-tensors")

            quant_cfg["quant_method"] = quant_method

            # Quantization methods which are overrides (i.e. they have a
            # `override_quantization_method` method) must be checked in order
            # of preference (this is particularly important for GPTQ).
            overrides = [
                "bitblas",
                "gptq_marlin_24",
                "gptq_marlin",
                "gptq_bitblas",
                "awq_marlin",
                "ipex",
                "moe_wna16",
                "modelopt",
                "modelopt_fp4",
                "petit_nvfp4",
            ]
            quantization_methods = [
                q for q in supported_quantization if q not in overrides
            ]
            # Any custom overrides will be in quantization_methods so we place
            # them at the start of the list so custom overrides have preference
            # over the built-in ones.
            quantization_methods = quantization_methods + overrides

            # Detect which checkpoint is it
            for name in quantization_methods:
                method = me_quant.get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override is not None:
                    # Raise error if the override is not custom (custom would
                    # be in QUANTIZATION_METHODS but not QuantizationMethods)
                    # and hasn't been added to the overrides list.
                    if (name in get_args(me_quant.QuantizationMethods)
                            and name not in overrides):
                        raise ValueError(
                            f"Quantization method {name} is an override but "
                            "is has not been added to the `overrides` list "
                            "above. This is necessary to ensure that the "
                            "overrides are checked in order of preference.")
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            from vllm.platforms import current_platform
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.", self.quantization)

    def _verify_cuda_graph(self) -> None:
        # The `max_seq_len_to_capture` was incorrectly
        # based on the encoder's input length (448)
        # but not the decoder's larger input length (1500).
        # This change ensures the CUDA Graph captures the correct,
        # larger sequence length, allowing it to work as intended.
        effective_max_seq_len = self.max_model_len
        if self.is_encoder_decoder:
            effective_max_seq_len = max(
                effective_max_seq_len,
                getattr(self.hf_config, "max_source_positions", 0))
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          effective_max_seq_len)
        # CUDAGraph capture not supported for encoder-decoder models on ROCm
        unsupported_rocm = self.is_encoder_decoder

        if (unsupported_rocm and not self.enforce_eager
                and current_platform.is_rocm()):
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to eager mode.", self.hf_config.model_type)
            self.enforce_eager = True

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.46.1) with 8-bit models does not
        yet support CUDA graph.
        # TODO Remove this when bitsandbytes supports.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = (getattr(self.hf_config,
                                           "quantization_config", None)
                                   is not None)
        is_8bit = (self.hf_config.quantization_config.get(
            "load_in_8bit", False) if has_quantization_config else False)
        if all([
                is_bitsandbytes,
                has_quantization_config,
                is_8bit,
                not self.enforce_eager,
        ]):
            logger.warning(
                "CUDA graph is not supported on BitsAndBytes 8bit yet, "
                "fallback to the eager mode.")

            self.enforce_eager = True

    def _verify_with_expert_parallelism(self) -> None:
        num_expert_names = [
            "moe_num_experts",  # Dbrx
            "num_experts",  # Jamba
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(self.hf_text_config, name, 0)
            if num_experts > 0:
                break
        if num_experts < 1:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled.")

    def verify_dual_chunk_attention_config(
        self,
        load_config: "LoadConfig",
    ) -> None:
        if hasattr(self.hf_config, "dual_chunk_attention_config"):
            # Try loading the sparse attention config
            from vllm.model_executor.model_loader.weight_utils import (
                get_sparse_attention_config)
            sparse_attn_config = get_sparse_attention_config(self, load_config)
            if sparse_attn_config:
                self.hf_config.dual_chunk_attention_config[
                    "sparse_attention_config"] = sparse_attn_config
                if "sparse_attention_enabled" not in \
                        self.hf_config.dual_chunk_attention_config:
                    self.hf_config.dual_chunk_attention_config[
                        "sparse_attention_enabled"] = True

            if envs.VLLM_ATTENTION_BACKEND != STR_DUAL_CHUNK_FLASH_ATTN_VAL:
                raise ValueError("please set VLLM_ATTENTION_BACKEND to "
                                 f"{STR_DUAL_CHUNK_FLASH_ATTN_VAL}")

    def verify_async_output_proc(self, parallel_config, speculative_config,
                                 device_config) -> None:
        if not self.use_async_output_proc:
            # Nothing to check
            return

        if parallel_config.pipeline_parallel_size > 1:
            self.use_async_output_proc = False
            return

        # Reminder: Please update docs/features/compatibility_matrix.md
        # If the feature combo become valid
        from vllm.platforms import current_platform
        if not current_platform.is_async_output_supported(self.enforce_eager):
            self.use_async_output_proc = False
            return

        if envs.VLLM_USE_RAY_SPMD_WORKER:
            self.use_async_output_proc = False
            return

        # Async postprocessor is not necessary for pooling models
        # since there is no token generation
        if self.runner_type == "pooling":
            self.use_async_output_proc = False

        # Reminder: Please update docs/features/compatibility_matrix.md
        # If the feature combo become valid
        if speculative_config:
            self.use_async_output_proc = False

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:

        if parallel_config.distributed_executor_backend == "external_launcher":
            assert self.seed is not None, (
                "Seed must be set when using external launcher backend to "
                "make sure sampling results are the same across workers.")

        total_num_attention_heads = getattr(self.hf_text_config,
                                            "num_attention_heads", 0)
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        if parallel_config.enable_expert_parallel:
            self._verify_with_expert_parallelism()

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1:
            if not self.registry.is_pp_supported_model(self.architectures,
                                                       self):
                raise NotImplementedError(
                    "Pipeline parallelism is not supported for this model. "
                    "Supported models implement the `SupportsPP` interface.")

            if self.use_async_output_proc:
                self.use_async_output_proc = False

    def get_sliding_window(self) -> Optional[int]:
        """Get the sliding window size from the HF text config if present."""
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "vocab_size", 0)

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "hidden_size", 0)

    @property
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in \
            ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp', 'kimi_k2'):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == 'eagle':
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return self.hf_text_config.model.model_type in \
                    ('deepseek_v2', 'deepseek_v3') \
                and self.hf_text_config.kv_lora_rank is not None
        return False

    def get_head_size(self) -> int:
        # TODO remove hard code
        if self.is_deepseek_mla:
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim",
                                       0)
            if self.use_mla:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config,
                                           "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        if hasattr(self.hf_text_config,
                   "model_type") and (self.hf_text_config.model_type
                                      == "zamba2"):
            return self.hf_text_config.attention_head_dim

        if self.is_attention_free:
            return 0

        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_text_config, "head_dim", None) is not None:
            return self.hf_text_config.head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(self.hf_text_config, "hidden_size_per_head",
                   None) is not None:
            return self.hf_text_config.hidden_size_per_head

        # FIXME(woosuk): This may not be true for all models.
        return (self.hf_text_config.hidden_size //
                self.hf_text_config.num_attention_heads)

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_text_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(self.hf_config.attn_config, "kv_n_heads",
                           self.hf_config.num_attention_heads)

        if self.hf_config.model_type == "nemotron-nas":
            for block in self.hf_config.block_configs:
                if not block.attention.no_op:
                    return self.hf_config.num_attention_heads \
                        // block.attention.n_heads_in_group

            raise RuntimeError("Couldn't determine number of kv heads")

        if self.is_attention_free:
            return 0

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self,
                                parallel_config: "ParallelConfig") -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_layers_start_end_indices(
            self, parallel_config: "ParallelConfig") -> tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices
        if (self.hf_text_config.model_type == "deepseek_mtp"
                or self.hf_config.model_type == "mimo_mtp"
                or self.hf_config.model_type == "glm4_moe_mtp"
                or self.hf_config.model_type == "ernie_mtp"
                or self.hf_config.model_type == "qwen3_next_mtp"):
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_nextn_predict_layers", 0)
        else:
            total_num_hidden_layers = getattr(self.hf_text_config,
                                              "num_hidden_layers", 0)
        # the layout order is: DP x PP x TP
        pp_rank = (parallel_config.rank // parallel_config.tensor_parallel_size
                   ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(
        self,
        parallel_config: "ParallelConfig",
        block_type: LayerBlockType = LayerBlockType.attention,
    ) -> int:
        # This function relies on 'layers_block_type' in hf_config,
        # for w/o this attribute, we will need to have workarounds like so
        attn_block_type = block_type == LayerBlockType.attention
        is_transformer = not self.is_hybrid and \
                            not self.has_noops and \
                            not self.is_attention_free
        start, end = self.get_layers_start_end_indices(parallel_config)

        if is_transformer:
            # Handle the basic case first
            return end - start if attn_block_type else 0
        elif self.is_attention_free:
            # Attention free
            # Note that this code assumes there
            # is only one type of attention-free block type.
            return 0 if attn_block_type else end - start
        elif self.has_noops:
            block_configs = self.hf_config.block_configs
            return sum(not bc.attention.no_op
                       for bc in block_configs[start:end])
        else:
            # Hybrid model Jamba
            layers_block_type_value = getattr(self.hf_text_config,
                                              "layers_block_type", None)
            if layers_block_type_value is not None:
                if hasattr(self.hf_text_config,
                           "model_type") and (self.hf_text_config.model_type
                                              == "zamba2"):
                    if attn_block_type:
                        return sum(t == "hybrid"
                                   for t in layers_block_type_value[start:end])
                    else:
                        return self.get_num_layers(parallel_config)
                return sum(t == block_type.value
                           for t in layers_block_type_value[start:end])

            # Hybrid model Minimax
            attn_type_list = getattr(self.hf_config, "attn_type_list", None)
            if attn_type_list:
                return sum(t == 1 for t in attn_type_list[start:end])

            # Hybrid model Qwen3Next
            layer_types_value = getattr(self.hf_config, "layer_types", None)
            if layer_types_value is not None:
                if getattr(block_type, "value", block_type) == "attention":
                    return sum(t == "full_attention"
                               for t in layer_types_value[start:end])
                elif getattr(block_type, "value",
                             block_type) == "linear_attention":
                    return sum(t == "linear_attention"
                               for t in layer_types_value[start:end])
                else:
                    return sum(t == getattr(block_type, "value", block_type)
                               for t in layer_types_value[start:end])

            if (layers_block_type_value is None and attn_type_list is None
                    and layer_types_value is None):
                raise ValueError(
                    "The model is an hybrid without a"
                    "layers_block_type or an attn_type_list, or a layer_types "
                    "in the hf_config, cannot determine the num of "
                    f"{block_type.value} layers")

    def get_mamba_chunk_size(self) -> Optional[int]:
        """
        Returns the mamba chunk size if it exists
        """
        # used by e.g. Bamba, FalconH1, Granite, PLaMo2
        chunk_size = getattr(self.hf_text_config, "mamba_chunk_size", None)
        if chunk_size is None:
            # used by e.g. Mamba2, NemotronH, Zamba
            chunk_size = getattr(self.hf_text_config, "chunk_size", None)
        return chunk_size

    def get_multimodal_config(self) -> "MultiModalConfig":
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")

        return self.multimodal_config

    def try_get_generation_config(self) -> dict[str, Any]:
        """
        This method attempts to retrieve the non-default values of the
        generation config for this model.

        The generation config can contain information about special tokens, as
        well as sampling parameters. Which is why this method exists separately
        to `get_diff_sampling_param`.

        Returns:
            A dictionary containing the non-default generation config.
        """
        if self.generation_config in {"auto", "vllm"}:
            config = try_get_generation_config(
                self.hf_config_path or self.model,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
        else:
            config = try_get_generation_config(
                self.generation_config,
                trust_remote_code=self.trust_remote_code,
            )

        if config is None:
            return {}

        return config.to_diff_dict()

    def get_diff_sampling_param(self) -> dict[str, Any]:
        """
        This method returns a dictionary containing the non-default sampling
        parameters with `override_generation_config` applied.

        The default sampling parameters are:

        - vLLM's neutral defaults if `self.generation_config="vllm"`
        - the model's defaults if `self.generation_config="auto"`
        - as defined in `generation_config.json` if
            `self.generation_config="path/to/generation_config/dir"`

        Returns:
            A dictionary containing the non-default sampling parameters.
        """
        if self.generation_config == "vllm":
            config = {}
        else:
            config = self.try_get_generation_config()

        # Overriding with given generation config
        config.update(self.override_generation_config)

        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "max_new_tokens",
        ]
        if any(p in config for p in available_params):
            diff_sampling_param = {
                p: config.get(p)
                for p in available_params if config.get(p) is not None
            }
            # Huggingface definition of max_new_tokens is equivalent
            # to vLLM's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens")
        else:
            diff_sampling_param = {}

        if diff_sampling_param:
            logger.warning_once(
                "Default sampling parameters have been overridden by the "
                "model's Hugging Face generation config recommended from the "
                "model creator. If this is not intended, please relaunch "
                "vLLM instance with `--generation-config vllm`.")
        return diff_sampling_param

    @property
    def is_encoder_decoder(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        return is_encoder_decoder(self.hf_config)

    @property
    def uses_mrope(self) -> bool:
        return uses_mrope(self.hf_config)

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None

    @property
    def is_multimodal_raw_input_only_model(self) -> bool:
        return self._model_info.supports_multimodal_raw_input_only

    @property
    def is_cross_encoder(self) -> bool:
        return (self._model_info.supports_cross_encoding
                or self.convert_type == "classify")

    @property
    def is_pp_supported(self) -> bool:
        return self._model_info.supports_pp

    @property
    def is_attention_free(self) -> bool:
        return self._model_info.is_attention_free

    @property
    def is_hybrid(self) -> bool:
        return self._model_info.is_hybrid

    @property
    def has_noops(self) -> bool:
        return self._model_info.has_noops

    @property
    def has_inner_state(self):
        return self._model_info.has_inner_state

    @property
    def is_v1_compatible(self) -> bool:
        return not self._model_info.supports_v0_only

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE

    @property
    def is_matryoshka(self) -> bool:
        return (bool(getattr(self.hf_config, "matryoshka_dimensions", None))
                or getattr(self.hf_config, "is_matryoshka", False))

    @property
    def matryoshka_dimensions(self):
        return getattr(self.hf_config, "matryoshka_dimensions", None)

    @property
    def use_pad_token(self) -> bool:
        # cross_encoder models defaults to using pad_token.
        # `llm as reranker` models defaults to not using pad_token.
        return getattr(self.hf_config, "use_pad_token", True)

    @property
    def head_dtype(self) -> torch.dtype:
        """
        "head" refers to the last Linear layer(s) of an LLM,
        such as the lm_head in a generation model,
        or the score or classifier in a classification model.

        `head_dtype` currently only supports pooling models.\n
        - The pooling model defaults to using fp32 head,
        you can use --hf-overrides '{"head_dtype": "model"}' to disable it.
        """

        head_dtype = _get_head_dtype(config=self.hf_config,
                                     dtype=self.dtype,
                                     runner_type=self.runner_type)

        if self.runner_type != "pooling" and head_dtype != self.dtype:
            logger.warning_once(
                "`head_dtype` currently only supports pooling models."
                "fallback to model dtype [%s].", self.dtype)
            return self.dtype

        if head_dtype not in current_platform.supported_dtypes:
            logger.warning_once(
                "The current platform does not support [%s] head dtype, "
                "fallback to model dtype [%s].", head_dtype, self.dtype)
            return self.dtype

        logger.debug_once("head dtype: %s", head_dtype)
        return head_dtype

    def get_and_verify_max_len(self, max_model_len: int):
        # Consider max_model_len in tokenizer_config only when
        # pooling models use absolute position_embedding.
        tokenizer_config = None
        if (self.runner_type == "pooling" and getattr(
                self.hf_config, "position_embedding_type", "") == "absolute"):
            tokenizer_config = try_get_tokenizer_config(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                revision=self.tokenizer_revision)
        max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            tokenizer_config=tokenizer_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window=self.get_sliding_window(),
            spec_target_max_model_len=self.spec_target_max_model_len,
            encoder_config=self.encoder_config)
        logger.info("Using max model len %s", max_model_len)
        return max_model_len


Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    """Configuration for the device to use for vLLM execution."""

    device: SkipValidation[Optional[Union[Device, torch.device]]] = "auto"
    """Device type for vLLM execution.
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
    on the current platform."""
    device_type: str = field(init=False)
    """Device type from the current platform. This is set in
    `__post_init__`."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # the device/platform information will be summarized
        # by torch/vllm automatically.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.device == "auto":
            # Automated device type detection
            from vllm.platforms import current_platform
            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue.")
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some device types require processing inputs on CPU
        if self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)


@config
@dataclass
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    pooling_type: Optional[str] = None
    """
    The pooling method of the pooling model. This should be a key in
    [`vllm.model_executor.layers.pooler.PoolingType`][].
    """

    ## for embeddings models
    normalize: Optional[bool] = None
    """
    Whether to normalize the embeddings outputs. Defaults to True.
    """
    dimensions: Optional[int] = None
    """
    Reduce the dimensions of embeddings if model
    support matryoshka representation. Defaults to None.
    """
    enable_chunked_processing: Optional[bool] = None
    """
    Whether to enable chunked processing for long inputs that exceed the model's
    maximum position embeddings. When enabled, long inputs will be split into
    chunks, processed separately, and then aggregated using weighted averaging.
    This allows embedding models to handle arbitrarily long text without CUDA
    errors. Defaults to False.
    """
    max_embed_len: Optional[int] = None
    """
    Maximum input length allowed for embedding generation. When set, allows
    inputs longer than max_embed_len to be accepted for embedding models.
    When an input exceeds max_embed_len, it will be handled according to 
    the original max_model_len validation logic. 
    Defaults to None (i.e. set to max_model_len).
    """

    ## for classification models
    activation: Optional[bool] = None
    """
    Whether to apply activation function to the classification outputs.
    Defaults to True.
    """
    logit_bias: Optional[float] = None
    """
    If provided, apply classification logit biases. Defaults to None.
    """

    ## for reward models
    softmax: Optional[bool] = None
    """
    Whether to apply softmax to the reward outputs.
    Defaults to True.
    """
    step_tag_id: Optional[int] = None
    """
    If set, only the score corresponding to the ``step_tag_id`` in the
    generated sentence should be returned. Otherwise, the scores for all tokens
    are returned.
    """
    returned_token_ids: Optional[list[int]] = None
    """
    A list of indices for the vocabulary dimensions to be extracted,
    such as the token IDs of ``good_token`` and ``bad_token`` in the
    ``math-shepherd-mistral-7b-prm`` model.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# model_type -> reason
_FLOAT16_NOT_SUPPORTED_MODELS = {
    "gemma2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "gemma3": "Numerical instability. Please use bfloat16 or float32 instead.",
    "gemma3_text":
    "Numerical instability. Please use bfloat16 or float32 instead.",
    "plamo2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "glm4": "Numerical instability. Please use bfloat16 or float32 instead.",
}


def _is_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:  # noqa: E501, SIM103
        return False

    return True


def _check_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:
        reason = _FLOAT16_NOT_SUPPORTED_MODELS[model_type]
        raise ValueError(f"The model type {model_type!r} "
                         f"does not support float16. Reason: {reason}")

    return True


def _find_dtype(
    model_id: str,
    config: PretrainedConfig,
    *,
    revision: Optional[str],
):
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)

    # Fallbacks for multi-modal models if the root config
    # does not define torch_dtype
    if config_dtype is None:
        config_dtype = getattr(config.get_text_config(), "torch_dtype", None)
    if config_dtype is None and hasattr(config, "vision_config"):
        config_dtype = getattr(config.vision_config, "torch_dtype", None)
    if config_dtype is None and hasattr(config, "encoder_config"):
        config_dtype = getattr(config.encoder_config, "torch_dtype", None)

    # Try to read the dtype of the weights if they are in safetensors format
    if config_dtype is None:
        repo_mt = try_get_safetensors_metadata(model_id, revision=revision)

        if repo_mt and (files_mt := repo_mt.files_metadata):
            param_dtypes: set[torch.dtype] = {
                _SAFETENSORS_TO_TORCH_DTYPE[dtype_str]
                for file_mt in files_mt.values()
                for dtype_str in file_mt.parameter_count
                if dtype_str in _SAFETENSORS_TO_TORCH_DTYPE
            }

            if param_dtypes:
                return common_broadcastable_dtype(param_dtypes)

    if config_dtype is None:
        config_dtype = torch.float32

    return config_dtype


def _resolve_auto_dtype(
    model_type: str,
    config_dtype: torch.dtype,
    *,
    is_pooling_model: bool,
):
    from vllm.platforms import current_platform

    supported_dtypes = [
        dtype for dtype in current_platform.supported_dtypes
        if _is_valid_dtype(model_type, dtype)
    ]

    if is_pooling_model and torch.float16 in supported_dtypes:
        preferred_dtype = torch.float16
    else:
        preferred_dtype = supported_dtypes[0]

    # Downcast for float32 models
    if config_dtype == torch.float32:
        config_dtype = preferred_dtype

    if config_dtype in supported_dtypes:
        return config_dtype

    # Ensure device compatibility
    device_name = current_platform.get_device_name()
    device_capability = current_platform.get_device_capability()

    if device_capability is None:
        device_str = f"{device_name!r}"
    else:
        version_str = device_capability.as_version_str()
        device_str = f"{device_name!r} (with compute capability {version_str})"

    logger.warning(
        "Your device %s doesn't support %s. "
        "Falling back to %s for compatibility.",
        device_str,
        config_dtype,
        preferred_dtype,
    )

    return preferred_dtype


def _get_and_verify_dtype(
    model_id: str,
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
    *,
    is_pooling_model: bool,
    revision: Optional[str] = None,
) -> torch.dtype:
    config_dtype = _find_dtype(model_id, config, revision=revision)
    model_type = config.model_type

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            # Set default dtype from model config
            torch_dtype = _resolve_auto_dtype(
                model_type,
                config_dtype,
                is_pooling_model=is_pooling_model,
            )
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype!r}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    _check_valid_dtype(model_type, torch_dtype)

    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


def _get_head_dtype(config: PretrainedConfig, dtype: torch.dtype,
                    runner_type: str) -> torch.dtype:
    head_dtype: Optional[Union[str,
                               torch.dtype]] = getattr(config, "head_dtype",
                                                       None)

    if head_dtype == "model":
        return dtype
    elif isinstance(head_dtype, str):
        head_dtype = head_dtype.lower()
        if head_dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {head_dtype!r}")
        return _STR_DTYPE_TO_TORCH_DTYPE[head_dtype]
    elif isinstance(head_dtype, torch.dtype):
        return head_dtype
    elif head_dtype is None:
        if torch.float32 not in current_platform.supported_dtypes:
            return dtype
        if runner_type == "pooling":
            return torch.float32
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {head_dtype}")


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    tokenizer_config: Optional[dict],
    max_model_len: Optional[int],
    disable_sliding_window: bool,
    sliding_window: Optional[int],
    spec_target_max_model_len: Optional[int] = None,
    encoder_config: Optional[Any] = None,
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len \
                else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)
    # For Command-R / Cohere, Cohere2 / Aya Vision models
    if tmp_max_len := getattr(hf_config, "model_max_length", None):
        max_len_key = "model_max_length"
        derived_max_model_len = tmp_max_len

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if (disable_sliding_window and sliding_window is not None
            and sliding_window < derived_max_model_len):
        max_len_key = "sliding_window"
        derived_max_model_len = sliding_window

    # Consider model_max_length in tokenizer_config
    if tokenizer_config:
        tokenizer_model_max_length = tokenizer_config.get(
            "model_max_length", derived_max_model_len)
        derived_max_model_len = min(derived_max_model_len,
                                    tokenizer_model_max_length)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        if spec_target_max_model_len is not None:
            # If this is a speculative draft model, we use the max model len
            # from the target model.
            return spec_target_max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.", possible_keys,
            default_max_len)
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    # NOTE(woosuk): Gemma3's max_model_len (128K) is already scaled by RoPE
    # scaling, so we skip applying the scaling factor again.
    if rope_scaling is not None and "gemma3" not in hf_config.model_type:
        # No need to consider "type" key because of patch_rope_scaling when
        # loading HF config
        rope_type = rope_scaling["rope_type"]

        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that supports rope_scaling
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate.")

            # NOTE: rope_type == "default" does not define factor
            # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
            scaling_factor = rope_scaling.get("factor", 1.0)

            if rope_type == "yarn":
                derived_max_model_len = rope_scaling[
                    "original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    if encoder_config and "max_seq_length" in encoder_config:
        derived_max_model_len = encoder_config["max_seq_length"]

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
        if current_platform.is_tpu():
            logger.warning(
                "--max-model-len is not specified, "
                "it's currently using model's default length %s, "
                "which might be too large."
                "Please input with --max-model-len based on your "
                "request input length and output length, to avoid "
                "unnecessary degradation.", max_model_len)
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is not None and max_model_len <= model_max_length:
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that has model_max_length
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "model_max_length in the config. Please raise an issue "
                    "so we can investigate.")
        else:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json).")
            warning = (
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme "
                "caution. If the model uses relative position encoding (RoPE), "
                "positions exceeding derived_max_model_len lead to nan. If the "
                "model uses absolute position encoding, positions exceeding "
                "derived_max_model_len will cause a CUDA array out-of-bounds "
                "error.")
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning_once("%s %s", msg, warning)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    f"the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. {warning}")
    return int(max_model_len)


def get_served_model_name(model: str,
                          served_model_name: Optional[Union[str, list[str]]]):
    """
    If the input is a non-empty list, the first model_name in
    `served_model_name` is taken.
    If the input is a non-empty string, it is used directly.
    For cases where the input is either an empty string or an
    empty list, the fallback is to use `self.model`.
    """
    if not served_model_name:
        return model
    if isinstance(served_model_name, list):
        return served_model_name[0]
    return served_model_name


GuidedDecodingBackend = Literal["auto", "xgrammar", "guidance", "outlines",
                                "lm-format-enforcer"]


@config
@dataclass
class DecodingConfig:
    """Dataclass which contains the decoding strategy of the engine."""

    backend: GuidedDecodingBackend = "auto"
    """Which engine will be used for guided decoding (JSON schema / regex etc)
    by default. With "auto", we will make opinionated choices based on request
    contents and what the backend libraries currently support, so the behavior
    is subject to change in each release."""

    disable_fallback: bool = False
    """If `True`, vLLM will not fallback to a different backend on error."""

    disable_any_whitespace: bool = False
    """If `True`, the model will not generate any whitespace during guided
    decoding. This is only supported for xgrammar and guidance backends."""

    disable_additional_properties: bool = False
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""

    reasoning_backend: str = ""
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if (self.disable_any_whitespace
                and self.backend not in ("xgrammar", "guidance")):
            raise ValueError("disable_any_whitespace is only supported for "
                             "xgrammar and guidance backends.")
        if (self.disable_additional_properties and self.backend != "guidance"):
            raise ValueError("disable_additional_properties is only supported "
                             "for the guidance backend.")


DetailedTraceModules = Literal["model", "worker", "all"]


@config
@dataclass
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""

    show_hidden_metrics_for_version: Optional[str] = None
    """Enable deprecated Prometheus metrics that have been hidden since the
    specified version. For example, if a previously deprecated metric has been
    hidden since the v0.7.0 release, you use
    `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while
    you migrate to new metrics. The metric is likely to be removed completely
    in an upcoming release."""

    @cached_property
    def show_hidden_metrics(self) -> bool:
        """Check if the hidden metrics should be shown."""
        if self.show_hidden_metrics_for_version is None:
            return False
        return version._prev_minor_version_was(
            self.show_hidden_metrics_for_version)

    otlp_traces_endpoint: Optional[str] = None
    """Target URL to which OpenTelemetry traces will be sent."""

    collect_detailed_traces: Optional[list[DetailedTraceModules]] = None
    """It makes sense to set this only if `--otlp-traces-endpoint` is set. If
    set, it will collect detailed traces for the specified modules. This
    involves use of possibly costly and or blocking operations and hence might
    have a performance impact.

    Note that collecting detailed timing information for each request can be
    expensive."""

    @cached_property
    def collect_model_forward_time(self) -> bool:
        """Whether to collect model forward time for the request."""
        return (self.collect_detailed_traces is not None
                and ("model" in self.collect_detailed_traces
                     or "all" in self.collect_detailed_traces))

    @cached_property
    def collect_model_execute_time(self) -> bool:
        """Whether to collect model execute time for the request."""
        return (self.collect_detailed_traces is not None
                and ("worker" in self.collect_detailed_traces
                     or "all" in self.collect_detailed_traces))

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        if (self.collect_detailed_traces is not None
                and len(self.collect_detailed_traces) == 1
                and "," in self.collect_detailed_traces[0]):
            self._parse_collect_detailed_traces()

        from vllm.tracing import is_otel_available, otel_import_error_traceback
        if not is_otel_available() and self.otlp_traces_endpoint is not None:
            raise ValueError(
                "OpenTelemetry is not available. Unable to configure "
                "'otlp_traces_endpoint'. Ensure OpenTelemetry packages are "
                f"installed. Original error:\n{otel_import_error_traceback}")

    def _parse_collect_detailed_traces(self):
        assert isinstance(self.collect_detailed_traces, list)
        self.collect_detailed_traces = cast(
            list[DetailedTraceModules],
            self.collect_detailed_traces[0].split(","))


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    # TODO: use default_factory once default constructing ModelConfig doesn't
    # try to download a model
    model_config: ModelConfig = None  # type: ignore
    """Model configuration."""
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    """Cache configuration."""
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    """Parallel configuration."""
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler configuration."""
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    """Device configuration."""
    load_config: LoadConfig = field(default_factory=LoadConfig)
    """Load configuration."""
    lora_config: Optional[LoRAConfig] = None
    """LoRA configuration."""
    speculative_config: Optional[SpeculativeConfig] = None
    """Speculative decoding configuration."""
    decoding_config: DecodingConfig = field(default_factory=DecodingConfig)
    """Decoding configuration."""
    observability_config: Optional[ObservabilityConfig] = None
    """Observability configuration."""
    quant_config: Optional[QuantizationConfig] = None
    """Quantization configuration."""
    compilation_config: CompilationConfig = field(
        default_factory=CompilationConfig)
    """`torch.compile` and cudagraph capture configuration for the model.

    As a shorthand, `-O<n>` can be used to directly specify the compilation
    level `n`: `-O3` is equivalent to `-O.level=3` (same as `-O='{"level":3}'`).
    Currently, -O <n> and -O=<n> are supported as well but this will likely be
    removed in favor of clearer -O<n> syntax in the future.

    NOTE: level 0 is the default level without any optimization. level 1 and 2
    are for internal testing only. level 3 is the recommended level for
    production, also default in V1.

    You can specify the full compilation config like so:
    `{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    kv_transfer_config: Optional[KVTransferConfig] = None
    """The configurations for distributed KV cache transfer."""
    kv_events_config: Optional[KVEventsConfig] = None
    """The configurations for event publishing."""
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: Union[dict, SupportsHash] = field(default_factory=dict)
    """Additional config for specified platform. Different platforms may
    support different configs. Make sure the configs are valid for the platform
    you are using. Contents must be hashable."""
    instance_id: str = ""
    """The ID of the vLLM instance."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # summarize vllm config
        vllm_factors: list[Any] = []
        from vllm import __version__
        vllm_factors.append(__version__)
        vllm_factors.append(envs.VLLM_USE_V1)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.device_config:
            vllm_factors.append(self.device_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.load_config:
            vllm_factors.append(self.load_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
            # LoRA creates static buffers based on max_num_batched_tokens.
            # The tensor sizes and strides get captured in the torch.compile
            # graph explicitly.
            vllm_factors.append(
                str(self.scheduler_config.max_num_batched_tokens))
        else:
            vllm_factors.append("None")
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.decoding_config:
            vllm_factors.append(self.decoding_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.observability_config:
            vllm_factors.append(self.observability_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.quant_config:
            pass  # should be captured by model_config.quantization
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.kv_transfer_config:
            vllm_factors.append(self.kv_transfer_config.compute_hash())
        else:
            vllm_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                additional_config_hash = hashlib.md5(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                additional_config_hash = additional_config.compute_hash()
            vllm_factors.append(additional_config_hash)
        else:
            vllm_factors.append("None")
        factors.append(vllm_factors)

        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()[:10]
        return hash_str

    def pad_for_cudagraph(self, batch_size: int) -> int:
        # if batch_size > self.compilation_config.max_capture_size,
        # it should raise an IndexError.
        # the caller should make sure the batch_size is within the range,
        # i.e., batch_size <= self.compilation_config.max_capture_size
        return self.compilation_config.bs_to_padded_graph_size[batch_size]

    @staticmethod
    def _get_quantization_config(
            model_config: ModelConfig,
            load_config: LoadConfig) -> Optional[QuantizationConfig]:
        """Get the quantization config."""
        from vllm.platforms import current_platform
        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import (
                get_quant_config)
            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()

            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}.")
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}")
            return quant_config
        return None

    @staticmethod
    def get_quantization_config(
            model_config: ModelConfig,
            load_config: LoadConfig) -> Optional[QuantizationConfig]:
        import copy

        # For some reason, the _ version of this modifies the model_config
        # object, so using deepcopy to avoid this problem.
        return VllmConfig._get_quantization_config(copy.deepcopy(model_config),
                                                   load_config)

    def with_hf_config(
        self,
        hf_config: PretrainedConfig,
        architectures: Optional[list[str]] = None,
    ) -> "VllmConfig":
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        model_config = copy.deepcopy(self.model_config)
        model_config.hf_config = hf_config

        return replace(self, model_config=model_config)

    def __post_init__(self):
        """Verify configs are valid & consistent with each other.
        """

        self.try_verify_and_update_config()

        if self.model_config is not None:
            self.model_config.verify_async_output_proc(self.parallel_config,
                                                       self.speculative_config,
                                                       self.device_config)
            self.model_config.verify_with_parallel_config(self.parallel_config)
            self.model_config.verify_dual_chunk_attention_config(
                self.load_config)

        self.cache_config.verify_with_parallel_config(self.parallel_config)

        if self.lora_config is not None:
            self.lora_config.verify_with_cache_config(self.cache_config)
            self.lora_config.verify_with_model_config(self.model_config)

        if self.quant_config is None and self.model_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config)

        from vllm.platforms import current_platform
        if self.model_config is not None and \
            self.scheduler_config.chunked_prefill_enabled and \
            self.model_config.dtype == torch.float32 and \
            current_platform.get_device_capability() == (7, 5):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels.")

        # If the user does not explicitly set a compilation level, then
        # we use the default level. The default level depends on other
        # settings (see the below code).
        if self.compilation_config.level is None:
            if envs.VLLM_USE_V1:
                if (self.model_config is not None
                        and not self.model_config.enforce_eager):
                    self.compilation_config.level = CompilationLevel.PIECEWISE
                else:
                    self.compilation_config.level = \
                            CompilationLevel.NO_COMPILATION

            else:
                # NB: Passing both --enforce-eager and a compilation level
                # in V0 means the compilation level wins out.
                self.compilation_config.level = CompilationLevel.NO_COMPILATION

        # async tp is built on top of sequence parallelism
        # and requires it to be enabled.
        if self.compilation_config.pass_config.enable_async_tp:
            self.compilation_config.pass_config.enable_sequence_parallelism = \
                True
        if self.compilation_config.pass_config.enable_sequence_parallelism:
            self.compilation_config.custom_ops.append("+rms_norm")

        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            # if cudagraph_mode is not explicitly set by users, set default
            # value
            if self.compilation_config.cudagraph_mode is None:
                if envs.VLLM_USE_V1 and self.compilation_config.level \
                    == CompilationLevel.PIECEWISE:
                    self.compilation_config.cudagraph_mode = \
                        CUDAGraphMode.PIECEWISE
                else:
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

            # disable cudagraph when enforce eager execution
            if self.model_config is not None and \
                    self.model_config.enforce_eager:
                logger.info("Cudagraph is disabled under eager mode")
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            elif envs.VLLM_USE_V1:
                self.compilation_config.cudagraph_num_of_warmups = 1

            self._set_cudagraph_sizes()
        else:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if self.cache_config.cpu_offload_gb > 0 and \
            self.compilation_config.level != CompilationLevel.NO_COMPILATION \
                and not envs.VLLM_USE_V1:
            logger.warning(
                "CPU offload is not supported with `torch.compile` in v0 yet."
                " Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if self.cache_config.kv_sharing_fast_prefill:
            if not envs.VLLM_USE_V1:
                raise NotImplementedError(
                    "Fast prefill optimization for KV sharing is not supported "
                    "in V0 currently.")

            if self.speculative_config is not None and \
                self.speculative_config.use_eagle():
                raise NotImplementedError(
                    "Fast prefill optimization for KV sharing is not "
                    "compatible with EAGLE as EAGLE requires correct logits "
                    "for all tokens while fast prefill gives incorrect logits "
                    "for prompt tokens.")

            logger.warning_once(
                "--kv-sharing-fast-prefill requires changes on model side for "
                "correctness and to realize prefill savings. ")

        if ((not envs.VLLM_USE_V1) and self.lora_config is not None
                and self.compilation_config.level
                != CompilationLevel.NO_COMPILATION):
            logger.warning(
                "LoRA for V0 is not supported with `torch.compile` yet. "
                "Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        disable_chunked_prefill_reasons: list[str] = []

        if self.model_config:
            if self.model_config.pooler_config:
                pooling_type = self.model_config.pooler_config.pooling_type
                if pooling_type is None or pooling_type.lower() != "last":
                    disable_chunked_prefill_reasons.append(
                        "Only \"last\" pooling supports chunked "
                        "prefill and prefix caching; disabling both.")
                if not getattr(self.model_config.hf_config, "is_causal", True):
                    disable_chunked_prefill_reasons.append(
                        "Only models using causal attention supports chunked "
                        "prefill and prefix caching; disabling both.")
            elif self.model_config.is_encoder_decoder:
                self.scheduler_config.max_num_encoder_input_tokens = \
                    MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(self.model_config)
                logger.debug(
                    "Encoder-decoder model detected: setting "
                    "`max_num_encoder_input_tokens` to encoder length (%s)",
                    self.scheduler_config.max_num_encoder_input_tokens)
                self.scheduler_config.disable_chunked_mm_input = True
                disable_chunked_prefill_reasons.append(
                    "Encoder-decoder models do not support chunked prefill nor"
                    " prefix caching; disabling both.")
                if (self.model_config.architecture
                        == "WhisperForConditionalGeneration"
                        and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
                        != "spawn"):
                    logger.warning(
                        "Whisper is known to have issues with "
                        "forked workers. If startup is hanging, "
                        "try setting 'VLLM_WORKER_MULTIPROC_METHOD' "
                        "to 'spawn'.")

        if disable_chunked_prefill_reasons:
            for reason in disable_chunked_prefill_reasons:
                logger.info(reason)
            self.scheduler_config.chunked_prefill_enabled = False
            self.scheduler_config.long_prefill_token_threshold = 0

            if self.cache_config is not None:
                self.cache_config.enable_prefix_caching = False

        if (self.kv_events_config is not None
                and self.kv_events_config.enable_kv_cache_events
                and not self.cache_config.enable_prefix_caching):
            logger.warning(
                "KV cache events are on, but prefix caching is not enabled."
                "Use --enable-prefix-caching to enable.")
        if (self.kv_events_config is not None
                and self.kv_events_config.publisher != "null"
                and not self.kv_events_config.enable_kv_cache_events):
            logger.warning("KV cache events are disabled,"
                           "but the scheduler is configured to publish them."
                           "Modify KVEventsConfig.enable_kv_cache_events"
                           "to True to enable.")
        current_platform.check_and_update_config(self)

        # final check of cudagraph mode after platform-specific update
        if envs.VLLM_USE_V1 and current_platform.is_cuda_alike():
            if self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL \
                and self.model_config is not None and \
                not self.model_config.disable_cascade_attn:
                logger.info("CUDAGraphMode.FULL is not supported with "
                            "cascade attention currently. Disabling cascade"
                            "attention.")
                self.model_config.disable_cascade_attn = True

            if self.compilation_config.cudagraph_mode\
                .requires_piecewise_compilation():
                assert self.compilation_config.level == \
                    CompilationLevel.PIECEWISE, \
                    "Compilation level should be CompilationLevel.PIECEWISE "\
                    "when cudagraph_mode piecewise cudagraphs is used, "\
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"

        if self.parallel_config.enable_dbo:
            a2a_backend = envs.VLLM_ALL2ALL_BACKEND
            assert a2a_backend == "deepep_low_latency", \
            "Microbatching currently only supports the deepep_low_latency "\
            f"all2all backend. {a2a_backend} is not supported. To fix set "\
            "the VLLM_ALL2ALL_BACKEND environment variable to "\
            "deepep_low_latency and install the DeepEP kerenls."

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        # Do this after all the updates to compilation_config.level
        if envs.VLLM_USE_V1 and \
            self.compilation_config.level == CompilationLevel.PIECEWISE:
            self.compilation_config.set_splitting_ops_for_v1()

        if (envs.VLLM_USE_V1
                and not self.scheduler_config.disable_hybrid_kv_cache_manager):
            # logger should only print warning message for hybrid models. As we
            # can't know whether the model is hybrid or not now, so we don't log
            # warning message here and will log it later.
            if not current_platform.support_hybrid_kv_cache():
                # Hybrid KV cache manager is not supported on non-GPU platforms.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_transfer_config is not None:
                # Hybrid KV cache manager is not compatible with KV transfer.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.kv_events_config is not None:
                # Hybrid KV cache manager is not compatible with KV events.
                self.scheduler_config.disable_hybrid_kv_cache_manager = True
            if self.model_config is not None and \
                self.model_config.attention_chunk_size is not None:
                if self.speculative_config is not None and \
                    self.speculative_config.use_eagle():
                    # Hybrid KV cache manager is not yet supported with chunked
                    # local attention + eagle.
                    self.scheduler_config.disable_hybrid_kv_cache_manager = True
                elif \
                    not envs.VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE:
                    logger.warning(
                        "There is a latency regression when using chunked local"
                        " attention with the hybrid KV cache manager. Disabling"
                        " it, by default. To enable it, set the environment "
                        "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1."
                    )
                    # Hybrid KV cache manager is not yet supported with chunked
                    # local attention.
                    self.scheduler_config.disable_hybrid_kv_cache_manager = True

    def update_sizes_for_sequence_parallelism(self,
                                              possible_sizes: list) -> list:
        # remove the sizes that not multiple of tp_size when
        # enable sequence parallelism
        removed_sizes = [
            size for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size != 0
        ]
        if removed_sizes:
            logger.warning(
                "Batch sizes %s are removed because they are not "
                "multiple of tp_size %d when "
                "sequence parallelism is enabled", removed_sizes,
                self.parallel_config.tensor_parallel_size)

        return [
            size for size in possible_sizes
            if size % self.parallel_config.tensor_parallel_size == 0
        ]

    def _set_cudagraph_sizes(self):
        """
        vLLM defines the default candidate list of batch sizes for CUDA graph
        capture as:

        ```python
        max_graph_size = min(max_num_seqs * 2, 512)
        # 1, 2, 4, then multiples of 8 up to max_graph_size
        cuda_graph_sizes = [1, 2, 4, 8, 16, 24, 32, 40, ..., max_graph_size]

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in descending order).

        These sizes are used to capture and reuse CUDA graphs for
        performance-critical paths (e.g., decoding). Capturing enables
        significantly faster kernel dispatch by avoiding Python overhead. The
        list is then filtered based on `max_num_batched_tokens` (e.g., 8192 on
        most GPUs), which controls the total allowed number of tokens in a
        batch. Since each sequence may have a variable number of tokens, the
        maximum usable batch size will depend on actual sequence lengths.

        Example:
            With `max_num_batched_tokens = 8192`, and typical sequences
            averaging ~32 tokens, most practical batch sizes fall below 256.
            However, the system will still allow capture sizes up to 512 if
            shape and memory permit.

        Note:
            If users explicitly specify cudagraph capture sizes in the
            compilation config, those will override this default logic.
            At runtime:

            - If batch size <= one of the `cudagraph_capture_sizes`, the closest
            padded CUDA graph will be used.
            - If batch size > largest `cudagraph_capture_sizes`, cudagraph will
            not be used.
        """

        # calculate the default `batch_size_capture_list`
        if not envs.VLLM_USE_V1:
            batch_size_capture_list = []
            if self.scheduler_config is not None and \
                self.model_config is not None and \
                    not self.model_config.enforce_eager:

                possible_sizes = [1, 2, 4] + [8 * i for i in range(1, 1025)]
                if self.parallel_config.tensor_parallel_size > 1 and \
                    self.compilation_config.pass_config.enable_sequence_parallelism:
                    possible_sizes = self.update_sizes_for_sequence_parallelism(
                        possible_sizes)

                # find the minimum size that is larger than max_num_seqs,
                # which then becomes the max_batchsize_to_capture
                larger_sizes = [
                    x for x in possible_sizes
                    if x >= self.scheduler_config.max_num_seqs
                ]
                if larger_sizes:
                    max_batchsize_to_capture = larger_sizes[0]
                else:
                    max_batchsize_to_capture = possible_sizes[-1]

                # filter out the sizes that are
                # larger than max_batchsize_to_capture
                batch_size_capture_list = [
                    size for size in possible_sizes
                    if size <= max_batchsize_to_capture
                ]
        else:
            batch_size_capture_list = []
            if self.model_config is not None and \
                not self.model_config.enforce_eager:
                cuda_graph_sizes = self.scheduler_config.cuda_graph_sizes
                if len(cuda_graph_sizes) == 1:
                    batch_size_capture_list = [1, 2, 4] + [
                        i for i in range(8, cuda_graph_sizes[0] + 1, 8)
                    ]
                elif len(cuda_graph_sizes) > 1:
                    batch_size_capture_list = sorted(cuda_graph_sizes)
                else:
                    raise TypeError(f"Invalid value for {cuda_graph_sizes=}.")
                if self.parallel_config.tensor_parallel_size > 1 and \
                    self.compilation_config.pass_config.enable_sequence_parallelism:
                    batch_size_capture_list = \
                        self.update_sizes_for_sequence_parallelism(batch_size_capture_list)
                max_num_tokens = self.scheduler_config.max_num_batched_tokens
                batch_size_capture_list = [
                    size for size in batch_size_capture_list
                    if size <= max_num_tokens
                ]

        self.compilation_config.init_with_cudagraph_sizes(
            batch_size_capture_list)

    def recalculate_max_model_len(self, max_model_len: int):
        # Can only be called in try_verify_and_update_config
        model_config = self.model_config
        max_model_len = model_config.get_and_verify_max_len(max_model_len)
        self.model_config.max_model_len = max_model_len
        self.scheduler_config.max_model_len = max_model_len

    def try_verify_and_update_config(self):
        if self.model_config is None:
            return

        # Avoid running try_verify_and_update_config multiple times
        if getattr(self.model_config, "config_updated", False):
            return
        self.model_config.config_updated = True

        architecture = self.model_config.architecture
        if architecture is None:
            return

        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP, HybridAttentionMambaModelConfig)
        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_config(self)

        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)

        if self.model_config.convert_type == "classify":
            # Maybe convert ForCausalLM into ForSequenceClassification model.
            from vllm.model_executor.models.adapters import (
                SequenceClassificationConfig)
            SequenceClassificationConfig.verify_and_update_config(self)

    def __str__(self):
        return (
            f"model={self.model_config.model!r}, "
            f"speculative_config={self.speculative_config!r}, "
            f"tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init}, "
            f"tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len}, "
            f"download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size}, "  # noqa
            f"pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"data_parallel_size={self.parallel_config.data_parallel_size}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f"device_config={self.device_config.device}, "
            f"decoding_config={self.decoding_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"chunked_prefill_enabled={self.scheduler_config.chunked_prefill_enabled}, "  # noqa
            f"use_async_output_proc={self.model_config.use_async_output_proc}, "
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}")


_current_vllm_config: Optional[VllmConfig] = None
_current_prefix: Optional[str] = None


@contextmanager
def set_current_vllm_config(vllm_config: VllmConfig,
                            check_compile=False,
                            prefix: Optional[str] = None):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    from vllm.compilation.counter import compilation_counter
    num_models_seen = compilation_counter.num_models_seen
    try:
        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        logger.debug("enabled custom ops: %s",
                     vllm_config.compilation_config.enabled_custom_ops)
        logger.debug("disabled custom ops: %s",
                     vllm_config.compilation_config.disabled_custom_ops)
        if check_compile and \
            vllm_config.compilation_config.level == CompilationLevel.PIECEWISE \
            and compilation_counter.num_models_seen == num_models_seen:
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                vllm_config.model_config.model)
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix
        # Clear the compilation config cache when context changes
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_vllm_config()"""
    return get_current_vllm_config().compilation_config


def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the vllm config. In that case, we set a default
        # config.
        logger.warning("Current vLLM config is not set.")
        from vllm.config import VllmConfig
        return VllmConfig()
    return _current_vllm_config


def get_current_model_prefix() -> str:
    """
    Get the prefix of the model that's currently being initialized.
    """
    assert _current_prefix is not None, \
        "Current model prefix is not set. "
    return _current_prefix


def contains_object_print(text):
    """
    Check if the text looks like a printed Python object, e.g.
    contains any substring matching the pattern: "at 0xFFFFFFF>"
    We match against 0x followed by 2-16 hex chars (there's
    a max of 16 on a 64-bit system).

    Args:
        text (str): The text to check

    Returns:
        result (bool): `True` if a match is found, `False` otherwise.
    """
    pattern = r'at 0x[a-fA-F0-9]{2,16}>'
    match = re.search(pattern, text)
    return match is not None


def assert_hashable(text):
    if not contains_object_print(text):
        return True
    raise AssertionError(
        f"vLLM tried to hash some configs that may have Python objects ids "
        f"in them. This is a bug, please file an issue. "
        f"Text being hashed: {text}")


T = TypeVar("T")


def get_layers_from_vllm_config(
        vllm_config: VllmConfig,
        layer_type: type[T],
        layer_names: Optional[list[str]] = None) -> dict[str, T]:
    """
    Get layers from the vLLM config.

    Args:
        vllm_config: The vLLM config.
        layer_type: The type of the layer to get.
        layer_names: The names of the layers to get. If None, return all layers.
    """

    if layer_names is None:
        layer_names = list(
            vllm_config.compilation_config.static_forward_context.keys())

    forward_context = vllm_config.compilation_config.static_forward_context

    return {
        layer_name: forward_context[layer_name]
        for layer_name in layer_names
        if isinstance(forward_context[layer_name], layer_type)
    }


@config
@dataclass
class SpeechToTextConfig:
    """Configuration for speech-to-text models."""

    sample_rate: float = 16_000
    """Sample rate (Hz) to resample input audio to. Most speech models expect
    16kHz audio input. The input audio will be automatically resampled to this
    rate before processing."""

    max_audio_clip_s: int = 30
    """Maximum duration in seconds for a single audio clip without chunking.
    Audio longer than this will be split into smaller chunks if
    `allow_audio_chunking` evaluates to True, otherwise it will be rejected."""

    overlap_chunk_second: int = 1
    """Overlap duration in seconds between consecutive audio chunks when
    splitting long audio. This helps maintain context across chunk boundaries
    and improves transcription quality at split points."""

    min_energy_split_window_size: Optional[int] = 1600
    """Window size in samples for finding low-energy (quiet) regions to split
    audio chunks. The algorithm looks for the quietest moment within this
    window to minimize cutting through speech. Default 1600 samples ≈ 100ms
    at 16kHz. If None, no chunking will be done."""

    @property
    def allow_audio_chunking(self) -> bool:
        return self.min_energy_split_window_size is not None


def update_config(config: DataclassInstanceT,
                  overrides: dict[str, Any]) -> DataclassInstanceT:
    processed_overrides = {}
    for field_name, value in overrides.items():
        assert hasattr(
            config, field_name), f"{type(config)} has no field `{field_name}`"
        current_value = getattr(config, field_name)
        if is_dataclass(current_value) and not is_dataclass(value):
            assert isinstance(value, dict), (
                f"Overrides to {type(config)}.{field_name} must be a dict"
                f"  or {type(current_value)}, but got {type(value)}")
            value = update_config(
                current_value,  # type: ignore[type-var]
                value)
        processed_overrides[field_name] = value
    return replace(config, **processed_overrides)
