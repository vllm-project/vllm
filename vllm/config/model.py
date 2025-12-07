# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from collections.abc import Callable
from dataclasses import InitVar, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import torch
from pydantic import ConfigDict, SkipValidation, field_validator, model_validator
from pydantic.dataclasses import dataclass
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers.configuration_utils import ALLOWED_LAYER_TYPES

import vllm.envs as envs
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode, MultiModalConfig
from vllm.config.pooler import PoolerConfig
from vllm.config.scheduler import RunnerType
from vllm.config.utils import config, getattr_iter
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.transformers_utils.config import (
    ConfigFormat,
    get_config,
    get_hf_image_processor_config,
    get_hf_text_config,
    get_pooling_config,
    get_sentence_transformer_tokenizer_config,
    is_encoder_decoder,
    try_get_dense_modules,
    try_get_generation_config,
    try_get_safetensors_metadata,
    try_get_tokenizer_config,
    uses_mrope,
    uses_xdrope_dim,
)
from vllm.transformers_utils.gguf_utils import (
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from vllm.transformers_utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.utils.import_utils import LazyLoader
from vllm.utils.torch_utils import common_broadcastable_dtype

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
    import vllm.model_executor.models as me_models
    from vllm.config.load import LoadConfig
    from vllm.config.parallel import ParallelConfig
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.v1.sample.logits_processor import LogitsProcessor
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )
    me_models = LazyLoader("model_executor", globals(), "vllm.model_executor.models")
    LoadConfig = Any
    ParallelConfig = Any
    QuantizationMethods = Any
    LogitsProcessor = Any

logger = init_logger(__name__)

RunnerOption = Literal["auto", RunnerType]
ConvertType = Literal["none", "embed", "classify", "reward"]
ConvertOption = Literal["auto", ConvertType]
TaskOption = Literal[
    "auto",
    "generate",
    "embedding",
    "embed",
    "classify",
    "score",
    "reward",
    "transcription",
    "draft",
]
TokenizerMode = Literal["auto", "hf", "slow", "mistral", "deepseek_v32"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
LogprobsMode = Literal[
    "raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"
]
HfOverrides = dict[str, Any] | Callable[[PretrainedConfig], PretrainedConfig]
ModelImpl = Literal["auto", "vllm", "transformers", "terratorch"]
LayerBlockType = Literal["attention", "linear_attention", "mamba"]

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

AttnTypeStr = Literal[
    "decoder", "encoder", "encoder_only", "encoder_decoder", "attention_free", "hybrid"
]


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
    task: TaskOption | None = None
    """[DEPRECATED] The task to use the model for. If the model supports more
    than one model runner, this is used to select which model runner to run.

    Note that the model may support other tasks using the same model runner.
    """
    tokenizer: SkipValidation[str] = None  # type: ignore
    """Name or path of the Hugging Face tokenizer to use. If unspecified, model
    name or path will be used."""
    tokenizer_mode: TokenizerMode | str = "auto"
    """Tokenizer mode:\n
    - "auto" will use the tokenizer from `mistral_common` for Mistral models
    if available, otherwise it will use the "hf" tokenizer.\n
    - "hf" will use the fast tokenizer if available.\n
    - "slow" will always use the slow tokenizer.\n
    - "mistral" will always use the tokenizer from `mistral_common`.\n
    - "deepseek_v32" will always use the tokenizer from `deepseek_v32`.\n
    - Other custom values can be supported via plugins."""
    trust_remote_code: bool = False
    """Trust remote code (e.g., from HuggingFace) when downloading the model
    and tokenizer."""
    dtype: ModelDType | torch.dtype = "auto"
    """Data type for model weights and activations:\n
    - "auto" will use FP16 precision for FP32 and FP16 models, and BF16
    precision for BF16 models.\n
    - "half" for FP16. Recommended for AWQ quantization.\n
    - "float16" is the same as "half".\n
    - "bfloat16" for a balance between precision and range.\n
    - "float" is shorthand for FP32 precision.\n
    - "float32" for FP32 precision."""
    seed: int = 0
    """Random seed for reproducibility.

    We must set the global seed because otherwise,
    different tensor parallel workers would sample different tokens,
    leading to inconsistent results."""
    hf_config: PretrainedConfig = field(init=False)
    """The Hugging Face config of the model."""
    hf_text_config: PretrainedConfig = field(init=False)
    """The Hugging Face config of the text model (same as hf_config for text models)."""
    hf_config_path: str | None = None
    """Name or path of the Hugging Face config to use. If unspecified, model
    name or path will be used."""
    allowed_local_media_path: str = ""
    """Allowing API requests to read local images or videos from directories
    specified by the server file system. This is a security risk. Should only
    be enabled in trusted environments."""
    allowed_media_domains: list[str] | None = None
    """If set, only media URLs that belong to this domain can be used for
    multi-modal inputs. """
    revision: str | None = None
    """The specific model version to use. It can be a branch name, a tag name,
    or a commit id. If unspecified, will use the default version."""
    code_revision: str | None = None
    """The specific revision to use for the model code on the Hugging Face Hub.
    It can be a branch name, a tag name, or a commit id. If unspecified, will
    use the default version."""
    tokenizer_revision: str | None = None
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
    spec_target_max_model_len: int | None = None
    """Specify the maximum length for spec decoding draft models."""
    quantization: SkipValidation[QuantizationMethods | None] = None
    """Method used to quantize the weights. If `None`, we first check the
    `quantization_config` attribute in the model config file. If that is
    `None`, we assume the model weights are not quantized and use `dtype` to
    determine the data type of the weights."""
    enforce_eager: bool = False
    """Whether to always use eager-mode PyTorch. If True, we will disable CUDA
    graph and always execute the model in eager mode. If False, we will use
    CUDA graph and eager execution in hybrid for maximal performance and
    flexibility."""
    max_logprobs: int = 20
    """Maximum number of log probabilities to return when `logprobs` is
    specified in `SamplingParams`. The default value comes the default for the
    OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length *
    vocab_size) logprobs are allowed to be returned and it may cause OOM."""
    logprobs_mode: LogprobsMode = "raw_logprobs"
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
    `prompt_embeds` key.

    WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed.
    Only enable this flag for trusted users!"""
    served_model_name: str | list[str] | None = None
    """The model name(s) used in the API. If multiple names are provided, the
    server will respond to any of the provided names. The model name in the
    model field of a response will be the first name in this list. If not
    specified, the model name will be the same as the `--model` argument. Noted
    that this name(s) will also be used in `model_name` tag content of
    prometheus metrics, if multiple names provided, metrics tag will take the
    first one."""
    config_format: str | ConfigFormat = "auto"
    """The format of the model config to load:\n
    - "auto" will try to load the config in hf format if available after trying
    to load in mistral format.\n
    - "hf" will load the config in hf format.\n
    - "mistral" will load the config in mistral format."""
    hf_token: bool | str | None = None
    """The token to use as HTTP bearer authorization for remote files . If
    `True`, will use the token generated when running `huggingface-cli login`
    (stored in `~/.huggingface`)."""
    hf_overrides: HfOverrides = field(default_factory=dict)
    """If a dictionary, contains arguments to be forwarded to the Hugging Face
    config. If a callable, it is called to update the HuggingFace config."""
    logits_processor_pattern: str | None = None
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
    """Enable sleep mode for the engine (only cuda and
    hip platforms are supported)."""
    model_impl: str | ModelImpl = "auto"
    """Which implementation of the model to use:\n
    - "auto" will try to use the vLLM implementation, if it exists, and fall
    back to the Transformers implementation if no vLLM implementation is
    available.\n
    - "vllm" will use the vLLM model implementation.\n
    - "transformers" will use the Transformers model implementation.\n
    - "terratorch" will use the TerraTorch model implementation.
    """
    override_attention_dtype: str | None = None
    """Override dtype for attention"""
    logits_processors: list[str | type[LogitsProcessor]] | None = None
    """One or more logits processors' fully-qualified class names or class
    definitions"""
    io_processor_plugin: str | None = None
    """IOProcessor plugin name to load at model startup"""

    # Pooler config
    pooler_config: PoolerConfig | None = None
    """Pooler config which controls the behaviour of output pooling in pooling
    models."""

    # Multimodal config and init vars
    multimodal_config: MultiModalConfig | None = None
    """Configuration for multimodal model. If `None`, this will be inferred
    from the architecture of `self.model`."""
    limit_mm_per_prompt: InitVar[dict[str, int | dict[str, int]] | None] = None
    enable_mm_embeds: InitVar[bool | None] = None
    media_io_kwargs: InitVar[dict[str, dict[str, Any]] | None] = None
    mm_processor_kwargs: InitVar[dict[str, Any] | None] = None
    mm_processor_cache_gb: InitVar[float | None] = None
    mm_processor_cache_type: InitVar[MMCacheType | None] = None
    mm_shm_cache_max_object_size_mb: InitVar[int | None] = None
    mm_encoder_tp_mode: InitVar[MMEncoderTPMode | None] = None
    mm_encoder_attn_backend: InitVar[AttentionBackendEnum | str | None] = None
    interleave_mm_strings: InitVar[bool | None] = None
    skip_mm_profiling: InitVar[bool | None] = None
    video_pruning_rate: InitVar[float | None] = None

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
        ignored_factors = {
            "runner",
            "convert",
            "task",
            "tokenizer",
            "tokenizer_mode",
            "seed",
            "hf_config_path",
            "allowed_local_media_path",
            "allowed_media_domains",
            "tokenizer_revision",
            "spec_target_max_model_len",
            "enforce_eager",
            "logprobs_mode",
            "disable_cascade_attn",
            "skip_tokenizer_init",
            "served_model_name",
            "config_format",
            "hf_token",
            "hf_overrides",
            "logits_processor_pattern",
            "override_attention_dtype",
            "logits_processors",
            "io_processor_plugin",
            "pooler_config",
            "multimodal_config",
            "limit_mm_per_prompt",
            "media_io_kwargs",
            "mm_processor_kwargs",
            "mm_processor_cache_gb",
            "mm_processor_cache_type",
            "mm_shm_cache_max_object_size_mb",
            "mm_encoder_tp_mode",
            "interleave_mm_strings",
            "skip_mm_profiling",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def _update_nested(
        self,
        target: PretrainedConfig | dict[str, Any],
        updates: dict[str, Any],
    ) -> None:
        """Recursively updates a config or dict with nested updates."""
        for key, value in updates.items():
            if isinstance(value, dict):
                # Get the nested target
                if isinstance(target, dict):
                    nested_target = target.get(key)
                else:
                    nested_target = getattr(target, key, None)

                # If nested target exists and can be updated recursively
                if nested_target is not None and (
                    isinstance(nested_target, dict)
                    or hasattr(nested_target, "__dict__")
                ):
                    self._update_nested(nested_target, value)
                    continue

            # Set the value (base case)
            if isinstance(target, dict):
                target[key] = value
            else:
                setattr(target, key, value)

    def _apply_dict_overrides(
        self,
        config: PretrainedConfig,
        overrides: dict[str, Any],
    ) -> None:
        """Apply dict overrides, handling both nested configs and dict values."""
        from transformers import PretrainedConfig

        for key, value in overrides.items():
            attr = getattr(config, key, None)
            if attr is not None and isinstance(attr, PretrainedConfig):
                # It's a nested config - recursively update it
                self._update_nested(attr, value)
            else:
                # It's a dict-valued parameter - set it directly
                setattr(config, key, value)

    def __post_init__(
        self,
        # Multimodal config init vars
        limit_mm_per_prompt: dict[str, int | dict[str, int]] | None,
        enable_mm_embeds: bool | None,
        media_io_kwargs: dict[str, dict[str, Any]] | None,
        mm_processor_kwargs: dict[str, Any] | None,
        mm_processor_cache_gb: float | None,
        mm_processor_cache_type: MMCacheType | None,
        mm_shm_cache_max_object_size_mb: int | None,
        mm_encoder_tp_mode: MMEncoderTPMode | None,
        mm_encoder_attn_backend: AttentionBackendEnum | str | None,
        interleave_mm_strings: bool | None,
        skip_mm_profiling: bool | None,
        video_pruning_rate: float | None,
    ) -> None:
        # Keep set served_model_name before maybe_model_redirect(self.model)
        self.served_model_name = get_served_model_name(
            self.model, self.served_model_name
        )
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
            dict_overrides: dict[str, Any] = {}
        else:
            # Separate dict overrides from flat ones
            # We'll determine how to apply dict overrides after loading the config
            hf_overrides_kw = {}
            dict_overrides = {}
            for key, value in self.hf_overrides.items():
                if isinstance(value, dict):
                    dict_overrides[key] = value
                else:
                    hf_overrides_kw[key] = value
            hf_overrides_fn = None

        self.maybe_pull_model_tokenizer_for_runai(self.model, self.tokenizer)

        from vllm.platforms import current_platform

        if self.override_attention_dtype is not None and not current_platform.is_rocm():
            warnings.warn(
                "override-attention-dtype is set but not using ROCm platform",
                stacklevel=2,
            )

        if self.enable_sleep_mode and not current_platform.is_sleep_mode_available():
            raise ValueError("Sleep mode is not supported on current platform.")

        hf_config = get_config(
            self.hf_config_path or self.model,
            self.trust_remote_code,
            self.revision,
            self.code_revision,
            self.config_format,
            hf_overrides_kw=hf_overrides_kw,
            hf_overrides_fn=hf_overrides_fn,
        )
        hf_config = maybe_patch_hf_config_from_gguf(
            self.model,
            hf_config,
        )

        self.hf_config = hf_config
        if dict_overrides:
            self._apply_dict_overrides(hf_config, dict_overrides)
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.attention_chunk_size = getattr(
            self.hf_text_config, "attention_chunk_size", None
        )
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, hf_token=self.hf_token, revision=self.revision
        )

        architectures = self.architectures
        registry = self.registry
        is_generative_model = registry.is_text_generation_model(architectures, self)
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
            msg_prefix = (
                "The 'task' option has been deprecated and will be "
                "removed in v0.13.0 or v1.0, whichever comes first."
            )
            msg_hint = "Please remove this option."

            is_generative_task = self.task in _RUNNER_TASKS["generate"]
            is_pooling_task = self.task in _RUNNER_TASKS["pooling"]

            if is_generative_model and is_pooling_model:
                if is_generative_task:
                    runner = "generate"
                    convert = "auto"
                    msg_hint = (
                        "Please replace this option with `--runner "
                        "generate` to continue using this model "
                        "as a generative model."
                    )
                elif is_pooling_task:
                    runner = "pooling"
                    convert = "auto"
                    msg_hint = (
                        "Please replace this option with `--runner "
                        "pooling` to continue using this model "
                        "as a pooling model."
                    )
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
                    msg_hint = (
                        "Please replace this option with `--convert "
                        f"{convert}` to continue using this model "
                        "as a pooling model."
                    )
                else:  # task == "auto"
                    pass
            else:
                # Neither generative nor pooling model - try to convert if possible
                if is_pooling_task:
                    runner = "pooling"
                    convert = _task_to_convert(self.task)
                    msg_hint = (
                        "Please replace this option with `--runner pooling "
                        f"--convert {convert}` to continue using this model "
                        "as a pooling model."
                    )
                else:
                    debug_info = {
                        "architectures": architectures,
                        "is_generative_model": is_generative_model,
                        "is_pooling_model": is_pooling_model,
                    }
                    raise AssertionError(
                        "The model should be a generative or "
                        "pooling model when task is set to "
                        f"{self.task!r}. Found: {debug_info}"
                    )

            self.runner = runner
            self.convert = convert

            msg = f"{msg_prefix} {msg_hint}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        self.runner_type = self._get_runner_type(architectures, self.runner)
        self.convert_type = self._get_convert_type(
            architectures, self.runner_type, self.convert
        )

        if self.runner_type == "generate" and not is_generative_model:
            generate_converts = _RUNNER_CONVERTS["generate"]
            if self.convert_type not in generate_converts:
                # Currently we don't have any converters for generative models
                raise ValueError("This model does not support `--runner generate`.")
        if self.runner_type == "pooling" and not is_pooling_model:
            pooling_converts = _RUNNER_CONVERTS["pooling"]
            if self.convert_type not in pooling_converts:
                convert_option = "<" + "|".join(pooling_converts) + ">"
                raise ValueError(
                    "This model does not support `--runner pooling`. "
                    f"You can pass `--convert {convert_option} to adapt "
                    "it into a pooling model."
                )

        # Note: Initialize these attributes early because transformers fallback
        # may fail to load dynamic modules in child processes
        model_info, arch = registry.inspect_model_cls(architectures, self)
        self._model_info = model_info
        self._architecture = arch
        logger.info("Resolved architecture: %s", arch)

        # Init pooler config if needed
        if self.runner_type == "pooling":
            if self.pooler_config is None:
                self.pooler_config = PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                for k, v in base_config.items():
                    if getattr(self.pooler_config, k) is None:
                        setattr(self.pooler_config, k, v)

            default_pooling_type = self._model_info.default_pooling_type
            if self.pooler_config.pooling_type is None:
                self.pooler_config.pooling_type = default_pooling_type

        self.dtype: torch.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            is_pooling_model=self.runner_type == "pooling",
            revision=self.revision,
        )

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
        # Init multimodal config if needed
        if self._model_info.supports_multimodal:
            if (
                mm_encoder_tp_mode == "data"
                and not self._model_info.supports_multimodal_encoder_tp_data
            ):
                logger.warning_once(
                    "This model does not support `--mm-encoder-tp-mode data`. "
                    "Falling back to `--mm-encoder-tp-mode weights`."
                )
                mm_encoder_tp_mode = "weights"

            mm_config_kwargs = dict(
                limit_per_prompt=limit_mm_per_prompt,
                enable_mm_embeds=enable_mm_embeds,
                media_io_kwargs=media_io_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                mm_processor_cache_gb=mm_processor_cache_gb,
                mm_processor_cache_type=mm_processor_cache_type,
                mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
                mm_encoder_tp_mode=mm_encoder_tp_mode,
                mm_encoder_attn_backend=mm_encoder_attn_backend,
                interleave_mm_strings=interleave_mm_strings,
                skip_mm_profiling=skip_mm_profiling,
                video_pruning_rate=video_pruning_rate,
            )

            mm_config_kwargs = {
                k: v for k, v in mm_config_kwargs.items() if v is not None
            }

            self.multimodal_config = MultiModalConfig(**mm_config_kwargs)

        # Multimodal GGUF models must use original repo for mm processing
        if is_gguf(self.tokenizer) and self.is_multimodal_model:
            raise ValueError(
                "Loading a multimodal GGUF model needs to use original "
                "tokenizer. Please specify the unquantized hf model's "
                "repo name or path using the --tokenizer argument."
            )

        if self.disable_sliding_window:
            # Set after get_and_verify_max_len to ensure that max_model_len
            # can be correctly capped to sliding window size
            self.hf_text_config.sliding_window = None

        # Avoid running try_verify_and_update_config multiple times
        self.config_updated = False

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

    @field_validator("tokenizer_mode", mode="after")
    def _lowercase_tokenizer_mode(cls, tokenizer_mode: str) -> str:
        return tokenizer_mode.lower()

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
            raise ValueError("max_model_len must be an integer after __post_init__.")
        return self

    def _get_transformers_backend_cls(self) -> str:
        """Determine which Transformers modeling backend class will be used if
        `model_impl` is set to `transformers` or `auto`."""
        cls = "Transformers"
        # If 'hf_config != hf_text_config' it's a nested config, i.e. multimodal
        cls += "MultiModal" if self.hf_config != self.hf_text_config else ""
        cls += "MoE" if self.get_num_experts() > 1 else ""
        # Check if the architecture we're wrapping has defaults
        runner = None
        task = None
        if defaults := try_match_architecture_defaults(self.architectures[0]):
            _, (runner, task) = defaults
        # User specified value take precedence
        if self.runner != "auto":
            runner = self.runner
        # Only consider Transformers modeling backend pooling classes if we're wrapping
        # an architecture that defaults to pooling. Otherwise, we return the LM class
        # and use adapters.
        if runner == "pooling" and task in {"embed", "classify"}:
            if task == "embed":
                cls += "EmbeddingModel"
            elif task == "classify":
                cls += "ForSequenceClassification"
        else:
            cls += "ForCausalLM"
        return cls

    def using_transformers_backend(self) -> bool:
        """Check if the model is using the Transformers modeling backend class."""
        used_cls = self._model_info.architecture
        transformers_backend_cls = self._get_transformers_backend_cls()
        return used_cls == transformers_backend_cls

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

    def maybe_pull_model_tokenizer_for_runai(self, model: str, tokenizer: str) -> None:
        """Pull model/tokenizer from Object Storage to temporary
        directory when needed.

        Args:
            model: Model name or path
            tokenizer: Tokenizer name or path
        """

        if not (is_runai_obj_uri(model) or is_runai_obj_uri(tokenizer)):
            return

        if is_runai_obj_uri(model):
            object_storage_model = ObjectStorageModel(url=model)
            object_storage_model.pull_files(
                model, allow_pattern=["*.model", "*.py", "*.json"]
            )
            self.model_weights = model
            self.model = object_storage_model.dir

            # If tokenizer is same as model, download to same directory
            if model == tokenizer:
                object_storage_model.pull_files(
                    model,
                    ignore_pattern=[
                        "*.pt",
                        "*.safetensors",
                        "*.bin",
                        "*.tensors",
                        "*.pth",
                    ],
                )
                self.tokenizer = object_storage_model.dir
                return

        # Only download tokenizer if needed and not already handled
        if is_runai_obj_uri(tokenizer):
            object_storage_tokenizer = ObjectStorageModel(url=tokenizer)
            object_storage_tokenizer.pull_files(
                model,
                ignore_pattern=["*.pt", "*.safetensors", "*.bin", "*.tensors", "*.pth"],
            )
            self.tokenizer = object_storage_tokenizer.dir

    def _get_encoder_config(self):
        model = self.model
        if is_remote_gguf(model):
            model, _ = split_remote_gguf(model)
        return get_sentence_transformer_tokenizer_config(model, self.revision)

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
                runner_type,
            )

        return runner_type

    def _get_default_convert_type(
        self,
        architectures: list[str],
        runner_type: RunnerType,
    ) -> ConvertType:
        registry = self.registry

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if runner_type == "generate" and registry.is_text_generation_model(
                    architectures, self
                ):
                    return "none"
                if runner_type == "pooling" and registry.is_pooling_model(
                    architectures, self
                ):
                    return "none"

            match = try_match_architecture_defaults(arch, runner_type=runner_type)
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

        convert_type = self._get_default_convert_type(architectures, runner_type)

        # Don't log the most common case
        if convert_type != "none":
            logger.info(
                "Resolved `--convert auto` to `--convert %s`. "
                "Pass the value explicitly to silence this message.",
                convert_type,
            )

        return convert_type

    def _get_default_pooling_task(
        self,
        architectures: list[str],
    ) -> Literal["embed", "classify", "reward"]:
        if self.registry.is_cross_encoder_model(architectures, self):
            return "classify"

        for arch in architectures:
            match = try_match_architecture_defaults(arch, runner_type="pooling")
            if match:
                _, (_, convert_type) = match
                assert convert_type != "none"
                return convert_type

        return "embed"

    def _parse_quant_hf_config(self, hf_config: PretrainedConfig):
        quant_cfg = getattr(hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(hf_config, "compression_config", None)

        else:
            # Set quant_method for ModelOpt models.
            producer_name = quant_cfg.get("producer", {}).get("name")
            if producer_name == "modelopt":
                quant_algo = quant_cfg.get("quantization", {}).get("quant_algo")
                if quant_algo == "FP8":
                    quant_cfg["quant_method"] = "modelopt"
                elif quant_algo == "NVFP4":
                    quant_cfg["quant_method"] = "modelopt_fp4"
                elif quant_algo is not None:
                    raise ValueError(f"Unknown ModelOpt quant algo: {quant_algo}")

        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = me_quant.QUANTIZATION_METHODS
        if self.quantization is not None:
            self.quantization = cast(me_quant.QuantizationMethods, self.quantization)

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config(self.hf_config)
        if quant_cfg is None and (
            text_config := getattr(self.hf_config, "text_config", None)
        ):
            # Check the text config as well for multi-modal models.
            quant_cfg = self._parse_quant_hf_config(text_config)

        if quant_cfg is not None:
            # Use the community standard 'quant_method'
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Normalize library names
            quant_method = quant_method.replace(
                "compressed_tensors", "compressed-tensors"
            )

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
                # Ensure heavy backends are probed last to avoid unnecessary
                # imports during override detection (e.g., MXFP4 imports Triton)
                "mxfp4",
                "cpu_gptq",
                "cpu_awq",
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
                    quant_cfg, self.quantization
                )
                if quantization_override is not None:
                    # Raise error if the override is not custom (custom would
                    # be in QUANTIZATION_METHODS but not QuantizationMethods)
                    # and hasn't been added to the overrides list.
                    if (
                        name in get_args(me_quant.QuantizationMethods)
                        and name not in overrides
                    ):
                        raise ValueError(
                            f"Quantization method {name} is an override but "
                            "is has not been added to the `overrides` list "
                            "above. This is necessary to ensure that the "
                            "overrides are checked in order of preference."
                        )
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            quant_method = quant_method if quant_method != "" else None
            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization})."
                )

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}."
                )
            from vllm.platforms import current_platform

            current_platform.verify_quantization(self.quantization)

    def _verify_cuda_graph(self) -> None:
        # CUDAGraph capture not supported for encoder-decoder models on ROCm
        unsupported_rocm = self.is_encoder_decoder
        if unsupported_rocm and not self.enforce_eager and current_platform.is_rocm():
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to eager mode.",
                self.hf_config.model_type,
            )
            self.enforce_eager = True

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.46.1) with 8-bit models does not
        yet support CUDA graph.
        # TODO Remove this when bitsandbytes supports.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = (
            getattr(self.hf_config, "quantization_config", None) is not None
        )
        is_8bit = (
            self.hf_config.quantization_config.get("load_in_8bit", False)
            if has_quantization_config
            else False
        )
        if all(
            [
                is_bitsandbytes,
                has_quantization_config,
                is_8bit,
                not self.enforce_eager,
            ]
        ):
            logger.warning(
                "CUDA graph is not supported on BitsAndBytes 8bit yet, "
                "fallback to the eager mode."
            )

            self.enforce_eager = True

    def _verify_with_expert_parallelism(self) -> None:
        num_experts = self.get_num_experts()
        if num_experts < 1:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled."
            )

    def verify_dual_chunk_attention_config(
        self,
        load_config: LoadConfig,
    ) -> None:
        if hasattr(self.hf_config, "dual_chunk_attention_config"):
            # Try loading the sparse attention config
            from vllm.model_executor.model_loader.weight_utils import (
                get_sparse_attention_config,
            )

            sparse_attn_config = get_sparse_attention_config(self, load_config)
            if sparse_attn_config:
                self.hf_config.dual_chunk_attention_config[
                    "sparse_attention_config"
                ] = sparse_attn_config
                if (
                    "sparse_attention_enabled"
                    not in self.hf_config.dual_chunk_attention_config
                ):
                    self.hf_config.dual_chunk_attention_config[
                        "sparse_attention_enabled"
                    ] = True

    def verify_with_parallel_config(
        self,
        parallel_config: ParallelConfig,
    ) -> None:
        total_num_attention_heads = getattr(
            self.hf_text_config, "num_attention_heads", 0
        )
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )

        if parallel_config.enable_expert_parallel:
            self._verify_with_expert_parallelism()

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1 and not self.registry.is_pp_supported_model(
            self.architectures, self
        ):
            raise NotImplementedError(
                "Pipeline parallelism is not supported for this model. "
                "Supported models implement the `SupportsPP` interface."
            )

        decode_context_parallel_size = parallel_config.decode_context_parallel_size
        if decode_context_parallel_size > 1 and not self.use_mla:
            total_num_kv_heads = self.get_total_num_kv_heads()
            assert tensor_parallel_size > total_num_kv_heads, (
                f"tensor parallel size {tensor_parallel_size} must be greater "
                f"than total num kv heads {total_num_kv_heads} when enable "
                f"decode context parallel for GQA/MQA"
            )

            max_dcp_size = tensor_parallel_size // total_num_kv_heads
            assert decode_context_parallel_size <= max_dcp_size, (
                f"decode context parallel size must less than or equal to "
                f"(tensor parallel size {tensor_parallel_size} // total "
                f"num kv heads {total_num_kv_heads}) = {max_dcp_size}, "
                f"but got {decode_context_parallel_size}"
            )

            num_q_per_kv = total_num_attention_heads // total_num_kv_heads
            assert num_q_per_kv % decode_context_parallel_size == 0, (
                f"Total number of q per kv attn heads ({num_q_per_kv})"
                " must be divisible by dcp world size when enable "
                "decode context parallel for GQA "
                f"({parallel_config.decode_context_parallel_size})."
            )

    def get_sliding_window(self) -> int | None:
        """Get the sliding window size from the HF text config if present."""
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "vocab_size", 0)

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "hidden_size", 0)

    def get_inputs_embeds_size(self) -> int:
        # The size of inputs_embeds is usually identical to the size
        # of the hidden states, however there are exceptions, such as
        # embedding models like CLIP and SigLIP
        for target_attr in ("projection_dim", "projection_size"):
            if hasattr(self.hf_text_config, target_attr):
                return getattr(self.hf_text_config, target_attr)

        return self.get_hidden_size()

    @property
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type
                in ("deepseek_v2", "deepseek_v3", "deepseek_v32")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    @cached_property
    def is_mm_prefix_lm(self) -> bool:
        """Whether to use bidirectional attention for mm positions."""
        MM_PREFIX_LM_MODELS = (
            "gemma3",
            # TODO(Isotr0py): Disable paligemma for now before
            # we supports soft cap attention for FlexAttention
            # "paligemma",
        )
        if not hasattr(self.hf_config, "model_type"):
            return False
        return self.hf_config.model_type in MM_PREFIX_LM_MODELS

    def get_head_size(self) -> int:
        # TODO remove hard code
        if self.is_deepseek_mla:
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
            if self.use_mla:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        if hasattr(self.hf_text_config, "model_type") and (
            self.hf_text_config.model_type == "zamba2"
        ):
            return self.hf_text_config.attention_head_dim

        if self.is_attention_free:
            return 0

        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_text_config, "head_dim", None) is not None:
            return self.hf_text_config.head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(self.hf_text_config, "hidden_size_per_head", None) is not None:
            return self.hf_text_config.hidden_size_per_head

        # FIXME(woosuk): This may not be true for all models.
        return (
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads
        )

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_text_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(
                self.hf_config.attn_config,
                "kv_n_heads",
                self.hf_config.num_attention_heads,
            )

        if self.hf_config.model_type == "nemotron-nas":
            for block in self.hf_config.block_configs:
                if not block.attention.no_op:
                    return (
                        self.hf_config.num_attention_heads
                        // block.attention.n_heads_in_group
                    )

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

    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_num_experts(self) -> int:
        """Returns the number of experts in the model."""
        num_expert_names = [
            "num_experts",  # Jamba
            "moe_num_experts",  # Dbrx
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]
        num_experts = getattr_iter(self.hf_text_config, num_expert_names, 0)
        if isinstance(num_experts, list):
            # Ernie VL's remote code uses list[int]...
            # The values are always the same so we just take the first one.
            return num_experts[0]
        # Coerce to 0 if explicitly set to None
        return num_experts or 0

    def get_total_num_hidden_layers(self) -> int:
        if (
            self.hf_text_config.model_type == "deepseek_mtp"
            or self.hf_config.model_type == "mimo_mtp"
            or self.hf_config.model_type == "glm4_moe_mtp"
            or self.hf_config.model_type == "ernie_mtp"
            or self.hf_config.model_type == "qwen3_next_mtp"
            or self.hf_config.model_type == "pangu_ultra_moe_mtp"
        ):
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 0
            )
        elif self.hf_config.model_type == "longcat_flash_mtp":
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_nextn_predict_layers", 1
            )
        else:
            total_num_hidden_layers = getattr(
                self.hf_text_config, "num_hidden_layers", 0
            )
        return total_num_hidden_layers

    def get_layers_start_end_indices(
        self, parallel_config: ParallelConfig
    ) -> tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices

        total_num_hidden_layers = self.get_total_num_hidden_layers()

        # the layout order is: DP x PP x TP
        pp_rank = (
            parallel_config.rank // parallel_config.tensor_parallel_size
        ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    def get_num_layers(self, parallel_config: ParallelConfig) -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(
        self,
        parallel_config: ParallelConfig,
        block_type: LayerBlockType = "attention",
    ) -> int:
        # This function relies on 'layers_block_type' in hf_config,
        # for w/o this attribute, we will need to have workarounds like so
        attn_block_type = block_type == "attention"
        is_transformer = (
            not self.is_hybrid and not self.has_noops and not self.is_attention_free
        )
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
            return sum(not bc.attention.no_op for bc in block_configs[start:end])
        else:
            # Hybrid model Jamba
            layers_block_type_value = getattr(
                self.hf_text_config, "layers_block_type", None
            )
            if layers_block_type_value is not None:
                if hasattr(self.hf_text_config, "model_type") and (
                    self.hf_text_config.model_type == "zamba2"
                ):
                    if attn_block_type:
                        return sum(
                            t == "hybrid" for t in layers_block_type_value[start:end]
                        )
                    else:
                        return self.get_num_layers(parallel_config)
                return sum(t == block_type for t in layers_block_type_value[start:end])

            # Hybrid model Minimax
            attn_type_list = getattr(self.hf_config, "attn_type_list", None)
            if attn_type_list:
                return sum(t == 1 for t in attn_type_list[start:end])

            # Hybrid model Qwen3Next
            layer_types_value = getattr(self.hf_config, "layer_types", None)
            if layer_types_value is not None:
                if block_type == "attention":
                    return sum(
                        t == "full_attention" for t in layer_types_value[start:end]
                    )
                elif block_type == "linear_attention":
                    return sum(
                        t == "linear_attention" for t in layer_types_value[start:end]
                    )
                else:
                    return sum(t == block_type for t in layer_types_value[start:end])

            if (
                layers_block_type_value is None
                and attn_type_list is None
                and layer_types_value is None
            ):
                raise ValueError(
                    "The model is an hybrid without a layers_block_type or an "
                    "attn_type_list, or a layer_types in the hf_config, "
                    f"cannot determine the num of {block_type} layers"
                )

    def get_mamba_chunk_size(self) -> int | None:
        """
        Returns the mamba chunk size if it exists
        """
        # used by e.g. Bamba, FalconH1, Granite, PLaMo2
        chunk_size = getattr(self.hf_text_config, "mamba_chunk_size", None)
        if chunk_size is None:
            # used by e.g. Mamba2, NemotronH, Zamba
            chunk_size = getattr(self.hf_text_config, "chunk_size", None)

        # Since Mamba1 does not have a chunk notion
        # we use a default chunk size of 1024.
        if chunk_size is None:
            chunk_size = 2048

        return chunk_size

    def get_multimodal_config(self) -> MultiModalConfig:
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
                config_format=self.config_format,
            )
        else:
            config = try_get_generation_config(
                self.generation_config,
                trust_remote_code=self.trust_remote_code,
                config_format=self.config_format,
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
                p: config.get(p) for p in available_params if config.get(p) is not None
            }
            # Huggingface definition of max_new_tokens is equivalent
            # to vLLM's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens"
                )
        else:
            diff_sampling_param = {}

        if diff_sampling_param:
            logger.warning_once(
                "Default sampling parameters have been overridden by the "
                "model's Hugging Face generation config recommended from the "
                "model creator. If this is not intended, please relaunch "
                "vLLM instance with `--generation-config vllm`."
            )
        return diff_sampling_param

    @property
    def is_encoder_decoder(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        return is_encoder_decoder(self.hf_config)

    @property
    def uses_alibi(self) -> bool:
        cfg = self.hf_text_config

        return (
            getattr(cfg, "alibi", False)  # Falcon
            or "BloomForCausalLM" in self.architectures  # Bloom
            or getattr(cfg, "position_encoding_type", "") == "alibi"  # codellm_1b_alibi
            or (
                hasattr(cfg, "attn_config")  # MPT
                and (
                    (
                        isinstance(cfg.attn_config, dict)
                        and cfg.attn_config.get("alibi", False)
                    )
                    or (
                        not isinstance(cfg.attn_config, dict)
                        and getattr(cfg.attn_config, "alibi", False)
                    )
                )
            )
        )

    @property
    def uses_mrope(self) -> bool:
        return uses_mrope(self.hf_config)

    @property
    def uses_xdrope_dim(self) -> int:
        return uses_xdrope_dim(self.hf_config)

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None

    @property
    def is_multimodal_raw_input_only_model(self) -> bool:
        return self._model_info.supports_multimodal_raw_input_only

    @property
    def is_cross_encoder(self) -> bool:
        return (
            self._model_info.supports_cross_encoding or self.convert_type == "classify"
        )

    @property
    def is_pp_supported(self) -> bool:
        return self._model_info.supports_pp

    @property
    def is_attention_free(self) -> bool:
        return self._model_info.is_attention_free

    @property
    def is_hybrid(self) -> bool:
        # Handle granite-4.0-micro case which uses hybrid config but does not
        # actually contain any non-attention layers.
        layer_types = getattr(self.hf_config, "layer_types", None)
        if layer_types is not None and all(
            layer == "attention" for layer in layer_types
        ):
            return False
        return self._model_info.is_hybrid

    @property
    def has_noops(self) -> bool:
        return self._model_info.has_noops

    @property
    def has_inner_state(self):
        return self._model_info.has_inner_state

    @property
    def supports_mamba_prefix_caching(self) -> bool:
        return self._model_info.supports_mamba_prefix_caching

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE

    @property
    def is_matryoshka(self) -> bool:
        return bool(getattr(self.hf_config, "matryoshka_dimensions", None)) or getattr(
            self.hf_config, "is_matryoshka", False
        )

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

        head_dtype = _get_head_dtype(
            config=self.hf_config, dtype=self.dtype, runner_type=self.runner_type
        )

        if self.runner_type != "pooling" and head_dtype != self.dtype:
            logger.warning_once(
                "`head_dtype` currently only supports pooling models."
                "fallback to model dtype [%s].",
                self.dtype,
            )
            return self.dtype

        if head_dtype not in current_platform.supported_dtypes:
            logger.warning_once(
                "The current platform does not support [%s] head dtype, "
                "fallback to model dtype [%s].",
                head_dtype,
                self.dtype,
            )
            return self.dtype

        logger.debug_once("head dtype: %s", head_dtype)
        return head_dtype

    @property
    def embedding_size(self):
        dense_modules = try_get_dense_modules(self.model, revision=self.revision)
        if dense_modules is not None:
            return dense_modules[-1]["out_features"]
        return self.get_hidden_size()

    def get_and_verify_max_len(self, max_model_len: int):
        # Consider max_model_len in tokenizer_config only when
        # pooling models use absolute position_embedding.
        tokenizer_config = None
        if (
            self.runner_type == "pooling"
            and getattr(self.hf_config, "position_embedding_type", "") == "absolute"
        ):
            tokenizer_config = try_get_tokenizer_config(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                revision=self.tokenizer_revision,
            )
        max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            tokenizer_config=tokenizer_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window=self.get_sliding_window(),
            spec_target_max_model_len=self.spec_target_max_model_len,
            encoder_config=self.encoder_config,
        )
        logger.info("Using max model len %s", max_model_len)
        return max_model_len

    @property
    def attn_type(self) -> AttnTypeStr:
        if self.pooler_config is not None:
            pooling_type = self._model_info.default_pooling_type.lower()
            if pooling_type == "cls":
                return "encoder_only"
            else:
                is_causal = getattr(self.hf_config, "is_causal", True)
                return "encoder_only" if not is_causal else self._model_info.attn_type
        elif self.is_hybrid:
            return "hybrid"
        elif self.is_attention_free:
            return "attention_free"
        elif self.is_encoder_decoder:
            return "encoder_decoder"
        else:
            return "decoder"

    @property
    def is_chunked_prefill_supported(self) -> bool:
        attn_type = self.attn_type
        if self.pooler_config is not None:
            # for pooling models
            if attn_type == "encoder_only":
                logger.debug(
                    "Pooling models with bidirectional attn does not support "
                    "chunked prefill."
                )
                return False
            elif attn_type == "decoder":
                pooling_type = self.pooler_config.pooling_type.lower()
                if pooling_type in ["mean", "step", "cls"]:
                    logger.debug(
                        "Pooling models with %s pooling does not "
                        "support chunked prefill.",
                        pooling_type,
                    )
                    return False
                elif pooling_type in ["all", "last"]:
                    logger.debug(
                        "Pooling models with causal attn and %s pooling support "
                        "chunked prefill.",
                        pooling_type,
                    )
                    return True
                else:
                    raise ValueError(f"{pooling_type=} not supported.")
            # vllm currently does not have pooling models using hybrid,
            # attention_free or encoder_decoder attn types.
            return attn_type != "encoder_decoder"
        else:
            if attn_type == "encoder_decoder":
                logger.debug("Encoder decoder models does not support chunked prefill.")
                return False
            logger.debug("Generative models support chunked prefill.")
            return True

    @property
    def is_prefix_caching_supported(self) -> bool:
        attn_type = self.attn_type
        if self.pooler_config is not None:
            # for pooling models
            if attn_type == "encoder_only":
                logger.debug(
                    "Pooling models with bidirectional attn does not "
                    "support prefix caching."
                )
                return False
            elif attn_type == "decoder":
                pooling_type = self.pooler_config.pooling_type.lower()
                if pooling_type in ["mean", "step", "cls"]:
                    logger.debug(
                        "Pooling models with %s pooling does not "
                        "support prefix caching.",
                        pooling_type,
                    )
                    return False
                elif pooling_type in ["all", "last"]:
                    logger.debug(
                        "Pooling models with causal attn and %s pooling support "
                        "prefix caching.",
                        pooling_type,
                    )
                    return True
                else:
                    raise ValueError(f"{pooling_type=} not supported.")
            # vllm currently does not have pooling models using hybrid,
            # attention_free or encoder_decoder attn types.
            return False
        else:
            if attn_type == "hybrid":
                logger.debug(
                    "Hybrid models does not support prefix caching since the feature "
                    "is still experimental."
                )
                return False
            elif attn_type == "attention_free":
                logger.debug(
                    "Attention free models does not support prefix caching since the "
                    "feature is still experimental."
                )
                return False
            elif attn_type == "encoder_decoder":
                logger.debug("Encoder decoder models does not support prefix caching.")
                return False
            else:  # attn_type == "decoder"
                logger.debug("Generative models support prefix caching.")
                return True

    def is_model_moe(
        self,
    ) -> bool:
        return self.get_num_experts() > 1

    def is_quantized(self) -> bool:
        return getattr(self.hf_config, "quantization_config", None) is not None


def get_served_model_name(model: str, served_model_name: str | list[str] | None):
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
    runner_type: RunnerType | None = None,
    convert_type: ConvertType | None = None,
) -> tuple[str, tuple[RunnerType, ConvertType]] | None:
    for suffix, (
        default_runner_type,
        default_convert_type,
    ) in iter_architecture_defaults():
        if (
            (runner_type is None or runner_type == default_runner_type)
            and (convert_type is None or convert_type == default_convert_type)
            and architecture.endswith(suffix)
        ):
            return suffix, (default_runner_type, default_convert_type)

    return None


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
    "gemma3_text": "Numerical instability. Please use bfloat16 or float32 instead.",
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
        raise ValueError(
            f"The model type {model_type!r} does not support float16. Reason: {reason}"
        )

    return True


def _find_dtype(
    model_id: str,
    config: PretrainedConfig,
    *,
    revision: str | None,
):
    # NOTE: getattr(config, "dtype", torch.float32) is not correct
    # because config.dtype can be None.
    config_dtype = getattr(config, "dtype", None)

    # Fallbacks for multi-modal models if the root config
    # does not define dtype
    if config_dtype is None:
        config_dtype = getattr(config.get_text_config(), "dtype", None)
    if config_dtype is None and hasattr(config, "vision_config"):
        config_dtype = getattr(config.vision_config, "dtype", None)
    if config_dtype is None and hasattr(config, "encoder_config"):
        config_dtype = getattr(config.encoder_config, "dtype", None)

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
        dtype
        for dtype in current_platform.supported_dtypes
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
        "Your device %s doesn't support %s. Falling back to %s for compatibility.",
        device_str,
        config_dtype,
        preferred_dtype,
    )

    return preferred_dtype


def _get_and_verify_dtype(
    model_id: str,
    config: PretrainedConfig,
    dtype: str | torch.dtype,
    *,
    is_pooling_model: bool,
    revision: str | None = None,
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


def _get_head_dtype(
    config: PretrainedConfig, dtype: torch.dtype, runner_type: str
) -> torch.dtype:
    head_dtype: str | torch.dtype | None = getattr(config, "head_dtype", None)

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
    tokenizer_config: dict | None,
    max_model_len: int | None,
    disable_sliding_window: bool,
    sliding_window: int | None,
    spec_target_max_model_len: int | None = None,
    encoder_config: Any | None = None,
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
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)
    # For Command-R / Cohere, Cohere2 / Aya Vision models
    if tmp_max_len := getattr(hf_config, "model_max_length", None):
        max_len_key = "model_max_length"
        derived_max_model_len = tmp_max_len

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if (
        disable_sliding_window
        and sliding_window is not None
        and sliding_window < derived_max_model_len
    ):
        max_len_key = "sliding_window"
        derived_max_model_len = sliding_window

    # Consider model_max_length in tokenizer_config
    if tokenizer_config:
        tokenizer_model_max_length = tokenizer_config.get(
            "model_max_length", derived_max_model_len
        )
        derived_max_model_len = min(derived_max_model_len, tokenizer_model_max_length)

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
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    # In Transformers v5 rope_parameters could be TypedDict or dict[str, TypedDict].
    # To simplify the verification, we convert it to dict[str, TypedDict].
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters and not set(rope_parameters.keys()).issubset(
        ALLOWED_LAYER_TYPES
    ):
        rope_parameters = {"": rope_parameters}

    # NOTE(woosuk): Gemma3's max_model_len (128K) is already scaled by RoPE
    # scaling, so we skip applying the scaling factor again.
    if rope_parameters is not None and "gemma3" not in hf_config.model_type:
        scaling_factor = 1.0
        for rp in rope_parameters.values():
            # No need to consider "type" key because of patch_rope_parameters when
            # loading HF config
            rope_type = rp["rope_type"]

            if rope_type not in ("su", "longrope", "llama3"):
                # NOTE: rope_type == "default" does not define factor https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
                # NOTE: This assumes all layer types have the same scaling factor.
                scaling_factor = rp.get("factor", scaling_factor)

                if rope_type == "yarn":
                    derived_max_model_len = rp["original_max_position_embeddings"]
        # Do this outside loop since all layer types should have the same scaling
        derived_max_model_len *= scaling_factor

    if encoder_config and "max_seq_length" in encoder_config:
        derived_max_model_len = encoder_config["max_seq_length"]

    # If the user didn't specify `max_model_len`, then use that derived from
    # the model config as a default value.
    if max_model_len is None:
        # For LongRoPE, default to original_max_position_embeddings to avoid
        # performance degradation for shorter sequences
        if rope_parameters is not None and any(
            rp["rope_type"] == "longrope" for rp in rope_parameters.values()
        ):
            max_model_len = int(
                getattr(
                    hf_config, "original_max_position_embeddings", derived_max_model_len
                )
            )
        else:
            max_model_len = int(derived_max_model_len)
        max_model_len = current_platform.check_max_model_len(max_model_len)

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is None or max_model_len > model_max_length:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json)."
            )
            warning = (
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme "
                "caution. If the model uses relative position encoding (RoPE), "
                "positions exceeding derived_max_model_len lead to nan. If the "
                "model uses absolute position encoding, positions exceeding "
                "derived_max_model_len will cause a CUDA array out-of-bounds "
                "error."
            )
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning_once("%s %s", msg, warning)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    f"the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. {warning}"
                )
    return int(max_model_len)
