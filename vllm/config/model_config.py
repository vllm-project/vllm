# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import hashlib
import json
import warnings
from dataclasses import field
from importlib.util import find_spec
from typing import (TYPE_CHECKING, Any, Callable, Literal, Optional, TypeVar,
                    Union, cast, get_args)

import regex as re
import torch
from pydantic import (ConfigDict, SkipValidation, field_validator,
                      model_validator)
from pydantic.dataclasses import dataclass
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (QUANTIZATION_METHODS,
                                                     QuantizationMethods,
                                                     get_quantization_config)
from vllm.model_executor.models import ModelRegistry
from vllm.platforms import current_platform
from vllm.transformers_utils.config import (
    ConfigFormat, get_config, get_hf_image_processor_config,
    get_hf_text_config, get_pooling_config,
    get_sentence_transformer_tokenizer_config, is_encoder_decoder,
    try_get_generation_config, uses_mrope)
from vllm.transformers_utils.s3_utils import S3Model
from vllm.transformers_utils.utils import is_s3, maybe_model_redirect
from vllm.utils import LayerBlockType

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from vllm.config.load_config import LoadConfig
    from vllm.config.multimodal_config import MultiModalConfig
    from vllm.config.parallel_config import ParallelConfig
    from vllm.config.pooler_config import PoolerConfig
    ConfigType = type[DataclassInstance]
else:
    ConfigType = type
    LoadConfig = type
    MultiModalConfig = type
    ParallelConfig = type
    PoolerConfig = type

logger = init_logger(__name__)

ConfigT = TypeVar("ConfigT", bound=ConfigType)

TaskOption = Literal["auto", "generate", "embedding", "embed", "classify",
                     "score", "reward", "transcription"]

_ResolvedTask = Literal["generate", "embed", "classify", "score", "reward",
                        "draft", "transcription"]

RunnerType = Literal["generate", "pooling", "draft", "transcription"]

_RUNNER_TASKS: dict[RunnerType, list[_ResolvedTask]] = {
    "generate": ["generate"],
    "pooling": ["embed", "classify", "score", "reward"],
    "draft": ["draft"],
    "transcription": ["transcription"],
}

_TASK_RUNNER: dict[_ResolvedTask, RunnerType] = {
    task: runner
    for runner, tasks in _RUNNER_TASKS.items()
    for task in tasks
}

HfOverrides = Union[dict[str, Any], Callable[[PretrainedConfig],
                                             PretrainedConfig]]

TokenizerMode = Literal["auto", "slow", "mistral", "custom"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]


class ModelImpl(str, enum.Enum):
    AUTO = "auto"
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


def contains_object_print(text):
    """
    Check if the text looks like a printed Python object, e.g.
    contains any substring matching the pattern: "at 0xFFFFFFF>"
    We match against 0x followed by 2-16 hex chars (there's
    a max of 16 on a 64 bit system).

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

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

_ROCM_NOT_SUPPORTED_DTYPE: list[str] = []  #


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)

    # Fallbacks for multi-modal models if the root config
    # does not define torch_dtype
    if config_dtype is None:
        config_dtype = getattr(config.get_text_config(), "torch_dtype", None)
    if config_dtype is None and hasattr(config, "vision_config"):
        config_dtype = getattr(config.vision_config, "torch_dtype", None)

    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            # Set default dtype from model config
            if config_dtype == torch.float32:
                # Following common practice, we use float16 for float32 models
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype

            if config.model_type == "plamo2":
                logger.warning(
                    "For PLaMo2, we cast models to bfloat16 instead of using "
                    "float16 by default. This is because float16 does not work."
                )
                torch_dtype = torch.bfloat16

            # Deal with torch dtype fallback for device compatibility.
            from vllm.platforms import current_platform
            if torch_dtype not in current_platform.supported_dtypes:
                device_name = current_platform.get_device_name()

                if ((capability := current_platform.get_device_capability())
                        is None):
                    compute_str = ""
                else:
                    version_str = capability.as_version_str()
                    compute_str = f" (with compute capability {version_str})"
                fallback_dtype = current_platform.supported_dtypes[0]
                logger.warning(
                    "Your %s device%s doesn't support %s. " \
                    "Falling back to %s for compatibility.",
                    device_name, compute_str, torch_dtype, fallback_dtype
                    )
                torch_dtype = fallback_dtype

            if current_platform.is_hpu() and torch_dtype == torch.float16:
                logger.warning(
                    "For HPU, we cast models to bfloat16 instead of "
                    "using float16 by default. Please specify `dtype` if you "
                    "want to use float16.")
                torch_dtype = torch.bfloat16
        elif dtype == "float16" and config.model_type == "plamo2":
            logger.warning(
                "For PLaMo2, using float16 is unstable and might cause "
                "unexpected behavior. Please use bfloat16 or float32 instead.")
            torch_dtype = torch.float16
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
    disable_sliding_window: bool,
    sliding_window_len: Optional[Union[int, list[Optional[int]]]],
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
    # Choose the smallest "max_length" from the possible keys.
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
    if disable_sliding_window and sliding_window_len is not None:

        sliding_window_len_min = get_min_sliding_window(sliding_window_len)
        max_len_key = "sliding_window" \
            if sliding_window_len_min < derived_max_model_len else max_len_key
        derived_max_model_len = min(derived_max_model_len,
                                    sliding_window_len_min)

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
                f"{model_max_length} in model's config.json). This may lead "
                "to incorrect model outputs or CUDA errors.")
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning(
                    "%s Make sure the value is correct and within the "
                    "model context size.", msg)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    "the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1")
    return int(max_model_len)


def get_min_sliding_window(
        sliding_window: Union[int, list[Optional[int]]]) -> int:
    if isinstance(sliding_window, list):
        return min(s for s in sliding_window if s is not None)

    return sliding_window


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


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """Configuration for the model."""

    model: str = "facebook/opt-125m"
    """Name or path of the Hugging Face model to use. It is also used as the
    content for `model_name` tag in metrics output when `served_model_name` is
    not specified."""
    task: Literal[TaskOption, Literal["draft"]] = "auto"
    """The task to use the model for. Each vLLM instance only supports one
    task, even if the same model can be used for multiple tasks. When the model
    only supports one task, "auto" can be used to select it; otherwise, you
    must specify explicitly which task to use."""
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
    OpenAI Chat Completions API."""
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
    limit_mm_per_prompt: dict[str, int] = field(default_factory=dict)
    """Maximum number of data items per modality per prompt. Only applicable
    for multimodal models."""
    use_async_output_proc: bool = True
    """Whether to use async output processor."""
    config_format: Union[str, ConfigFormat] = ConfigFormat.AUTO.value
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
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Arguments to be forwarded to the model's processor for multi-modal data,
    e.g., image processor. Overrides for the multi-modal processor obtained
    from `AutoProcessor.from_pretrained`. The available overrides depend on the
    model that is being run. For example, for Phi-3-Vision: `{"num_crops": 4}`.
    """
    disable_mm_preprocessor_cache: bool = False
    """If `True`, disable caching of the multi-modal preprocessor/mapper (not
    recommended)."""
    override_neuron_config: dict[str, Any] = field(default_factory=dict)
    """Initialize non-default neuron config or override default neuron config
    that are specific to Neuron devices, this argument will be used to
    configure the neuron config that can not be gathered from the vllm
    arguments. e.g. `{"cast_logits_dtype": "bfloat16"}`."""
    pooler_config: Optional[PoolerConfig] = field(init=False)
    """Pooler config which controls the behaviour of output pooling in pooling
    models."""
    override_pooler_config: Optional[Union[dict, PoolerConfig]] = None
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
    - "transformers" will use the Transformers model implementation."""

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

    def __post_init__(self) -> None:
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

        self.maybe_pull_model_tokenizer_for_s3(self.model, self.tokenizer)

        if (backend := envs.VLLM_ATTENTION_BACKEND
            ) and backend == "FLASHINFER" and find_spec("flashinfer") is None:
            raise ValueError(
                "VLLM_ATTENTION_BACKEND is set to FLASHINFER, but flashinfer "
                "module was not found. See "
                "https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile "  # noqa: E501
                "for instructions on how to install it.")

        from vllm.platforms import current_platform

        if (self.enable_sleep_mode
                and not current_platform.is_sleep_mode_available()):
            raise ValueError(
                "Sleep mode is not supported on current platform.")

        if isinstance(self.config_format, str):
            self.config_format = ConfigFormat(self.config_format)

        hf_config = get_config(self.hf_config_path or self.model,
                               self.trust_remote_code, self.revision,
                               self.code_revision, self.config_format)

        if hf_overrides_kw:
            logger.info("Overriding HF config with %s", hf_overrides_kw)
            hf_config.update(hf_overrides_kw)
        if hf_overrides_fn:
            logger.info("Overriding HF config with %s", hf_overrides_fn)
            hf_config = hf_overrides_fn(hf_config)

        self.hf_config = hf_config

        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.attention_chunk_size = getattr(self.hf_text_config,
                                            "attention_chunk_size", None)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, hf_token=self.hf_token, revision=self.revision)
        self.dtype = _get_and_verify_dtype(self.hf_config, self.dtype)

        # Workaround for Gemma 2 which uses interleaved sliding window
        # attention, but it's not specified in its config. TODO: remove this
        # when Gemma 2 is fixed in Transformers.
        if self.hf_text_config.model_type == "gemma2":
            self.hf_text_config.sliding_window_pattern = 2

        sliding_window = getattr(self.hf_text_config, "sliding_window", None)
        sliding_window_pattern = getattr(self.hf_text_config,
                                         "sliding_window_pattern", None)
        has_interleaved_attention = sliding_window_pattern is not None or (
            isinstance(sliding_window, list))

        if not self.disable_sliding_window and has_interleaved_attention:
            if (backend :=
                    envs.VLLM_ATTENTION_BACKEND) in ("XFORMERS", "FLASHINFER"):
                sliding_window_len_min = get_min_sliding_window(
                    self.hf_text_config.sliding_window)

                logger.warning_once(
                    "%s has interleaved attention, which is currently not supported by the %s backend. Disabling sliding window and capping the max length to the sliding window size (%d).",  # noqa: E501
                    self.hf_text_config.model_type,
                    backend,
                    sliding_window_len_min,
                )
                self.disable_sliding_window = True
            else:
                # for a model with interleaved attention,
                # the scheduler and the model treat it as full attention
                # (i.e., not dropping any tokens outside the window).
                # only the attention layer itself is aware of the sliding
                # window, and use the window size to compute the attention.
                self.hf_text_config.interleaved_sliding_window = sliding_window

                if hasattr(self.hf_text_config, "sliding_window"):
                    delattr(self.hf_text_config, "sliding_window")

                sliding_window = None

        self.original_max_model_len = self.max_model_len
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
        self.served_model_name = get_served_model_name(self.model,
                                                       self.served_model_name)
        self.multimodal_config = self._init_multimodal_config()
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        self.is_attention_free = self._init_attention_free()
        self.is_hybrid = self._init_is_hybrid()
        self.has_noops = self._init_has_noops()
        self.has_inner_state = self._init_has_inner_state()

        if (not current_platform.is_neuron() and self.override_neuron_config):
            raise ValueError(
                "`override_neuron_config` is only supported on Neuron.")

        supported_tasks, task = self._resolve_task(self.task)
        self.supported_tasks = supported_tasks
        self.task = task
        if self.task in ("draft", "generate"):
            self.truncation_side = "left"
        else:
            self.truncation_side = "right"

        self.pooler_config = self._init_pooler_config()

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
    def validate_model_config_after(self: ModelConfig) -> ModelConfig:
        if not isinstance(self.tokenizer, str):
            raise ValueError("tokenizer must be a string after __post_init__.")
        if not isinstance(self.max_model_len, int):
            raise ValueError(
                "max_model_len must be an integer after __post_init__.")
        return self

    @property
    def registry(self):
        return ModelRegistry

    @property
    def architectures(self) -> list[str]:
        return getattr(self.hf_config, "architectures", [])

    def maybe_pull_model_tokenizer_for_s3(self, model: str,
                                          tokenizer: str) -> None:
        """Pull model/tokenizer from S3 to temporary directory when needed.
        
        Args:
            model: Model name or path
            tokenizer: Tokenizer name or path
        """
        if not (is_s3(model) or is_s3(tokenizer)):
            return

        if is_s3(model):
            s3_model = S3Model()
            s3_model.pull_files(model,
                                allow_pattern=["*.model", "*.py", "*.json"])
            self.model_weights = model
            self.model = s3_model.dir

            # If tokenizer is same as model, download to same directory
            if model == tokenizer:
                s3_model.pull_files(
                    model, ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
                self.tokenizer = s3_model.dir
                return

        # Only download tokenizer if needed and not already handled
        if is_s3(tokenizer):
            s3_tokenizer = S3Model()
            s3_tokenizer.pull_files(
                model, ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
            self.tokenizer = s3_tokenizer.dir

    def _init_multimodal_config(self) -> Optional[MultiModalConfig]:
        from vllm.config.multimodal_config import MultiModalConfig
        if self.registry.is_multimodal_model(self.architectures):
            return MultiModalConfig(
                limit_per_prompt=self.limit_mm_per_prompt,
                mm_processor_kwargs=self.mm_processor_kwargs,
                disable_mm_preprocessor_cache=self.
                disable_mm_preprocessor_cache)

        if self.limit_mm_per_prompt:
            raise ValueError("`limit_mm_per_prompt` is only supported for "
                             "multimodal models.")
        if self.mm_processor_kwargs:
            raise ValueError("`mm_processor_kwargs` is only supported for "
                             "multimodal models.")
        if self.disable_mm_preprocessor_cache:
            raise ValueError("`disable_mm_preprocessor_cache` is only "
                             "supported for multimodal models.")

        return None

    def _get_encoder_config(self):
        return get_sentence_transformer_tokenizer_config(
            self.model, self.revision)

    def _init_pooler_config(self) -> Optional[PoolerConfig]:
        from vllm.config.pooler_config import PoolerConfig
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

            if self.is_matryoshka:
                if pooler_config.normalize is None:
                    pooler_config.normalize = True
                elif not pooler_config.normalize:
                    raise ValueError(
                        "`normalize` must be enabled (set to True) "
                        "for models that are compatible with "
                        "Matryoshka Representation.")

            return pooler_config

        return None

    def _init_attention_free(self) -> bool:
        return self.registry.is_attention_free_model(self.architectures)

    def _init_is_hybrid(self) -> bool:
        return self.registry.is_hybrid_model(self.architectures)

    def _init_has_noops(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return self.registry.is_noops_model(architectures)

    def _init_has_inner_state(self) -> bool:
        return self.registry.model_has_inner_state(self.architectures)

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = cast(TokenizerMode, self.tokenizer_mode.lower())
        if tokenizer_mode not in get_args(TokenizerMode):
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                f"one of {get_args(TokenizerMode)}.")
        self.tokenizer_mode = tokenizer_mode

    def _get_preferred_task(
        self,
        architectures: list[str],
        supported_tasks: set[_ResolvedTask],
    ) -> Optional[_ResolvedTask]:
        model_id = self.model
        if get_pooling_config(model_id, self.revision):
            return "embed"
        if self.registry.is_cross_encoder_model(architectures):
            return "score"
        if self.registry.is_transcription_model(architectures):
            return "transcription"

        suffix_to_preferred_task: list[tuple[str, _ResolvedTask]] = [
            # Other models follow this pattern
            ("ForCausalLM", "generate"),
            ("ForConditionalGeneration", "generate"),
            ("ForSequenceClassification", "classify"),
            ("ChatModel", "generate"),
            ("LMHeadModel", "generate"),
            ("EmbeddingModel", "embed"),
            ("RewardModel", "reward"),
        ]
        _, arch = self.registry.inspect_model_cls(architectures)

        for suffix, pref_task in suffix_to_preferred_task:
            if arch.endswith(suffix) and pref_task in supported_tasks:
                return pref_task

        return None

    def _resolve_task(
        self,
        task_option: Literal[TaskOption, Literal["draft"]],
    ) -> tuple[set[_ResolvedTask], _ResolvedTask]:
        if task_option == "draft":
            return {"draft"}, "draft"

        registry = self.registry
        architectures = self.architectures

        runner_support: dict[RunnerType, bool] = {
            # NOTE: Listed from highest to lowest priority,
            # in case the model supports multiple of them
            "transcription": registry.is_transcription_model(architectures),
            "generate": registry.is_text_generation_model(architectures),
            "pooling": registry.is_pooling_model(architectures),
        }
        supported_runner_types_lst: list[RunnerType] = [
            runner_type
            for runner_type, is_supported in runner_support.items()
            if is_supported
        ]

        supported_tasks_lst: list[_ResolvedTask] = [
            task for runner_type in supported_runner_types_lst
            for task in _RUNNER_TASKS[runner_type]
        ]
        supported_tasks = set(supported_tasks_lst)

        if task_option == "auto":
            selected_task = next(iter(supported_tasks_lst))

            if len(supported_tasks_lst) > 1:
                preferred_task = self._get_preferred_task(
                    architectures, supported_tasks)
                if preferred_task is not None:
                    selected_task = preferred_task

                logger.info(
                    "This model supports multiple tasks: %s. "
                    "Defaulting to '%s'.", supported_tasks, selected_task)
        else:
            # Aliases
            if task_option == "embedding":
                msg = ("The 'embedding' task has been renamed to "
                       "'embed', please use the new name. The old name "
                       "will be removed in v1.0.")
                warnings.warn(msg, DeprecationWarning, stacklevel=2)

                task_option = "embed"

            if task_option not in supported_tasks:
                msg = (
                    f"This model does not support the '{task_option}' task. "
                    f"Supported tasks: {supported_tasks}")
                raise ValueError(msg)

            selected_task = task_option

        return supported_tasks, selected_task

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed-tensors", "experts_int8",
            "quark", "modelopt_fp4", "bitblas", "gptq_bitblas"
        ]
        if self.quantization is not None:
            self.quantization = cast(QuantizationMethods, self.quantization)

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()
            quant_method = quant_method.replace("compressed_tensors",
                                                "compressed-tensors")
            quant_cfg["quant_method"] = quant_method

            # Quantization methods which are overrides (i.e. they have a
            # `override_quantization_method` method) must be checked in order
            # of preference (this is particularly important for GPTQ).
            overrides = [
                "marlin",
                "bitblas",
                "gptq_marlin_24",
                "gptq_marlin",
                "gptq_bitblas",
                "awq_marlin",
                "ipex",
                "moe_wna16",
            ]
            quantization_methods = [
                q for q in supported_quantization if q not in overrides
            ]
            # Any custom overrides will be in quantization_methods so we place
            # them at the start of the list so custom overrides have preference
            # over the built in ones.
            quantization_methods = quantization_methods + overrides

            # Detect which checkpoint is it
            for name in quantization_methods:
                method = get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override is not None:
                    # Raise error if the override is not custom (custom would
                    # be in QUANTIZATION_METHODS but not QuantizationMethods)
                    # and hasn't been added to the overrides list.
                    if (name in get_args(QuantizationMethods)
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
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)
        # CUDAGraph capture not supported for enc-dec models and mllama on ROCm
        ROCM_UNSUPPORTED_MODELS = ['mllama']
        unsupported_rocm = (self.hf_config.model_type
                            in ROCM_UNSUPPORTED_MODELS
                            or self.is_encoder_decoder)

        if (unsupported_rocm and not self.enforce_eager
                and current_platform.is_rocm()):
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to eager mode.", self.hf_config.model_type)
            self.enforce_eager = True

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.45.3) with 8-bit models does not
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
        load_config: LoadConfig,
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
        parallel_config: ParallelConfig,
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
            if not self.registry.is_pp_supported_model(self.architectures):
                raise NotImplementedError(
                    "Pipeline parallelism is not supported for this model. "
                    "Supported models implement the `SupportsPP` interface.")

            if self.use_async_output_proc:
                self.use_async_output_proc = False

    def get_hf_config_sliding_window(
            self) -> Union[Optional[int], list[Optional[int]]]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (hasattr(self.hf_text_config, "use_sliding_window")
                and not self.hf_text_config.use_sliding_window):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_sliding_window(self) -> Optional[Union[int, list[Optional[int]]]]:
        """Get the sliding window size, or None if disabled.
        """
        # If user disables sliding window, return None.
        if self.disable_sliding_window:
            return None
        # Otherwise get the value from the hf config.
        return self.get_hf_config_sliding_window()

    def get_vocab_size(self) -> int:
        return self.hf_text_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_text_config.hidden_size

    @property
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in \
            ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp'):
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
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_layers_start_end_indices(
            self, parallel_config: ParallelConfig) -> tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices
        if (self.hf_text_config.model_type == "deepseek_mtp"
                or self.hf_config.model_type == "mimo_mtp"):
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

    def get_num_layers(self, parallel_config: ParallelConfig) -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(
        self,
        parallel_config: ParallelConfig,
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
            layers_block_type_value = getattr(self.hf_config,
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

            if layers_block_type_value is None and attn_type_list is None:
                raise ValueError(
                    "The model is an hybrid without a"
                    "layers_block_type or an attn_type_list in the hf_config,"
                    "cannot determine the num of "
                    f"{block_type.value} layers")

            return sum(t == 1 for t in attn_type_list[start:end])

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
        if self.generation_config in ("auto", "vllm"):
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
        This method returns a dictionary containing the parameters
        that differ from the default sampling parameters. If
        `generation_config` is `"vllm"`, an empty dictionary is returned.

        Returns:
            dict[str, Any]: A dictionary with the differing sampling
            parameters, if `generation_config` is `"vllm"` an empty dictionary.
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
    def is_cross_encoder(self) -> bool:
        return self.registry.is_cross_encoder_model(self.architectures)

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE

    @property
    def supported_runner_types(self) -> set[RunnerType]:
        return {_TASK_RUNNER[task] for task in self.supported_tasks}

    @property
    def runner_type(self) -> RunnerType:
        return _TASK_RUNNER[cast(_ResolvedTask, self.task)]

    @property
    def is_v1_compatible(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.is_v1_compatible(architectures)

    @property
    def is_matryoshka(self) -> bool:
        return (hasattr(self.hf_config, "matryoshka_dimensions")
                or getattr(self.hf_config, "is_matryoshka", False))

    @property
    def matryoshka_dimensions(self):
        return getattr(self.hf_config, "matryoshka_dimensions", None)

    def get_and_verify_max_len(self, max_model_len: int):
        max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
            spec_target_max_model_len=self.spec_target_max_model_len,
            encoder_config=self.encoder_config)
        return max_model_len
