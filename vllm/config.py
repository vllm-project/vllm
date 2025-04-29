# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import enum
import hashlib
import inspect
import json
import re
import sys
import textwrap
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import (MISSING, dataclass, field, fields, is_dataclass,
                         replace)
from importlib.util import find_spec
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Final, Literal,
                    Optional, Protocol, TypeVar, Union, get_args)

import torch
from pydantic import BaseModel, Field, PrivateAttr
from torch.distributed import ProcessGroup, ReduceOp
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (QUANTIZATION_METHODS,
                                                     QuantizationMethods,
                                                     get_quantization_config)
from vllm.model_executor.models import ModelRegistry
from vllm.platforms import CpuArchEnum, current_platform
from vllm.sampling_params import GuidedDecodingParams
from vllm.tracing import is_otel_available, otel_import_error_traceback
from vllm.transformers_utils.config import (
    ConfigFormat, get_config, get_hf_image_processor_config,
    get_hf_text_config, get_pooling_config,
    get_sentence_transformer_tokenizer_config, is_encoder_decoder,
    try_get_generation_config, uses_mrope)
from vllm.transformers_utils.s3_utils import S3Model
from vllm.transformers_utils.utils import is_s3, maybe_model_redirect
from vllm.utils import (GiB_bytes, LayerBlockType, cuda_device_count_stateless,
                        get_cpu_memory, get_open_port, is_torch_equal_or_newer,
                        random_uuid, resolve_obj_by_qualname)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    from ray.util.placement_group import PlacementGroup

    from vllm.executor.executor_base import ExecutorBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig)
    from vllm.model_executor.model_loader.loader import BaseModelLoader

    ConfigType = type[DataclassInstance]
else:
    QuantizationConfig = None
    ConfigType = type

logger = init_logger(__name__)

ConfigT = TypeVar("ConfigT", bound=ConfigType)

# This value is chosen to have a balance between ITL and TTFT. Note it is
# not optimized for throughput.
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
_POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
_MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120

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


def config(cls: ConfigT) -> ConfigT:
    """
    A decorator that ensures all fields in a dataclass have default values
    and that each field has a docstring.
    """
    if not is_dataclass(cls):
        raise TypeError("The decorated class must be a dataclass.")
    attr_docs = get_attr_docs(cls)
    for f in fields(cls):
        if f.init and f.default is MISSING and f.default_factory is MISSING:
            raise ValueError(
                f"Field '{f.name}' in {cls.__name__} must have a default value."
            )
        if f.name not in attr_docs:
            raise ValueError(
                f"Field '{f.name}' in {cls.__name__} must have a docstring.")
    return cls


def get_field(cls: ConfigType, name: str) -> Field:
    """Get the default factory field of a dataclass by name. Used for getting
    default factory fields in `EngineArgs`."""
    if not is_dataclass(cls):
        raise TypeError("The given class is not a dataclass.")
    cls_fields = {f.name: f for f in fields(cls)}
    if name not in cls_fields:
        raise ValueError(f"Field '{name}' not found in {cls.__name__}.")
    named_field: Field = cls_fields.get(name)
    if (default_factory := named_field.default_factory) is not MISSING:
        return field(default_factory=default_factory)
    if (default := named_field.default) is not MISSING:
        return field(default=default)
    raise ValueError(
        f"{cls.__name__}.{name} must have a default value or default factory.")


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
            It is also used as the content for `model_name` tag in metrics
            output when `served_model_name` is not specified.
        task: The task to use the model for. Each vLLM instance only supports
            one task, even if the same model can be used for multiple tasks.
            When the model only supports one task, "auto" can be used to select
            it; otherwise, you must specify explicitly which task to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, "slow" will always use the slow tokenizer,
            "mistral" will always use the tokenizer from `mistral_common`, and
            "custom" will use --tokenizer to select the preregistered tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images or
            videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        spec_target_max_model_len: Specify the the maximum length for spec
            decoding draft models.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
            If None, the user did not specify, so default to False.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        max_logprobs: Maximum number of log probabilities. Defaults to 20.
        disable_sliding_window: Whether to disable sliding window. If True,
            we will disable the sliding window functionality of the model.
            If the model does not support sliding window, this argument is
            ignored.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model
            names provided, the first name will be used. If not specified,
            the model name will be the same as `model`.
        limit_mm_per_prompt: Maximum number of data items per modality
            per prompt. Only applicable for multimodal models.
        use_async_output_proc: Whether to use async output processor.
            Defaults to True.
        config_format: The config format which shall be loaded.
            Defaults to 'auto' which defaults to 'hf'.
        hf_token: The token to use as HTTP bearer authorization for remote files
            . If `True`, will use the token generated when running
            `huggingface-cli login` (stored in `~/.huggingface`).
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor.
        disable_mm_preprocessor_cache: If true, then disables caching of the
            multi-modal preprocessor/mapper. (not recommended)
        override_neuron_config: Initialize non default neuron config or
            override default neuron config that are specific to Neuron devices,
            this argument will be used to configure the neuron config that
            can not be gathered from the vllm arguments.
        override_pooler_config: Initialize non default pooling config or
            override default pooling config for the pooling model.
        logits_processor_pattern: Optional regex pattern specifying valid
            logits processor qualified names that can be passed with the
            `logits_processors` extra completion argument. Defaults to None,
            which allows no processors.
        generation_config: Configuration parameter file for generation.
        model_impl: Which implementation of the model to use:
            "auto" will try to use the vLLM implementation if it exists and
                fall back to the Transformers implementation if no vLLM
                implementation is available.
            "vllm" will use the vLLM model implementation.
            "transformers" will use the Transformers model implementation.
        override_generation_config: Override the generation config with the
            given config.
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
        factors.append(self.mm_processor_kwargs)
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

    def __init__(
        self,
        model: str,
        task: Union[TaskOption, Literal["draft"]],
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        dtype: Union[str, torch.dtype],
        seed: int,
        hf_config_path: Optional[str] = None,
        allowed_local_media_path: str = "",
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        rope_scaling: Optional[dict[str, Any]] = None,
        rope_theta: Optional[float] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        spec_target_max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 20,
        disable_sliding_window: bool = False,
        disable_cascade_attn: bool = False,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, list[str]]] = None,
        limit_mm_per_prompt: Optional[dict[str, int]] = None,
        use_async_output_proc: bool = True,
        config_format: ConfigFormat = ConfigFormat.AUTO,
        hf_token: Optional[Union[bool, str]] = None,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        disable_mm_preprocessor_cache: bool = False,
        override_neuron_config: Optional[dict[str, Any]] = None,
        override_pooler_config: Optional["PoolerConfig"] = None,
        logits_processor_pattern: Optional[str] = None,
        generation_config: str = "auto",
        enable_sleep_mode: bool = False,
        override_generation_config: Optional[dict[str, Any]] = None,
        model_impl: Union[str, ModelImpl] = ModelImpl.AUTO,
    ) -> None:
        self.model = maybe_model_redirect(model)
        self.tokenizer = maybe_model_redirect(tokenizer)

        self.hf_config_path = hf_config_path
        if isinstance(hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(hf_config_path)

        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.allowed_local_media_path = allowed_local_media_path
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.model_impl = model_impl

        if hf_overrides is None:
            hf_overrides = {}

        if callable(hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = hf_overrides
        else:
            hf_overrides_kw = hf_overrides
            hf_overrides_fn = None

        if rope_scaling is not None:
            hf_override: dict[str, Any] = {"rope_scaling": rope_scaling}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides)
            msg = (
                "`--rope-scaling` will be removed in a future release. "
                f"'Please instead use `--hf-overrides '{hf_overrides_str}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        if rope_theta is not None:
            hf_override = {"rope_theta": rope_theta}
            hf_overrides_kw.update(hf_override)
            hf_overrides_str = json.dumps(hf_overrides)
            msg = (
                "`--rope-theta` will be removed in a future release. "
                f"'Please instead use `--hf-overrides '{hf_overrides_str}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)

        self.maybe_pull_model_tokenizer_for_s3(model, tokenizer)

        if (backend := envs.VLLM_ATTENTION_BACKEND
            ) and backend == "FLASHINFER" and find_spec("flashinfer") is None:
            raise ValueError(
                "VLLM_ATTENTION_BACKEND is set to FLASHINFER, but flashinfer "
                "module was not found. See "
                "https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile "  # noqa: E501
                "for instructions on how to install it.")

        # The tokenizer version is consistent with the model version by default.
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.disable_cascade_attn = disable_cascade_attn
        self.skip_tokenizer_init = skip_tokenizer_init
        self.enable_sleep_mode = enable_sleep_mode

        from vllm.platforms import current_platform

        if (self.enable_sleep_mode
                and not current_platform.is_sleep_mode_available()):
            raise ValueError(
                "Sleep mode is not supported on current platform.")

        hf_config = get_config(self.hf_config_path or self.model,
                               trust_remote_code, revision, code_revision,
                               config_format)

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
            self.model, hf_token=hf_token, revision=revision)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.use_async_output_proc = use_async_output_proc
        self.mm_processor_kwargs = mm_processor_kwargs
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache

        # Set enforce_eager to False if the value is unset.
        if self.enforce_eager is None:
            self.enforce_eager = False

        interleaved_attn_models = ["gemma2", "gemma3_text", "cohere2"]
        sliding_window = getattr(self.hf_text_config, "sliding_window", None)
        has_interleaved_attention = (sliding_window is not None) and (
            isinstance(sliding_window, list) or
            (self.hf_text_config.model_type in interleaved_attn_models))

        if (not self.disable_sliding_window and has_interleaved_attention):
            if (backend :=
                    envs.VLLM_ATTENTION_BACKEND) in ("XFORMERS", "FLASHINFER"):
                sliding_window_len_min = get_min_sliding_window(
                    self.hf_text_config.sliding_window)

                logger.warning_once(
                    f"{self.hf_text_config.model_type} has interleaved "
                    "attention, which is currently not supported by the "
                    f"{backend} backend. Disabling sliding window and capping "
                    "the max length to the sliding window size "
                    f"({sliding_window_len_min}).")
                self.disable_sliding_window = True
            else:
                # for a model with interleaved attention,
                # the scheduler and the model treat it as full attention
                # (i.e., not dropping any tokens outside the window).
                # only the attention layer itself is aware of the sliding
                # window, and use the window size to compute the attention.
                self.hf_text_config.interleaved_sliding_window = sliding_window
                delattr(self.hf_text_config, "sliding_window")
                sliding_window = None

        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
            spec_target_max_model_len=spec_target_max_model_len,
            encoder_config=self.encoder_config)
        self.served_model_name = get_served_model_name(model,
                                                       served_model_name)
        self.multimodal_config = self._init_multimodal_config(
            limit_mm_per_prompt)
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        self.is_attention_free = self._init_attention_free()
        self.is_hybrid = self._init_is_hybrid()
        self.has_noops = self._init_has_noops()
        self.has_inner_state = self._init_has_inner_state()

        if current_platform.is_neuron():
            self.override_neuron_config = override_neuron_config
        else:
            self.override_neuron_config = None

        supported_tasks, task = self._resolve_task(task)
        self.supported_tasks = supported_tasks
        self.task: Final = task
        if self.task in ("draft", "generate"):
            self.truncation_side = "left"
        else:
            self.truncation_side = "right"

        self.pooler_config = self._init_pooler_config(override_pooler_config)
        self.logits_processor_pattern = logits_processor_pattern

        self.generation_config = generation_config
        self.override_generation_config = override_generation_config or {}

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

    @property
    def registry(self):
        return ModelRegistry

    @property
    def architectures(self) -> list[str]:
        return getattr(self.hf_config, "architectures", [])

    def maybe_pull_model_tokenizer_for_s3(self, model: str,
                                          tokenizer: str) -> None:
        """
        Pull the model config or tokenizer to a temporary
        directory in case of S3.

        Args:
            model: The model name or path.
            tokenizer: The tokenizer name or path.

        """
        if is_s3(model) or is_s3(tokenizer):
            if is_s3(model):
                s3_model = S3Model()
                s3_model.pull_files(
                    model, allow_pattern=["*.model", "*.py", "*.json"])
                self.model_weights = self.model
                self.model = s3_model.dir

            if is_s3(tokenizer):
                s3_tokenizer = S3Model()
                s3_tokenizer.pull_files(
                    model, ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
                self.tokenizer = s3_tokenizer.dir

    def _init_multimodal_config(
        self, limit_mm_per_prompt: Optional[dict[str, int]]
    ) -> Optional["MultiModalConfig"]:
        if self.registry.is_multimodal_model(self.architectures):
            return MultiModalConfig(limit_per_prompt=limit_mm_per_prompt or {})

        if limit_mm_per_prompt:
            raise ValueError("`limit_mm_per_prompt` is only supported for "
                             "multimodal models.")

        return None

    def _get_encoder_config(self):
        return get_sentence_transformer_tokenizer_config(
            self.model, self.revision)

    def _init_pooler_config(
        self,
        override_pooler_config: Optional["PoolerConfig"],
    ) -> Optional["PoolerConfig"]:

        if self.runner_type == "pooling":
            user_config = override_pooler_config or PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                for k, v in base_config.items():
                    if getattr(user_config, k) is None:
                        setattr(user_config, k, v)

            if self.is_matryoshka:
                if user_config.normalize is None:
                    user_config.normalize = True
                elif not user_config.normalize:
                    raise ValueError(
                        "`normalize` must be enabled (set to True) "
                        "for models that are compatible with "
                        "Matryoshka Representation.")

            return user_config

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
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow", "mistral", "custom"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto', 'slow', 'mistral' or 'custom'.")
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
        task_option: Union[TaskOption, Literal["draft"]],
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
                preferred_task = self._get_preferred_task(
                    architectures, supported_tasks)
                if preferred_task != "embed":
                    msg = ("The 'embedding' task will be restricted to "
                           "embedding models in a future release. Please "
                           "pass `--task classify`, `--task score`, or "
                           "`--task reward` explicitly for other pooling "
                           "models.")
                    warnings.warn(msg, DeprecationWarning, stacklevel=2)

                task_option = preferred_task or "embed"

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
            "quark", "nvfp4", "bitblas", "gptq_bitblas"
        ]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

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
        if self.max_seq_len_to_capture is None:
            self.max_seq_len_to_capture = self.max_model_len
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)
        ROCM_UNSUPPORTED_MODELS = ['mllama']
        if (self.hf_config.model_type in ROCM_UNSUPPORTED_MODELS
                and not self.enforce_eager and current_platform.is_rocm()):
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to the eager mode.", self.hf_config.model_type)
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

    def verify_async_output_proc(self, parallel_config, speculative_config,
                                 device_config) -> None:
        if not self.use_async_output_proc:
            # Nothing to check
            return

        if parallel_config.pipeline_parallel_size > 1:
            self.use_async_output_proc = False
            return

        # Reminder: Please update docs/source/features/compatibility_matrix.md
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

        # Reminder: Please update docs/source/features/compatibility_matrix.md
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

        if hasattr(self.hf_text_config, "head_dim"):
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
        if self.hf_text_config.model_type == "deepseek_mtp":
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
        return _TASK_RUNNER[self.task]

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


BlockSize = Literal[1, 8, 16, 32, 64, 128]
CacheDType = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
PrefixCachingHashAlgo = Literal["builtin", "sha256"]


@config
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: BlockSize = None  # type: ignore
    """Size of a contiguous cache block in number of tokens. This is ignored on
    neuron devices and set to `--max-model-len`. On CUDA devices, only block
    sizes up to 32 are supported. On HPU devices, block size defaults to 128.

    This config has no static default. If left unspecified by the user, it will
    be set in `Platform.check_and_update_configs()` based on the current
    platform."""
    gpu_memory_utilization: float = 0.9
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    swap_space: float = 4
    """Size of the CPU swap space per GPU (in GiB)."""
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3)."""
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: Optional[int] = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: Optional[int] = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: Optional[bool] = None
    """Whether to enable prefix caching. Disabled by default for V0. Enabled by
    default for V1."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "builtin"
    """Set the hash algorithm for prefix caching:\n
    - "builtin" is Python's built-in hash.\n
    - "sha256" is collision resistant but with certain overheads."""
    cpu_offload_gb: float = 0
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    """
    calculate_kv_scales: bool = False
    """This enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""

    # Will be set after profiling.
    num_gpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: Optional[int] = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

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
        factors.append(self.cache_dtype)
        # `cpu_offload_gb` does not use `torch.compile` yet.
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        self.swap_space_bytes = self.swap_space * GiB_bytes

        self._verify_args()
        self._verify_cache_dtype()
        self._verify_prefix_caching()

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self) -> None:
        if self.cpu_offload_gb < 0:
            raise ValueError("CPU offload space must be non-negative"
                             f", but got {self.cpu_offload_gb}")

        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in get_args(CacheDType):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return

        if self.sliding_window is not None and not envs.VLLM_USE_V1:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")

        if (self.enable_prefix_caching and self.prefix_caching_hash_algo
                not in get_args(PrefixCachingHashAlgo)):
            raise ValueError(
                "Unknown prefix caching hash algorithm: "
                f"{self.prefix_caching_hash_algo}. Must be one of "
                f"{get_args(PrefixCachingHashAlgo)}.")

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / GiB_bytes:.2f} GiB out of the "
               f"{total_cpu_memory / GiB_bytes:.2f} GiB total CPU memory "
               "is allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. %s", msg)


@config
@dataclass
class TokenizerPoolConfig:
    """This config is deprecated and will be removed in a future release.

    Passing these parameters will have no effect. Please remove them from your
    configurations.
    """

    pool_size: int = 0
    """This parameter is deprecated and will be removed in a future release.
    Passing this parameter will have no effect. Please remove it from your
    configurations."""
    pool_type: str = "ray"
    """This parameter is deprecated and will be removed in a future release.
    Passing this parameter will have no effect. Please remove it from your
    configurations."""
    extra_config: dict = field(default_factory=dict)
    """This parameter is deprecated and will be removed in a future release.
    Passing this parameter will have no effect. Please remove it from your
    configurations."""

    def __post_init__(self) -> None:
        logger.warning_once(
            "TokenizerPoolConfig is deprecated and will be removed in a "
            "future release. Passing this parameter will have no effect. "
            "Please remove it from your configurations.")


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    TENSORIZER = "tensorizer"
    SHARDED_STATE = "sharded_state"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    MISTRAL = "mistral"
    RUNAI_STREAMER = "runai_streamer"
    RUNAI_STREAMER_SHARDED = "runai_streamer_sharded"
    FASTSAFETENSORS = "fastsafetensors"


@config
@dataclass
class LoadConfig:
    """Configuration for loading the model weights."""

    load_format: Union[str, LoadFormat,
                       "BaseModelLoader"] = LoadFormat.AUTO.value
    """The format of the model weights to load:\n
    - "auto" will try to load the weights in the safetensors format and fall
    back to the pytorch bin format if safetensors format is not available.\n
    - "pt" will load the weights in the pytorch bin format.\n
    - "safetensors" will load the weights in the safetensors format.\n
    - "npcache" will load the weights in pytorch format and store a numpy cache
    to speed up the loading.\n
    - "dummy" will initialize the weights with random values, which is mainly
    for profiling.\n
    - "tensorizer" will use CoreWeave's tensorizer library for fast weight
    loading. See the Tensorize vLLM Model script in the Examples section for
    more information.\n
    - "runai_streamer" will load the Safetensors weights using Run:ai Model
    Streamer.\n
    - "bitsandbytes" will load the weights using bitsandbytes quantization.\n
    - "sharded_state" will load weights from pre-sharded checkpoint files,
    supporting efficient loading of tensor-parallel models.\n
    - "gguf" will load weights from GGUF format files (details specified in
    https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).\n
    - "mistral" will load weights from consolidated safetensors files used by
    Mistral models."""
    download_dir: Optional[str] = None
    """Directory to download and load the weights, default to the default
    cache directory of Hugging Face."""
    model_loader_extra_config: dict = field(default_factory=dict)
    """Extra config for model loader. This will be passed to the model loader
    corresponding to the chosen load_format. This should be a JSON string that
    will be parsed into a dictionary."""
    ignore_patterns: Optional[Union[list[str], str]] = None
    """The list of patterns to ignore when loading the model. Default to
    "original/**/*" to avoid repeated loading of llama's checkpoints."""
    use_tqdm_on_load: bool = True
    """Whether to enable tqdm for showing progress bar when loading model
    weights."""

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
        if isinstance(self.load_format, str):
            load_format = self.load_format.lower()
            self.load_format = LoadFormat(load_format)

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]


DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]


@config
@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups."""
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups."""
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    _data_parallel_rank_local: Optional[int] = field(default=None, init=False)
    """Private field to store the local rank of the data parallel group."""

    @property
    def data_parallel_rank_local(self) -> int:
        """Local rank of the data parallel group, defaults to global rank."""
        if self._data_parallel_rank_local is None:
            return self.data_parallel_rank
        return self._data_parallel_rank_local

    @data_parallel_rank_local.setter
    def data_parallel_rank_local(self, value: int) -> None:
        """Set the local rank of the data parallel group."""
        self._data_parallel_rank_local = value

    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""

    max_parallel_loading_workers: Optional[int] = None
    """Maximum number of parallal loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    tokenizer_pool_config: Optional[TokenizerPoolConfig] = None
    """This parameter is deprecated and will be removed in a future release.
    Please remove it from your configs"""

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    placement_group: Optional["PlacementGroup"] = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: Optional[Union[DistributedExecutorBackend,
                                                 type["ExecutorBase"]]] = None
    """Backend to use for distributed model
    workers, either "ray" or "mp" (multiprocessing). If the product
    of pipeline_parallel_size and tensor_parallel_size is less than
    or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, this will default
    to "ray" if Ray is installed and fail otherwise. Note that tpu
    and hpu only support Ray for distributed inference."""

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decofing.
    If "auto", the worker class will be determined based on the platform."""
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls."""

    world_size: int = field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""
    world_size_across_dp: int = field(init=False)
    """world_size_across_dp is TPxPPxDP, it is the size of the world
    including data parallelism."""

    rank: int = 0
    """Global rank in distributed setup."""

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        increment the port number each time we need to initialize a
        new process group related to data parallelism.
        """
        answer = self.data_parallel_master_port
        self.data_parallel_master_port += 1
        return answer

    def stateless_init_dp_group(self) -> "ProcessGroup":
        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group)

        # use gloo since the engine process might not have cuda device
        dp_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            self.get_next_dp_init_port(),
            self.data_parallel_rank,
            self.data_parallel_size,
            backend="gloo")

        return dp_group

    @staticmethod
    def has_unfinished_dp(dp_group: "ProcessGroup",
                          has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished],
                              dtype=torch.int32,
                              device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.pipeline_parallel_size)
        factors.append(self.tensor_parallel_size)
        factors.append(self.enable_expert_parallel)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        if self.data_parallel_size > 1:
            # Data parallel was specified in the engine args.
            self.data_parallel_master_port = get_open_port()
            # TODO multi-node
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

        self.world_size_across_dp = self.world_size * self.data_parallel_size

        if self.distributed_executor_backend == "external_launcher":
            import os
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        ray_only_devices: list[str] = []
        from vllm.platforms import current_platform
        if (current_platform.device_type in ray_only_devices
                and self.world_size > 1):
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            if self.distributed_executor_backend != "ray":
                raise ValueError(
                    f"{current_platform.device_type.upper()} backend only "
                    "supports Ray for distributed inference.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_neuron():
                # neuron uses single process to control multiple devices
                backend = "uni"
            elif (current_platform.is_cuda()
                  and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.") from ray_utils.ray_import_err
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.info("Defaulting to use %s for distributed inference",
                        backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        self._verify_args()

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and self.distributed_executor_backend.uses_ray)

    def _verify_args(self) -> None:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase
        from vllm.platforms import current_platform
        if self.distributed_executor_backend not in (
                "ray", "mp", "uni",
                "external_launcher", None) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher' or"
                " custom ExecutorBase subclass.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.info(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform.")
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")

        assert isinstance(self.worker_extension_cls, str), (
            "worker_extension_cls must be a string (qualified class name).")


PreemptionMode = Literal["swap", "recompute"]
SchedulerPolicy = Literal["fcfs", "priority"]


@config
@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    max_num_batched_tokens: int = None  # type: ignore
    """Maximum number of tokens to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_num_seqs: int = None  # type: ignore
    """Maximum number of sequences to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_model_len: int = None  # type: ignore
    """Maximum length of a sequence (including prompt and generated text). This
    is primarily set in `ModelConfig` and that value should be manually
    duplicated here."""

    max_num_partial_prefills: int = 1
    """For chunked prefill, the maximum number of sequences that can be
    partially prefilled concurrently."""

    max_long_partial_prefills: int = 1
    """For chunked prefill, the maximum number of prompts longer than
    long_prefill_token_threshold that will be prefilled concurrently. Setting
    this less than max_num_partial_prefills will allow shorter prompts to jump
    the queue in front of longer prompts in some cases, improving latency."""

    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens."""

    num_lookahead_slots: int = 0
    """The number of slots to allocate per sequence per
    step, beyond the known token ids. This is used in speculative
    decoding to store KV activations of tokens which may or may not be
    accepted.

    NOTE: This will be replaced by speculative config in the future; it is
    present to enable correctness tests until then."""

    delay_factor: float = 0.0
    """Apply a delay (of delay factor multiplied by previous
    prompt latency) before scheduling next prompt."""

    enable_chunked_prefill: bool = None  # type: ignore
    """If True, prefill requests can be chunked based
    on the remaining max_num_batched_tokens."""

    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = field(init=False)
    """Multimodal encoder compute budget, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = field(init=False)
    """Multimodal encoder cache size, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    preemption_mode: Optional[PreemptionMode] = None
    """Whether to perform preemption by swapping or
    recomputation. If not specified, we determine the mode as follows:
    We use recomputation by default since it incurs lower overhead than
    swapping. However, when the sequence group has multiple sequences
    (e.g., beam search), recomputation is not currently supported. In
    such a case, we use swapping instead."""

    num_scheduler_steps: int = 1
    """Maximum number of forward steps per scheduler call."""

    multi_step_stream_outputs: bool = True
    """If False, then multi-step will stream outputs at the end of all steps"""

    send_delta_data: bool = False
    """Private API. If used, scheduler sends delta data to
    workers instead of an entire data. It should be enabled only
    when SPMD worker architecture is enabled. I.e.,
    VLLM_USE_RAY_SPMD_WORKER=1"""

    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:\n
    - "fcfs" means first come first served, i.e. requests are handled in order
    of arrival.\n
    - "priority" means requests are handled based on given priority (lower
    value means earlier handling) and time of arrival deciding any ties)."""

    chunked_prefill_enabled: bool = field(init=False)
    """True if chunked prefill is enabled."""

    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to
    partially schedule a multimodal item. Only used in V1
    This ensures that if a request has a mixed prompt
    (like text tokens TTTT followed by image tokens IIIIIIIIII) where only
    some image tokens can be scheduled (like TTTTIIIII, leaving IIIII),
    it will be scheduled as TTTT in one step and IIIIIIIIII in the next."""

    scheduler_cls: Union[str, type[object]] = "vllm.core.scheduler.Scheduler"
    """The scheduler class to use. "vllm.core.scheduler.Scheduler" is the
    default scheduler. Can be a class directly or the path to a class of form
    "mod.custom_class"."""

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

    def __post_init__(self) -> None:
        if self.max_model_len is None:
            self.max_model_len = 8192
            logger.warning(
                "max_model_len was is not set. Defaulting to arbitrary value "
                "of %d.", self.max_model_len)

        if self.max_num_seqs is None:
            self.max_num_seqs = 128
            logger.warning(
                "max_num_seqs was is not set. Defaulting to arbitrary value "
                "of %d.", self.max_num_seqs)

        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                if self.num_scheduler_steps > 1:
                    # Multi-step Chunked-Prefill doesn't allow prompt-chunking
                    # for now. Have max_num_batched_tokens set to max_model_len
                    # so we don't reject sequences on account of a short
                    # max_num_batched_tokens.
                    self.max_num_batched_tokens = max(
                        self.max_model_len, _DEFAULT_MAX_NUM_BATCHED_TOKENS)
                else:
                    self.max_num_batched_tokens = (
                        _DEFAULT_MAX_NUM_BATCHED_TOKENS)
            else:
                # If max_model_len is too short, use
                # _DEFAULT_MAX_NUM_BATCHED_TOKENS as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(
                    self.max_model_len, _DEFAULT_MAX_NUM_BATCHED_TOKENS)

            if self.runner_type == "pooling":
                # Choose specific value for higher throughput
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    _POOLING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if self.is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    _MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.chunked_prefill_enabled = self.enable_chunked_prefill
        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(self.max_model_len *
                                                        0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills, self.max_long_partial_prefills,
                self.long_prefill_token_threshold)

        self._verify_args()

    def _verify_args(self) -> None:
        if (self.max_num_batched_tokens < self.max_model_len
                and not self.chunked_prefill_enabled):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")

        if self.num_scheduler_steps < 1:
            raise ValueError(
                "num_scheduler_steps "
                f"({self.num_scheduler_steps}) must be greater than or "
                "equal to 1.")

        if self.max_num_partial_prefills < 1:
            raise ValueError(
                f"max_num_partial_prefills ({self.max_num_partial_prefills}) "
                "must be greater than or equal to 1.")
        elif self.max_num_partial_prefills > 1:
            if not self.chunked_prefill_enabled:
                raise ValueError("Chunked prefill must be enabled to set "
                                 "max_num_partial_prefills > 1.")

            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({self.max_model_len}).")

        if (self.max_long_partial_prefills
                < 1) or (self.max_long_partial_prefills
                         > self.max_num_partial_prefills):
            raise ValueError(
                f"max_long_partial_prefills ({self.max_long_partial_prefills}) "
                "must be greater than or equal to 1 and less than or equal to "
                f"max_num_partial_prefills ({self.max_num_partial_prefills}).")

    @property
    def is_multi_step(self) -> bool:
        return self.num_scheduler_steps > 1


Device = Literal["auto", "cuda", "neuron", "cpu", "tpu", "xpu", "hpu"]


@config
@dataclass
class DeviceConfig:
    """Configuration for the device to use for vLLM execution."""

    device: Union[Device, torch.device] = "auto"
    """Device type for vLLM execution."""
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
            self.device_type = self.device

        # Some device types require processing inputs on CPU
        if self.device_type in ["neuron"]:
            self.device = torch.device("cpu")
        elif self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)


SpeculativeMethod = Literal["ngram", "eagle", "medusa", "mlp_speculator",
                            "draft_model"]
SpeculativeAcceptanceMethod = Literal["rejection_sampler",
                                      "typical_acceptance_sampler"]


@config
@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # General speculative decoding control
    num_speculative_tokens: int = field(default=None,
                                        init=True)  # type: ignore
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: Optional[str] = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: Optional[SpeculativeMethod] = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    acceptance_method: SpeculativeAcceptanceMethod = "rejection_sampler"
    """The method to use for accepting draft tokens:\n
    - "rejection_sampler" maps to `RejectionSampler`.\n
    - "typical_acceptance_sampler" maps to `TypicalAcceptanceSampler`.

    If using `typical_acceptance_sampler`, the related configuration
    `posterior_threshold` and `posterior_alpha` should be considered."""
    draft_tensor_parallel_size: Optional[int] = None
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    disable_logprobs: bool = True
    """If set to True, token log probabilities are not returned during
    speculative decoding. If set to False, token log probabilities are returned
    according to the log probability settings in SamplingParams."""

    # Draft model configuration
    quantization: Optional[str] = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    max_model_len: Optional[int] = None
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: Optional[str] = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: Optional[str] = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_mqa_scorer: bool = False
    """Disable the MQA scorer and fall back to batch expansion for scoring
    proposals."""
    disable_by_batch_size: Optional[int] = None
    """Disable speculative decoding for new incoming requests when the number
    of enqueued requests is larger than this value, if provided."""

    # Ngram proposer configuration
    prompt_lookup_max: Optional[int] = None
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: Optional[int] = None
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    # Typical acceptance sampler configuration
    posterior_threshold: Optional[float] = None
    """A threshold value that sets a lower bound on the posterior probability
    of a token in the target model for it to be accepted. This threshold is
    used only when we use the `TypicalAcceptanceSampler` for token acceptance.
    """
    posterior_alpha: Optional[float] = None
    """Scaling factor for entropy-based threshold, applied when using
    `TypicalAcceptanceSampler`."""

    # required configuration params passed from engine
    target_model_config: ModelConfig = field(default=None,
                                             init=True)  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: ParallelConfig = field(default=None,
                                                   init=True)  # type: ignore
    """The parallel configuration for the target model."""
    enable_chunked_prefill: bool = field(default=None,
                                         init=True)  # type: ignore
    """Whether vLLM is configured to use chunked prefill or not. Used for
    raising an error since it's not yet compatible with speculative decode."""
    disable_log_stats: bool = field(default=None, init=True)  # type: ignore
    """Whether to disable the periodic printing of stage times in speculative
    decoding."""

    # params generated in the post-init stage
    draft_model_config: ModelConfig = field(default=None,
                                            init=True)  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: ParallelConfig = field(default=None,
                                                  init=True)  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

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
        # Eagle3 affects the computation graph because it returns intermediate
        # hidden states in addition to the final hidden state.
        factors.append(self.method == "eagle3")
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    @classmethod
    def from_dict(cls, dict_value: dict) -> "SpeculativeConfig":
        """Parse the CLI value for the speculative config."""
        return cls(**dict_value)

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type == "deepseek_v3":
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["DeepSeekMTPModel"]
            })
        return hf_config

    def __post_init__(self):

        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        if self.model is None and self.num_speculative_tokens is not None:
            # TODO(Shangming): Refactor mtp configuration logic when supporting
            # mtp acceleration for more models besides deepseek_v3
            if self.target_model_config and \
                self.target_model_config.hf_text_config.model_type \
                        == "deepseek_v3":
                # use the draft model from the same model:
                self.model = self.target_model_config.model
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            else:
                raise ValueError("num_speculative_tokens was provided without "
                                 "speculative model.")

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (self.model is not None
                                    and self.model in ("ngram", "[ngram]")):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if (self.prompt_lookup_min is None
                    and self.prompt_lookup_max is None):
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                assert self.prompt_lookup_max is not None
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                assert self.prompt_lookup_min is not None
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min < 1:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
            if self.prompt_lookup_max < 1:
                raise ValueError(
                    f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}")

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    task="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.
                    trust_remote_code,
                    allowed_local_media_path=self.target_model_config.
                    allowed_local_media_path,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.
                    tokenizer_revision,
                    max_model_len=None,
                    spec_target_max_model_len=self.target_model_config.
                    max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_seq_len_to_capture=self.target_model_config.
                    max_seq_len_to_capture,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                )

                # Automatically detect the method
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif "eagle-" in self.draft_model_config.model.lower() or \
                        "eagle3-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif (self.draft_model_config.hf_config.model_type ==
                      "mlp_speculator"):
                    self.method = "mlp_speculator"
                else:
                    self.method = "draft_model"

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                        raise ValueError(
                            "Chunked prefill and EAGLE are not compatible "
                            "when using V0.")

                    from vllm.transformers_utils.configs.eagle import (
                        EAGLEConfig)
                    if isinstance(self.draft_model_config.hf_config,
                                  EAGLEConfig):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method)
                        self.draft_model_config.hf_config = eagle_config

                if (self.num_speculative_tokens is not None
                        and hasattr(self.draft_model_config.hf_config,
                                    "num_lookahead_tokens")):
                    self.draft_model_config.hf_config.num_lookahead_tokens = \
                    self.num_speculative_tokens

                n_predict = getattr(self.draft_model_config.hf_config,
                                    "n_predict", None)
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif self.num_speculative_tokens > n_predict and \
                            self.num_speculative_tokens % n_predict != 0:
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}")

                self.draft_tensor_parallel_size = \
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    ))

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size))

        if self.acceptance_method == "typical_acceptance_sampler":
            if self.posterior_threshold is None:
                self.posterior_threshold = 0.09
            if self.posterior_alpha is None:
                self.posterior_alpha = 0.3

        self._verify_args()

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: Optional[int],
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:

            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(f"{speculative_max_model_len=} cannot be "
                                 f"larger than {draft_max_model_len=}")

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(f"{speculative_max_model_len=} cannot be "
                                 f"larger than {target_max_model_len=}")

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
            target_parallel_config: ParallelConfig,
            speculative_draft_tensor_parallel_size: Optional[int],
            draft_hf_config: PretrainedConfig) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type)
            else:
                speculative_draft_tensor_parallel_size = \
                    target_parallel_config.tensor_parallel_size
        elif speculative_draft_tensor_parallel_size not in (
                1, target_parallel_config.tensor_parallel_size):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size")
        return speculative_draft_tensor_parallel_size

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.
            pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.
            distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.
            max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.
            disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.
            ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    def _verify_args(self) -> None:
        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter.")

        if self.num_speculative_tokens <= 0:
            raise ValueError("Expected num_speculative_tokens to be greater "
                             f"than zero ({self.num_speculative_tokens}).")

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config)
            # Validate and set draft token acceptance related settings.

        if self.acceptance_method is None:
            raise ValueError("acceptance_method is not set. "
                             "Expected values are rejection_sampler or "
                             "typical_acceptance_sampler.")

        if (self.acceptance_method != 'rejection_sampler'
                and self.acceptance_method != 'typical_acceptance_sampler'):
            raise ValueError(
                "Expected acceptance_method to be either "
                "rejection_sampler or typical_acceptance_sampler. Instead it "
                f"is {self.acceptance_method}")

        if self.acceptance_method == "typical_acceptance_sampler" and (
            (self.posterior_threshold is not None
             and self.posterior_threshold < 0) or
            (self.posterior_alpha is not None and self.posterior_alpha < 0)):
            raise ValueError(
                "Expected the posterior_threshold and posterior_alpha of "
                "typical_acceptance_sampler to be > 0. "
                "Instead found posterior_threshold = "
                f"{self.posterior_threshold} and posterior_alpha = "
                f"{self.posterior_alpha}")

        if (self.disable_by_batch_size is not None
                and self.disable_by_batch_size < 2):
            raise ValueError("Expect the batch size threshold of disabling "
                             "speculative decoding is > 1, but got "
                             f"{self.disable_by_batch_size=}")

        if self.method == "eagle3" and self.target_model_config and \
            "llama" not in self.target_model_config.hf_text_config.model_type:
            raise ValueError(
                "Eagle3 is only supported for Llama models. "
                f"Got {self.target_model_config.hf_text_config.model_type=}")

    @property
    def num_lookahead_slots(self) -> int:
        """The number of additional slots the scheduler should allocate per
        step, in addition to the slots allocated for each known token.

        This is equal to the number of speculative tokens, as each speculative
        token must be scored.
        """
        return self.num_speculative_tokens

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3")

    def __repr__(self) -> str:
        method = self.method
        model = None if method == "ngram" else self.draft_model_config.model
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"


LoRADType = Literal["auto", "float16", "bfloat16"]


@config
@dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    max_lora_rank: int = 16
    """Max LoRA rank."""
    max_loras: int = 1
    """Max number of LoRAs in a single batch."""
    fully_sharded_loras: bool = False
    """By default, only half of the LoRA computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    max_cpu_loras: Optional[int] = None
    """Maximum number of LoRAs to store in CPU memory. Must be >= than
    `max_loras`."""
    lora_dtype: Union[torch.dtype, LoRADType] = "auto"
    """Data type for LoRA. If auto, will default to base model dtype."""
    lora_extra_vocab_size: int = 256
    """Maximum size of extra vocabulary that can be present in a LoRA adapter
    (added to the base model vocabulary)."""
    # This is a constant.
    lora_vocab_padding_size: ClassVar[int] = 256
    long_lora_scaling_factors: Optional[tuple[float, ...]] = None
    """Specify multiple scaling factors (which can be different from base model
    scaling factor - see eg. Long LoRA) to allow for multiple LoRA adapters
    trained with those scaling factors to be used at the same time. If not
    specified, only adapters trained with the base model scaling factor are
    allowed."""
    bias_enabled: bool = False
    """Enable bias for LoRA adapters."""

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
        factors.append(self.max_lora_rank)
        factors.append(self.max_loras)
        factors.append(self.fully_sharded_loras)
        factors.append(self.lora_dtype)
        factors.append(self.lora_extra_vocab_size)
        factors.append(self.long_lora_scaling_factors)
        factors.append(self.bias_enabled)
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        # Setting the maximum rank to 512 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
        possible_lora_extra_vocab_size = (256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")

    def verify_with_cache_config(self, cache_config: CacheConfig):
        if cache_config.cpu_offload_gb > 0 and not envs.VLLM_USE_V1:
            raise ValueError(
                "V0 LoRA does not support CPU offload, please use V1.")

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)

    def verify_lora_support(self):
        if self.long_lora_scaling_factors is not None and envs.VLLM_USE_V1:
            raise ValueError(
                "V1 LoRA does not support long LoRA, please use V0.")


@config
@dataclass
class PromptAdapterConfig:
    """Configuration for PromptAdapters."""

    max_prompt_adapters: int = 1
    """Max number of PromptAdapters in a batch."""
    max_prompt_adapter_token: int = 0
    """Max number of PromptAdapters tokens."""
    max_cpu_prompt_adapters: Optional[int] = None
    """Maximum number of PromptAdapters to store in CPU memory. Must be >= than
    `max_prompt_adapters`."""
    prompt_adapter_dtype: Union[torch.dtype, str] = "auto"
    """Data type for PromptAdapter. If auto, will default to base model dtype.
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

    def __post_init__(self):

        if self.max_prompt_adapters < 1:
            raise ValueError(f"max_prompt_adapters "
                             f"({self.max_prompt_adapters}) must be >= 1.")
        if self.max_prompt_adapter_token == 0:
            raise ValueError("max_prompt_adapter_token must be set.")
        if self.max_cpu_prompt_adapters is None:
            self.max_cpu_prompt_adapters = self.max_prompt_adapters

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.prompt_adapter_dtype == "auto":
            self.prompt_adapter_dtype = model_config.dtype
        elif isinstance(self.prompt_adapter_dtype, str):
            self.prompt_adapter_dtype = getattr(torch,
                                                self.prompt_adapter_dtype)


@config
@dataclass
class MultiModalConfig:
    """Controls the behavior of multimodal models."""

    limit_per_prompt: dict[str, int] = field(default_factory=dict)
    """
    The maximum number of input items allowed per prompt for each modality.
    This should be a JSON string that will be parsed into a dictionary.
    Defaults to 1 (V0) or 999 (V1) for each modality.

    For example, to allow up to 16 images and 2 videos per prompt:
    ``{"images": 16, "videos": 2}``
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

    def get_limit_per_prompt(self, modality: str) -> int:
        """
        Get the maximum number of input items allowed per prompt
        for the given modality.
        """
        return self.limit_per_prompt.get(
            modality,
            999 if envs.VLLM_USE_V1 else 1,
        )

    # TODO: Add configs to init vision tower or not.


@config
@dataclass
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    pooling_type: Optional[str] = None
    """
    The pooling method of the pooling model. This should be a key in
    :class:`vllm.model_executor.layers.pooler.PoolingType`.
    """

    normalize: Optional[bool] = None
    """
    Whether to normalize the pooled outputs. Usually, this should be set to
    ``True`` for embedding outputs.
    """

    softmax: Optional[bool] = None
    """
    Whether to apply softmax to the pooled outputs. Usually, this should be set
    to ``True`` for classification outputs.
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

    @staticmethod
    def from_json(json_str: str) -> "PoolerConfig":
        return PoolerConfig(**json.loads(json_str))


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
    config_dtype = getattr(config.get_text_config(), "torch_dtype", None)

    # Fallback for multi-modal models if the root config
    # does not define torch_dtype
    if config_dtype is None and hasattr(config, "vision_config"):
        config_dtype = getattr(config.vision_config, "torch_dtype", None)

    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                # Following common practice, we use float16 for float32 models
                torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype

            if config.model_type == "plamo2":
                logger.info(
                    "For PLaMo2, we cast models to bfloat16 instead of using "
                    "float16 by default. This is because float16 does not work."
                )
                torch_dtype = torch.bfloat16

            from vllm.platforms import current_platform
            if (current_platform.is_cpu()
                    and current_platform.get_cpu_architecture()
                    == CpuArchEnum.POWERPC
                    and (config_dtype == torch.float16
                         or config_dtype == torch.float32)):
                logger.info(
                    "For POWERPC, we cast models to bfloat16 instead of "
                    "using float16 by default. Float16 is not currently "
                    "supported for POWERPC.")
                torch_dtype = torch.bfloat16

            # TODO: change this condition to check if the platform support bf16
            # instead of checking the OS. For instance M2 shall supports bf16
            # already. But we need to modify `cpu_extension.cmake` to activate
            # the feature in the build.
            if (current_platform.is_cpu() and sys.platform.startswith("darwin")
                    and current_platform.get_cpu_architecture()
                    == CpuArchEnum.ARM and config_dtype == torch.bfloat16):
                logger.info("For macOS with Apple Silicon, currently bfloat16 "
                            "is not supported. Setting dtype to float16.")
                torch_dtype = torch.float16

            if current_platform.is_hpu() and config_dtype == torch.float16:
                logger.info(
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


GuidedDecodingBackendV0 = Literal["auto", "outlines", "lm-format-enforcer",
                                  "xgrammar", "guidance"]
GuidedDecodingBackendV1 = Literal["auto", "xgrammar", "guidance"]


@config
@dataclass
class DecodingConfig:
    """Dataclass which contains the decoding strategy of the engine."""

    guided_decoding_backend: Union[
        GuidedDecodingBackendV0,
        GuidedDecodingBackendV1] = "auto" if envs.VLLM_USE_V1 else "xgrammar"
    """Which engine will be used for guided decoding (JSON schema / regex etc)
    by default. With "auto", we will make opinionated choices based on request
    contents and what the backend libraries currently support, so the behavior
    is subject to change in each release."""

    reasoning_backend: Optional[str] = None
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format.
    Required for `--enable-reasoning`."""

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
        backend = GuidedDecodingParams(
            backend=self.guided_decoding_backend).backend_name
        if envs.VLLM_USE_V1:
            valid_guided_backends = get_args(GuidedDecodingBackendV1)
        else:
            valid_guided_backends = get_args(GuidedDecodingBackendV0)
        if backend not in valid_guided_backends:
            raise ValueError(f"Invalid guided_decoding_backend '{backend}',"
                             f" must be one of {valid_guided_backends}")


@dataclass
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""
    show_hidden_metrics: bool = False

    otlp_traces_endpoint: Optional[str] = None

    # Collecting detailed timing information for each request can be expensive.

    # If set, collects the model forward time for the request.
    collect_model_forward_time: bool = False

    # If set, collects the model execute time for the request.
    collect_model_execute_time: bool = False

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
        if not is_otel_available() and self.otlp_traces_endpoint is not None:
            raise ValueError(
                "OpenTelemetry is not available. Unable to configure "
                "'otlp_traces_endpoint'. Ensure OpenTelemetry packages are "
                f"installed. Original error:\n{otel_import_error_traceback}")


class KVTransferConfig(BaseModel):
    """Configuration for distributed KV cache transfer."""

    # The KV connector for vLLM to transmit KV caches between vLLM instances.
    kv_connector: Optional[str] = None

    # The device used by kv connector to buffer the KV cache.
    # Currently only support 'cuda'.
    kv_buffer_device: Optional[str] = "cuda"

    # The buffer size for TorchDistributedConnector. Measured in number of
    # bytes. Recommended value: 1e9 (about 1GB).
    kv_buffer_size: float = 1e9

    # Whether this vLLM instance produces, consumes KV cache, or both. Choices
    # are 'kv_producer', 'kv_consumer', and 'both'.
    kv_role: Optional[str] = None

    # The rank of this vLLM instance in the KV cache transfer. Typical value:
    # 0 for prefill instance, 1 for decode instance.
    # Currently only 1P1D is supported.
    kv_rank: Optional[int] = None

    # The number of parallel instances for KV cache transfer. For
    # PyNcclConnector, this should be 2.
    kv_parallel_size: int = 1

    # The KV connector ip, used to build distributed connection
    kv_ip: str = "127.0.0.1"

    # The KV connector port, used to build distributed connection
    kv_port: int = 14579

    # any extra config that the connector may need
    kv_connector_extra_config: dict[str, Any] = {}

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

    @classmethod
    def from_cli(cls, cli_value: str) -> "KVTransferConfig":
        """Parse the CLI value for the kv cache transfer config."""
        return KVTransferConfig.model_validate_json(cli_value)

    def model_post_init(self, __context: Any) -> None:

        if self.kv_role is not None and self.kv_role not in [
                "kv_producer", "kv_consumer", "kv_both"
        ]:
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are `kv_producer`, `kv_consumer`, "
                f"and `kv_both`")

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError("Please specify kv_disagg_role when kv_connector "
                             "is set, supported roles are `kv_producer`, "
                             "`kv_consumer`, and `kv_both`")

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in ["kv_producer", "kv_consumer", "kv_both"]

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in ["kv_producer", "kv_both"]

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in ["kv_consumer", "kv_both"]

    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)


class CompilationLevel:
    # constants for the levels of the compilation process
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    PIECEWISE = 3


class CompilationConfig(BaseModel):
    """
    Configuration for compilation.
    It has three parts:
    - Top-level Compilation control:
        - level: the level of compilation.
            - 0: no compilation.
            - 1: dynamo as is.
            - 2: dynamo once.
            - 3: piecewise compilation.
        - debug_dump_path: the path to dump the debug information.
        - cache_dir: the directory to store the compiled graph, to
            accelerate Inductor compilation. By default, it will use
            model-related information to generate a cache directory.
        - backend: the backend for compilation. It needs to be a string.
            - "" (empty string): use the default backend.
            - "eager"/"openxla"/...: use the specified backend registered in PyTorch.
            - "full.module.name": a qualified name which can be used to import the backend function.
            We use string to avoid serialization issues when using compilation in a distributed setting.
            When the compilation level is 1 or 2, the backend is used for the compilation directly (it sees the whole graph).
            When the compilation level is 3, the backend is used for the piecewise compilation (it sees a part of the graph).
        - custom_ops: fine-grained control over which custom ops to enable/disable.
            Use 'all' to enable all, 'none' to disable all.
            Also specify a list of custom op names to enable (prefixed with a '+'),
            or disable (prefixed with a '-').
            Examples:
                - 'all,-op1' to enable all except op1
                - 'none,+op1,+op2' to enable only op1 and op2
            By default, all custom ops are enabled when running without Inductor
                and disabled when running with Inductor (compile_level >= Inductor).
        - splitting_ops: a list of ops to split the full graph into subgraphs, used in piecewise compilation.
    - CudaGraph capture:
        - use_cudagraph: whether to use cudagraph inside compilation.
            - False: cudagraph inside compilation is not used.
            - True: cudagraph inside compilation is used. It requires
                that all input buffers have fixed addresses, and all
                splitting ops write their outputs to input buffers.
            Note that this is orthogonal to the cudagraph capture logic
            outside of compilation.
            TODO: move outside cudagraph logic into compilation.
            torch.compile will handle cudagraph capture logic in the future.
        - cudagraph_capture_sizes: sizes to capture cudagraph.
            - None (default): capture sizes are inferred from vllm config.
            - list[int]: capture sizes are specified as given.
        - cudagraph_num_of_warmups: number of warmup runs for cudagraph.
            It means the first several runs will be treated as warmup runs.
            Only after that, the execution will be recorded, and the recorded
            cudagraph will be used for subsequent runs.
        - cudagraph_copy_inputs: whether to copy input tensors for
            cudagraph. If the caller can guarantee that the same input buffers
            are always used, it can set this to False. Otherwise, it should
            set this to True, and the compiler will copy the input to an
            internally managed buffer. Default is False.
    - Inductor compilation:
        - use_inductor: whether to use inductor compilation.
            - False: inductor compilation is not used. graph runs in eager.
            - True: inductor compilation is used. one graph for symbolic shape
                is compiled. In addition, compile for compile_sizes,
                using configurations in inductor_compile_config.
        - compile_sizes: sizes to compile for inductor. In addition
            to integers, it also supports "cudagraph_capture_sizes" to
            specify the sizes for cudagraph capture.
        - inductor_compile_config: additional configurations for inductor.
            - None: use default configurations.
        - inductor_passes: additional passes for inductor. It is a dictionary
            from pass name to pass function qualified name. We use function
            name because the config uses json format. If we pass the config
            from Python, functions can also be passed directly via Python object
            constructor, e.g. `CompilationConfig(inductor_passes={"a": func})`
        - custom inductor passes: see PassConfig for more details

    Why we have different sizes for cudagraph and inductor:
    - cudagraph: a cudagraph captured for a specific size can only be used
        for the same size. We need to capture all the sizes we want to use.
    - inductor: a graph compiled by inductor for a general shape can be used
        for different sizes. Inductor can also compile for specific sizes,
        where it can have more information to optimize the graph with fully
        static shapes. However, we find the general shape compilation is
        sufficient for most cases. It might be beneficial to compile for
        certain small batchsizes, where inductor is good at optimizing.
    """ # noqa
    level: int = 0
    debug_dump_path: str = ""
    cache_dir: str = ""
    backend: str = ""
    custom_ops: list[str] = Field(default_factory=list)
    splitting_ops: list[str] = Field(default=None)  # type: ignore

    use_inductor: bool = True
    compile_sizes: Optional[list[Union[int, str]]] = Field(default=None)
    inductor_compile_config: dict = Field(default_factory=dict)
    inductor_passes: dict[str, str] = Field(default_factory=dict)

    use_cudagraph: bool = False
    cudagraph_num_of_warmups: int = 0
    cudagraph_capture_sizes: Optional[list[int]] = None
    cudagraph_copy_inputs: bool = False

    class PassConfig(BaseModel):
        """
        Configuration for custom Inductor passes.
        This is separate from general CompilationConfig so that inductor passes
        don't all have access to full configuration - that would create a cycle
        as the PassManager is set as a property of config.
        - dump_graph_stages: list of stages for which we want to dump the graph.
            Each pass defines its own stages (before, after, maybe in-between).
        - dump_graph_dir: directory to dump the graphs. Default is .
        - enable_fusion: whether to enable the custom fusion pass.
        - enable_noop: whether to enable the custom no-op elimination pass.
            TODO(luka) better pass enabling system.
        - enable_sequence_parallelism: whether to enable sequence parallelism.
        """
        dump_graph_stages: list[str] = Field(default_factory=list)
        dump_graph_dir: Path = Field(default=Path("."))
        enable_fusion: bool = True
        enable_noop: bool = True
        enable_sequence_parallelism: bool = False

        def uuid(self):
            """
            Produces a hash unique to the pass configuration.
            Any new fields that affect compilation should be added to the hash.
            Do not include dump_graph_* in the hash - they don't affect
            compilation.
            """
            dict_ = self.model_dump(include={"enable_fusion", "enable_noop", \
                "enable_sequence_parallelism"})
            return InductorPass.hash_dict(dict_)

        def model_post_init(self, __context: Any) -> None:
            if not self.enable_noop and self.enable_fusion:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm + quant (fp8) fusion might not work")

    pass_config: PassConfig = Field(default_factory=PassConfig)

    # not configurable, computed after init
    max_capture_size: int = PrivateAttr
    local_cache_dir: str = PrivateAttr  # local cache dir for each rank
    # optimization:
    # Intuitively, bs_to_padded_graph_size should be dict[int, int].
    # since we know all keys are in a range [0, max_capture_size],
    # we can optimize it to list[int] for better lookup performance.
    bs_to_padded_graph_size: list[int] = PrivateAttr

    # keep track of enabled and disabled custom ops
    enabled_custom_ops: Counter[str] = PrivateAttr
    disabled_custom_ops: Counter[str] = PrivateAttr
    traced_files: set[str] = PrivateAttr
    compilation_time: float = PrivateAttr

    # Per-model forward context
    # Map from layer name to layer objects that need to be accessed outside
    # model code, e.g., Attention, FusedMOE when dp_size>1.
    static_forward_context: dict[str, Any] = PrivateAttr

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
        factors.append(self.level)
        factors.append(self.backend)
        factors.append(self.custom_ops)
        factors.append(self.splitting_ops)
        factors.append(self.use_inductor)
        factors.append(self.inductor_compile_config)
        factors.append(self.inductor_passes)
        factors.append(self.pass_config.uuid())
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __repr__(self) -> str:
        exclude = {
            "static_forward_context",
            "enabled_custom_ops",
            "disabled_custom_ops",
            "compilation_time",
            "bs_to_padded_graph_size",
            "pass_config",
            "traced_files",
        }
        return self.model_dump_json(exclude=exclude, exclude_unset=True)

    __str__ = __repr__

    @classmethod
    def from_cli(cls, cli_value: str) -> "CompilationConfig":
        """Parse the CLI value for the compilation config."""
        if cli_value in ["0", "1", "2", "3"]:
            return cls(level=int(cli_value))
        # do not use `eval`, it is dangerous and can execute arbitrary code
        dict_value = ast.literal_eval(cli_value)
        return CompilationConfig.model_validate(dict_value)

    def model_post_init(self, __context: Any) -> None:

        count_none = self.custom_ops.count("none")
        count_all = self.custom_ops.count("all")
        assert count_none + count_all <= 1, "Can only specify 'none' or 'all'"

        # TODO(zou3519/luka): There are 2 issues with auto-functionalization V2:
        # 1. A bug in PyTorch, fixed in 2.7:
        #    https://github.com/pytorch/pytorch/issues/147924
        # 2. Custom passes (fusion) rely on auto-functionalization V1 and don't
        #    work with V2. Addressing this will take extra engineering effort
        #    and it is not yet a priority. RFC here:
        #    https://github.com/vllm-project/vllm/issues/14703

        if is_torch_equal_or_newer("2.6"):
            KEY = 'enable_auto_functionalized_v2'
            if KEY not in self.inductor_compile_config:
                self.inductor_compile_config[KEY] = False

        if self.splitting_ops is None:
            self.splitting_ops = []

        for k, v in self.inductor_passes.items():
            if not isinstance(v, str):
                assert callable(v), (
                    f"pass {k} should be callable or a qualified name")
                self.inductor_compile_config[k] = v if isinstance(
                    v, InductorPass) else CallableInductorPass(v)
                continue

            # resolve function from qualified name
            names = v.split(".")
            module = ".".join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = func if isinstance(
                func, InductorPass) else CallableInductorPass(func)

        self.enabled_custom_ops = Counter()
        self.disabled_custom_ops = Counter()
        self.traced_files = set()
        self.static_forward_context = {}
        self.compilation_time = 0.0

    def init_backend(self, vllm_config: "VllmConfig") -> Union[str, Callable]:
        if self.level == CompilationLevel.NO_COMPILATION:
            raise ValueError("No compilation level is set.")

        from torch._dynamo.backends.registry import list_backends
        torch_backends = list_backends(exclude_tags=tuple())
        if self.level in [
                CompilationLevel.DYNAMO_AS_IS, CompilationLevel.DYNAMO_ONCE
        ]:
            if self.backend == "":
                return "eager"
            if self.backend in torch_backends:
                return self.backend
            return resolve_obj_by_qualname(self.backend)

        # TODO: pass user-specified backend to piecewise compilation
        # merge with the config use_inductor
        assert self.level == CompilationLevel.PIECEWISE

        from vllm.compilation.backends import VllmBackend
        return VllmBackend(vllm_config)

    def init_with_cudagraph_sizes(self,
                                  cudagraph_capture_sizes: list[int]) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes."""

        if self.cudagraph_capture_sizes is None:
            self.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            # de-duplicate the sizes provided by the config
            self.cudagraph_capture_sizes = list(
                set(self.cudagraph_capture_sizes))
            logger.info(("cudagraph sizes specified by model runner"
                         " %s is overridden by config %s"),
                        cudagraph_capture_sizes, self.cudagraph_capture_sizes)

        computed_compile_sizes = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", \
                    "Unrecognized size type in compile_sizes, " \
                    f"expect 'cudagraph_capture_sizes', got {x}"
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

        # sort to make sure cudagraph capture sizes are in descending order
        self.cudagraph_capture_sizes.sort(reverse=True)
        self.max_capture_size = self.cudagraph_capture_sizes[
            0] if self.cudagraph_capture_sizes else 0

        # pre-compute the mapping from batch size to padded graph size
        self.bs_to_padded_graph_size = [
            0 for i in range(self.max_capture_size + 1)
        ]
        for end, start in zip(self.cudagraph_capture_sizes,
                              self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.bs_to_padded_graph_size[bs] = start
                else:
                    self.bs_to_padded_graph_size[bs] = end
        self.bs_to_padded_graph_size[
            self.max_capture_size] = self.max_capture_size

    def set_splitting_ops_for_v1(self):
        # If default, override splitting ops for piecewise cudagraph on V1.
        # NOTE: this function needs to be called
        if not self.splitting_ops:
            self.splitting_ops = [
                "vllm.unified_attention",
                "vllm.unified_attention_with_output",
            ]


@dataclass
class VllmConfig:
    """Dataclass which contains all vllm-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    model_config: ModelConfig = field(default=None, init=True)  # type: ignore
    cache_config: CacheConfig = field(default=None, init=True)  # type: ignore
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig,
                                            init=True)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig,
                                              init=True)
    device_config: DeviceConfig = field(default=None,
                                        init=True)  # type: ignore
    load_config: LoadConfig = field(default=None, init=True)  # type: ignore
    lora_config: Optional[LoRAConfig] = None
    speculative_config: SpeculativeConfig = field(default=None,
                                                  init=True)  # type: ignore
    decoding_config: Optional[DecodingConfig] = None
    observability_config: Optional[ObservabilityConfig] = None
    prompt_adapter_config: Optional[PromptAdapterConfig] = None
    quant_config: Optional[QuantizationConfig] = None
    compilation_config: CompilationConfig = field(default=None,
                                                  init=True)  # type: ignore
    kv_transfer_config: KVTransferConfig = field(default=None,
                                                 init=True)  # type: ignore
    # some opaque config, only used to provide additional information
    # for the hash computation, mainly used for testing, debugging or out of
    # tree config registration.
    additional_config: SupportsHash = field(default=None,
                                            init=True)  # type: ignore
    instance_id: str = ""

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
        if self.prompt_adapter_config:
            vllm_factors.append(self.prompt_adapter_config.compute_hash())
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
            vllm_factors.append(self.additional_config.compute_hash())
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
        if self.model_config is not None:
            self.model_config.verify_async_output_proc(self.parallel_config,
                                                       self.speculative_config,
                                                       self.device_config)
            self.model_config.verify_with_parallel_config(self.parallel_config)

        if self.cache_config is not None:
            self.cache_config.verify_with_parallel_config(self.parallel_config)

        if self.lora_config:
            self.lora_config.verify_with_cache_config(self.cache_config)
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_lora_support()
        if self.prompt_adapter_config:
            self.prompt_adapter_config.verify_with_model_config(
                self.model_config)

        if self.quant_config is None and \
            self.model_config is not None and self.load_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config)

        from vllm.platforms import current_platform
        if self.scheduler_config is not None and \
            self.model_config is not None and \
            self.scheduler_config.chunked_prefill_enabled and \
            self.model_config.dtype == torch.float32 and \
            current_platform.get_device_capability() == (7, 5):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels.")

        if self.compilation_config is None:
            self.compilation_config = CompilationConfig()
        if self.compilation_config.pass_config.enable_sequence_parallelism:
            self.compilation_config.custom_ops.append("+rms_norm")
        if envs.VLLM_USE_V1 and self.model_config is not None and \
            not self.model_config.enforce_eager:
            # NOTE(woosuk): Currently, we use inductor because the piecewise
            # CUDA graphs do not work properly with the custom CUDA kernels.
            # FIXME(woosuk): Disable inductor to reduce the compilation time
            # and avoid any potential issues with the inductor.
            # FIXME(rob): Add function to set all of these.
            if not self.compilation_config.custom_ops:
                self.compilation_config.custom_ops = ["none"]
            self.compilation_config.use_cudagraph = True
            self.compilation_config.use_inductor = True
            self.compilation_config.cudagraph_num_of_warmups = 1
            self.compilation_config.pass_config.enable_fusion = False
            self.compilation_config.pass_config.enable_noop = False
            self.compilation_config.level = CompilationLevel.PIECEWISE
            self.compilation_config.set_splitting_ops_for_v1()

        if self.parallel_config is not None and \
            self.parallel_config.tensor_parallel_size > 1 and \
            self.parallel_config.pipeline_parallel_size > 1 and \
            self.compilation_config is not None and \
                self.compilation_config.pass_config is not None and \
            self.compilation_config.pass_config.enable_sequence_parallelism:
            logger.warning_once(
                "Sequence parallelism is not supported with pipeline "
                "parallelism. Disabling sequence parallelism.")
            self.compilation_config.pass_config.\
                enable_sequence_parallelism = False

        self._set_cudagraph_sizes()

        if self.cache_config is not None and \
            self.cache_config.cpu_offload_gb > 0 and \
            self.compilation_config.level != CompilationLevel.NO_COMPILATION \
                and not envs.VLLM_USE_V1:
            logger.warning(
                "CPU offload is not supported with `torch.compile` in v0 yet."
                " Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if ((not envs.VLLM_USE_V1) and self.lora_config is not None
                and self.compilation_config.level
                != CompilationLevel.NO_COMPILATION):
            logger.warning(
                "LoRA for V0 is not supported with `torch.compile` yet. "
                "Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION


        if self.model_config and self.model_config.use_mla and \
            not (current_platform.is_cuda() or current_platform.is_rocm()):
            logger.info(
                "MLA is enabled on a non-GPU platform; forcing chunked "
                "prefill and prefix caching to be disabled.")
            self.scheduler_config.enable_chunked_prefill = False
            self.scheduler_config.chunked_prefill_enabled = False
            self.scheduler_config.max_num_batched_tokens = max(
                self.scheduler_config.max_model_len,
                _DEFAULT_MAX_NUM_BATCHED_TOKENS)

            if self.cache_config is not None:
                self.cache_config.enable_prefix_caching = False

        current_platform.check_and_update_config(self)

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

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
        cudagraph batchsize padding logic:

        `[1, 2, 4] + [8 * i for i in range(1, 1025)]` is a list of all possible
        batch sizes that cudagraph will capture.

        Depending on the engine's configuration of `max_num_seqs`, the
        candidate batch sizes to capture cudagraph will shrink to the subset
        which just cover the range of `[1, max_num_seqs]`. In the common case,
        `max_num_seqs` is 256, and the cudagraph batch sizes will be
        `[1, 2, 4, 8, 16, 24, 32, 40, ..., 256]`.

        However, if users specify the cudagraph capture sizes through
        compilation config, we will use the specified sizes instead.

        In the end, `vllm_config.compilation_config.cudagraph_capture_sizes`
        will be the final sizes to capture cudagraph (in descending order).

        During runtime, if batchsize is larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        no cudagraph will be used.
        If the batch size is no larger than
        `vllm_config.compilation_config.cudagraph_capture_sizes`,
        we can quickly find the padded graph size for a given batch size by
        looking up `vllm_config.compilation_config.bs_to_padded_graph_size`.
        """

        # calculate the default `batch_size_capture_list`
        if not envs.VLLM_USE_V1:
            batch_size_capture_list = []
            max_batchsize_to_capture = 0
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
                batch_size_capture_list = [1, 2, 4
                                           ] + [i for i in range(8, 513, 8)]
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

    def __str__(self):
        return (
            f"model={self.model_config.model!r},"
            f" speculative_config={self.speculative_config!r},"
            f" tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init},"
            f" tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"override_neuron_config={self.model_config.override_neuron_config},"
            f" tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len},"
            f" download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size},"
            f" pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f" device_config={self.device_config.device}, "
            f"decoding_config={self.decoding_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"num_scheduler_steps={self.scheduler_config.num_scheduler_steps}, "
            f"multi_step_stream_outputs={self.scheduler_config.multi_step_stream_outputs}, "  # noqa
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"chunked_prefill_enabled={self.scheduler_config.chunked_prefill_enabled}, "  # noqa
            f"use_async_output_proc={self.model_config.use_async_output_proc}, "
            f"disable_mm_preprocessor_cache={self.model_config.disable_mm_preprocessor_cache!r}, "  # noqa
            f"mm_processor_kwargs={self.model_config.mm_processor_kwargs}, "
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}")


_current_vllm_config: Optional[VllmConfig] = None


@contextmanager
def set_current_vllm_config(vllm_config: VllmConfig, check_compile=False):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    global _current_vllm_config
    old_vllm_config = _current_vllm_config
    from vllm.compilation.counter import compilation_counter
    num_models_seen = compilation_counter.num_models_seen
    try:
        _current_vllm_config = vllm_config
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


def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the vllm config. In that case, we set a default
        # config.
        logger.warning("Current vLLM config is not set.")
        from vllm.config import VllmConfig
        return VllmConfig()
    return _current_vllm_config


def contains_object_print(text):
    """
    Check if the text looks like a printed Python object, e.g.
    contains any substring matching the pattern: "at 0xFFFFFFF>"
    We match against 0x followed by 2-16 hex chars (there's
    a max of 16 on a 64 bit system).

    Args:
        text (str): The text to check

    Returns:
        bool: True if a match is found, False otherwise
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


def get_layers_from_vllm_config(vllm_config: VllmConfig,
                                layer_type: type[T]) -> dict[str, T]:
    return {
        layer_name: layer
        for layer_name, layer in
        vllm_config.compilation_config.static_forward_context.items()
        if isinstance(layer, layer_type)
    }
