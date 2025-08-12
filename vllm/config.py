import ast
import copy
import enum
import hashlib
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Counter, Dict,
                    Final, List, Literal, Mapping, Optional, Set, Tuple, Type,
                    Union)

import torch
from pydantic import BaseModel, Field, PrivateAttr
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (QUANTIZATION_METHODS,
                                                     get_quantization_config)
from vllm.model_executor.models import ModelRegistry
from vllm.platforms import current_platform, interface
from vllm.tracing import is_otel_available, otel_import_error_traceback
from vllm.transformers_utils.config import (
    ConfigFormat, get_config, get_hf_image_processor_config,
    get_hf_text_config, get_pooling_config,
    get_sentence_transformer_tokenizer_config, is_encoder_decoder,
    try_get_generation_config, uses_mrope)
from vllm.transformers_utils.utils import is_s3
from vllm.utils import (GiB_bytes, LayerBlockType, cuda_device_count_stateless,
                        get_cpu_memory, print_warning_once, random_uuid,
                        resolve_obj_by_qualname)

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from vllm.executor.executor_base import ExecutorBase
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig)
    from vllm.model_executor.model_loader.loader import BaseModelLoader
    from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (
        BaseTokenizerGroup)
else:
    QuantizationConfig = None

logger = init_logger(__name__)

_POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
_MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120

TaskOption = Literal["auto", "generate", "embedding", "embed", "classify",
                     "score", "reward"]

_ResolvedTask = Literal["generate", "embed", "classify", "score", "reward",
                        "draft"]

RunnerType = Literal["generate", "pooling", "draft"]

_RUNNER_TASKS: Dict[RunnerType, List[_ResolvedTask]] = {
    "generate": ["generate"],
    "pooling": ["embed", "classify", "score", "reward"],
    "draft": ["draft"],
}

_TASK_RUNNER: Dict[_ResolvedTask, RunnerType] = {
    task: runner
    for runner, tasks in _RUNNER_TASKS.items() for task in tasks
}

HfOverrides = Union[Dict[str, Any], Callable[[PretrainedConfig],
                                             PretrainedConfig]]


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
            available, "slow" will always use the slow tokenizer, and
            "mistral" will always use the tokenizer from `mistral_common`.
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
        quantization_param_path: Path to JSON file containing scaling factors.
            Used to load KV cache scaling factors into the model when KV cache
            type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also
            be used to load activation and weight scaling factors when the
            model dtype is FP8_E4M3 on ROCm.
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
        factors: List[Any] = []
        factors.append(self.model)
        factors.append(self.dtype)
        factors.append(self.quantization)
        factors.append(self.quantization_param_path)
        factors.append(self.revision)
        factors.append(self.code_revision)
        factors.append(self.trust_remote_code)
        factors.append(self.rope_scaling)
        factors.append(self.rope_theta)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __init__(self,
                 model: str,
                 task: Union[TaskOption, Literal["draft"]],
                 tokenizer: str,
                 tokenizer_mode: str,
                 trust_remote_code: bool,
                 dtype: Union[str, torch.dtype],
                 seed: int,
                 allowed_local_media_path: str = "",
                 revision: Optional[str] = None,
                 code_revision: Optional[str] = None,
                 rope_scaling: Optional[Dict[str, Any]] = None,
                 rope_theta: Optional[float] = None,
                 tokenizer_revision: Optional[str] = None,
                 max_model_len: Optional[int] = None,
                 spec_target_max_model_len: Optional[int] = None,
                 quantization: Optional[str] = None,
                 quantization_param_path: Optional[str] = None,
                 enforce_eager: Optional[bool] = None,
                 max_seq_len_to_capture: Optional[int] = None,
                 max_logprobs: int = 20,
                 disable_sliding_window: bool = False,
                 skip_tokenizer_init: bool = False,
                 served_model_name: Optional[Union[str, List[str]]] = None,
                 limit_mm_per_prompt: Optional[Mapping[str, int]] = None,
                 use_async_output_proc: bool = True,
                 config_format: ConfigFormat = ConfigFormat.AUTO,
                 hf_overrides: Optional[HfOverrides] = None,
                 mm_processor_kwargs: Optional[Dict[str, Any]] = None,
                 disable_mm_preprocessor_cache: bool = False,
                 override_neuron_config: Optional[Dict[str, Any]] = None,
                 override_pooler_config: Optional["PoolerConfig"] = None,
                 logits_processor_pattern: Optional[str] = None,
                 generation_config: Optional[str] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.allowed_local_media_path = allowed_local_media_path
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

        if hf_overrides is None:
            hf_overrides = {}

        if callable(hf_overrides):
            hf_overrides_kw = {}
            hf_overrides_fn = hf_overrides
        else:
            hf_overrides_kw = hf_overrides
            hf_overrides_fn = None

        if rope_scaling is not None:
            hf_override: Dict[str, Any] = {"rope_scaling": rope_scaling}
            hf_overrides_kw.update(hf_override)
            msg = ("`--rope-scaling` will be removed in a future release. "
                   f"'Please instead use `--hf-overrides '{hf_override!r}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)
        if rope_theta is not None:
            hf_override = {"rope_theta": rope_theta}
            hf_overrides_kw.update(hf_override)
            msg = ("`--rope-theta` will be removed in a future release. "
                   f"'Please instead use `--hf-overrides '{hf_override!r}'`")
            warnings.warn(DeprecationWarning(msg), stacklevel=2)

        self.maybe_pull_model_tokenizer_for_s3(model, tokenizer)

        # The tokenizer version is consistent with the model version by default.
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init

        hf_config = get_config(self.model, trust_remote_code, revision,
                               code_revision, config_format)

        if hf_overrides_kw:
            logger.info("Overriding HF config with %s", hf_overrides_kw)
            hf_config.update(hf_overrides_kw)
        if hf_overrides_fn:
            logger.info("Overriding HF config with %s", hf_overrides_fn)
            hf_config = hf_overrides_fn(hf_config)

        self.hf_config = hf_config

        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.encoder_config = self._get_encoder_config()
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, revision)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        self.use_async_output_proc = use_async_output_proc
        self.mm_processor_kwargs = mm_processor_kwargs
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache

        # Set enforce_eager to False if the value is unset.
        if self.enforce_eager is None:
            self.enforce_eager = False

        sliding_window = getattr(self.hf_text_config, "sliding_window", None)
        has_interleaved_attention = (sliding_window is not None) and (
            isinstance(sliding_window, list) or
            (self.hf_text_config.model_type in ["gemma2"]))

        if (not self.disable_sliding_window and has_interleaved_attention):
            if envs.VLLM_ATTENTION_BACKEND == "XFORMERS":
                sliding_window_len_min = get_min_sliding_window(
                    self.hf_text_config.sliding_window)

                print_warning_once(
                    f"{self.hf_text_config.model_type} has interleaved "
                    "attention, which is currently not supported by the "
                    "XFORMERS backend. Disabling sliding window and capping "
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
        self.has_inner_state = self._init_has_inner_state()

        if current_platform.is_neuron():
            self.override_neuron_config = override_neuron_config
        else:
            self.override_neuron_config = None

        supported_tasks, task = self._resolve_task(task, self.hf_config)
        self.supported_tasks = supported_tasks
        self.task: Final = task

        self.pooler_config = self._init_pooler_config(override_pooler_config)
        self.logits_processor_pattern = logits_processor_pattern

        self.generation_config = generation_config

        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

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
            try:
                from vllm.transformers_utils.s3_utils import S3Model
            except ImportError as err:
                raise ImportError(
                    "Please install Run:ai optional dependency "
                    "to use the S3 capabilities. "
                    "You can install it with: pip install vllm[runai]"
                ) from err

            if is_s3(model):
                self.s3_model = S3Model()
                self.s3_model.pull_files(model, allow_pattern=["*config.json"])
                self.model_weights = self.model
                self.model = self.s3_model.dir

            if is_s3(tokenizer):
                self.s3_tokenizer = S3Model()
                self.s3_tokenizer.pull_files(
                    model, ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
                self.tokenizer = self.s3_tokenizer.dir

    def _init_multimodal_config(
        self, limit_mm_per_prompt: Optional[Mapping[str, int]]
    ) -> Optional["MultiModalConfig"]:
        architectures = getattr(self.hf_config, "architectures", [])
        if ModelRegistry.is_multimodal_model(architectures):
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

            return user_config

        return None

    def _init_attention_free(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.is_attention_free_model(architectures)

    def _init_is_hybrid(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.is_hybrid_model(architectures)

    def _init_has_inner_state(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.model_has_inner_state(architectures)

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow", "mistral"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto', 'slow' or 'mistral'.")
        self.tokenizer_mode = tokenizer_mode

    def _get_preferred_task(
        self,
        architectures: List[str],
        supported_tasks: Set[_ResolvedTask],
    ) -> Optional[_ResolvedTask]:
        model_id = self.model
        if get_pooling_config(model_id, self.revision):
            return "embed"
        if ModelRegistry.is_cross_encoder_model(architectures):
            return "score"

        suffix_to_preferred_task: List[Tuple[str, _ResolvedTask]] = [
            # Other models follow this pattern
            ("ForCausalLM", "generate"),
            ("ForConditionalGeneration", "generate"),
            ("ForSequenceClassification", "classify"),
            ("ChatModel", "generate"),
            ("LMHeadModel", "generate"),
            ("EmbeddingModel", "embed"),
            ("RewardModel", "reward"),
        ]
        _, arch = ModelRegistry.inspect_model_cls(architectures)

        for suffix, pref_task in suffix_to_preferred_task:
            if arch.endswith(suffix) and pref_task in supported_tasks:
                return pref_task

        return None

    def _resolve_task(
        self,
        task_option: Union[TaskOption, Literal["draft"]],
        hf_config: PretrainedConfig,
    ) -> Tuple[Set[_ResolvedTask], _ResolvedTask]:
        if task_option == "draft":
            return {"draft"}, "draft"

        architectures = getattr(hf_config, "architectures", [])

        runner_support: Dict[RunnerType, bool] = {
            # NOTE: Listed from highest to lowest priority,
            # in case the model supports multiple of them
            "generate": ModelRegistry.is_text_generation_model(architectures),
            "pooling": ModelRegistry.is_pooling_model(architectures),
        }
        supported_runner_types_lst: List[RunnerType] = [
            runner_type
            for runner_type, is_supported in runner_support.items()
            if is_supported
        ]

        supported_tasks_lst: List[_ResolvedTask] = [
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
            "awq_marlin", "fbgemm_fp8", "compressed_tensors",
            "compressed-tensors", "experts_int8"
        ]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Detect which checkpoint is it
            for name in QUANTIZATION_METHODS:
                method = get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override:
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

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.44.0) with 8-bit models does not
        yet support CUDA graph.
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
                "CUDA graph is not supported on BitAndBytes 8bit yet, "
                "fallback to the eager mode.")
            self.enforce_eager = True

    def verify_async_output_proc(self, parallel_config, speculative_config,
                                 device_config) -> None:
        if not self.use_async_output_proc:
            # Nothing to check
            return

        if parallel_config.pipeline_parallel_size > 1:
            logger.warning("Async output processing can not be enabled "
                           "with pipeline parallel")
            self.use_async_output_proc = False
            return

        # Reminder: Please update docs/source/usage/compatibility_matrix.rst
        # If the feature combo become valid
        if not current_platform.is_async_output_supported(self.enforce_eager):
            logger.warning(
                "Async output processing is not supported on the "
                "current platform type %s.", current_platform.device_type)
            self.use_async_output_proc = False
            return

        if envs.VLLM_USE_RAY_SPMD_WORKER:
            logger.warning(
                "Async output processing can not be enabled with ray spmd")
            self.use_async_output_proc = False
            return

        # Async postprocessor is not necessary for pooling models
        # since there is no token generation
        if self.runner_type == "pooling":
            self.use_async_output_proc = False

        # Reminder: Please update docs/source/usage/compatibility_matrix.rst
        # If the feature combo become valid
        if speculative_config:
            logger.warning("Async output processing is not supported with"
                           " speculative decoding currently.")
            self.use_async_output_proc = False

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = getattr(self.hf_text_config,
                                            "num_attention_heads", 0)
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1:
            architectures = getattr(self.hf_config, "architectures", [])
            if not ModelRegistry.is_pp_supported_model(architectures):
                raise NotImplementedError(
                    "Pipeline parallelism is not supported for this model. "
                    "Supported models implement the `SupportsPP` interface.")

            if self.use_async_output_proc:
                logger.warning("Async output processor is not supported with "
                               "pipeline parallelism currently. Disabling it.")
                self.use_async_output_proc = False

    def get_hf_config_sliding_window(
            self) -> Union[Optional[int], List[Optional[int]]]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (hasattr(self.hf_text_config, "use_sliding_window")
                and not self.hf_text_config.use_sliding_window):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_sliding_window(self) -> Optional[Union[int, List[Optional[int]]]]:
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

    def get_head_size(self) -> int:
        # TODO remove hard code
        if hasattr(self.hf_text_config, "model_type"
                   ) and self.hf_text_config.model_type == 'deepseek_v2':
            # FlashAttention supports only head_size 32, 64, 128, 256,
            # we need to pad head_size 192 to 256
            return 256

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
            self, parallel_config: "ParallelConfig") -> Tuple[int, int]:
        from vllm.distributed.utils import get_pp_indices
        total_num_hidden_layers = getattr(self.hf_text_config,
                                          "num_hidden_layers", 0)
        pp_rank = parallel_config.rank // parallel_config.tensor_parallel_size
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
        is_transformer = not self.is_hybrid and not self.is_attention_free
        start, end = self.get_layers_start_end_indices(parallel_config)

        if is_transformer:
            # Handle the basic case first
            return end - start if attn_block_type else 0
        elif self.is_attention_free:
            # Attention free
            # Note that this code assumes there
            # is only one type of attention-free block type.
            return 0 if attn_block_type else end - start
        else:
            # Hybrid model
            layers_block_type_value = getattr(self.hf_config,
                                              "layers_block_type", None)
            if layers_block_type_value is None:
                raise ValueError("The model is an hybrid without a"
                                 "layers_block_type in the hf_config,"
                                 "cannot determine the num of "
                                 f"{block_type.value} layers")

            return sum(t == block_type.value
                       for t in layers_block_type_value[start:end])

    def get_multimodal_config(self) -> "MultiModalConfig":
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")

        return self.multimodal_config

    def try_get_generation_config(self) -> Dict[str, Any]:
        if self.generation_config is None or self.generation_config == "auto":
            config = try_get_generation_config(
                self.model,
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

    def get_diff_sampling_param(self) -> Dict[str, Any]:
        """
        This method returns a dictionary containing the parameters 
        that differ from the default sampling parameters, but only 
        if `generation_config` is set. If `generation_config` is not 
        set, an empty dictionary is returned.

        Returns:
            Dict[str, Any]: A dictionary with the differing sampling 
            parameters if `generation_config` is set, otherwise an 
            empty dictionary.
        """
        if self.generation_config is None:
            # When generation_config is not set
            return {}
        config = self.try_get_generation_config()
        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
        ]
        if any(p in config for p in available_params):
            diff_sampling_param = {
                p: config.get(p)
                for p in available_params if config.get(p) is not None
            }
        else:
            diff_sampling_param = {}
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
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.is_cross_encoder_model(architectures)

    @property
    def supported_runner_types(self) -> Set[RunnerType]:
        return {_TASK_RUNNER[task] for task in self.supported_tasks}

    @property
    def runner_type(self) -> RunnerType:
        return _TASK_RUNNER[self.task]


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
        is_attention_free: Whether the model is attention-free.
        num_gpu_blocks_override: Number of GPU blocks to use. This overrides the
            profiled num_gpu_blocks if specified. Does nothing if None.
        sliding_window: Sliding window size for the KV cache. Can not work with
            prefix caching enabled.
        enable_prefix_caching: Whether to enable prefix caching.
        cpu_offload_gb: Size of the CPU offload buffer in GiB.
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
        factors: List[Any] = []
        factors.append(self.cache_dtype)
        # `cpu_offload_gb` does not use `torch.compile` yet.
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: float,
        cache_dtype: str,
        is_attention_free: bool = False,
        num_gpu_blocks_override: Optional[int] = None,
        sliding_window: Optional[int] = None,
        enable_prefix_caching: bool = False,
        cpu_offload_gb: float = 0,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * GiB_bytes
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.cache_dtype = cache_dtype
        self.is_attention_free = is_attention_free
        self.sliding_window = sliding_window
        self.enable_prefix_caching = enable_prefix_caching
        self.cpu_offload_gb = cpu_offload_gb

        self._verify_args()
        self._verify_cache_dtype()
        self._verify_prefix_caching()

        # Will be set after profiling.
        self.num_gpu_blocks: Optional[int] = None
        self.num_cpu_blocks: Optional[int] = None

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
        if (current_platform.is_cuda() and self.block_size is not None
                and self.block_size > 32):
            raise ValueError("CUDA Paged Attention kernel only supports "
                             f"block sizes up to 32. Got {self.block_size}.")

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
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

        if self.sliding_window is not None:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")

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


@dataclass
class TokenizerPoolConfig:
    """Configuration for the tokenizer pool.

    Args:
        pool_size: Number of tokenizer workers in the pool.
        pool_type: Type of the pool.
        extra_config: Additional config for the pool.
            The way the config will be used depends on the
            pool type.
    """
    pool_size: int
    pool_type: Union[str, Type["BaseTokenizerGroup"]]
    extra_config: dict

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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __post_init__(self):
        if self.pool_type not in ("ray", ) and not isinstance(
                self.pool_type, type):
            raise ValueError(f"Unknown pool type: {self.pool_type}")
        if not isinstance(self.extra_config, dict):
            raise ValueError("extra_config must be a dictionary.")

    @classmethod
    def create_config(
        cls, tokenizer_pool_size: int,
        tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]],
        tokenizer_pool_extra_config: Optional[Union[str, dict]]
    ) -> Optional["TokenizerPoolConfig"]:
        """Create a TokenizerPoolConfig from the given parameters.

        If tokenizer_pool_size is 0, return None.

        Args:
            tokenizer_pool_size: Number of tokenizer workers in the pool.
            tokenizer_pool_type: Type of the pool.
            tokenizer_pool_extra_config: Additional config for the pool.
                The way the config will be used depends on the
                pool type. This can be a JSON string (will be parsed).
        """
        if tokenizer_pool_size:
            if isinstance(tokenizer_pool_extra_config, str):
                tokenizer_pool_extra_config_parsed = json.loads(
                    tokenizer_pool_extra_config)
            else:
                tokenizer_pool_extra_config_parsed = (
                    tokenizer_pool_extra_config or {})
            tokenizer_pool_config = cls(tokenizer_pool_size,
                                        tokenizer_pool_type,
                                        tokenizer_pool_extra_config_parsed)
        else:
            tokenizer_pool_config = None
        return tokenizer_pool_config


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


@dataclass
class LoadConfig:
    """
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
            "tensorizer" will use CoreWeave's tensorizer library for
                fast weight loading.
            "bitsandbytes" will load nf4 type weights.
        model_loader_extra_config: The extra config for the model loader.
        ignore_patterns: The list of patterns to ignore when loading the model.
            Default to "original/**/*" to avoid repeated loading of llama's
            checkpoints.
    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(
        default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(
                model_loader_extra_config)
        if isinstance(self.load_format, str):
            load_format = self.load_format.lower()
            self.load_format = LoadFormat(load_format)

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]


@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1  # Number of pipeline parallel groups.
    tensor_parallel_size: int = 1  # Number of tensor parallel groups.

    # Deprecated, use distributed_executor_backend instead.
    worker_use_ray: Optional[bool] = None

    # Maximum number of multiple batches
    # when load model sequentially. To avoid RAM OOM when using tensor
    # parallel and large models.
    max_parallel_loading_workers: Optional[int] = None

    # Disable the custom all-reduce kernel and fall back to NCCL.
    disable_custom_all_reduce: bool = False

    # Config for the tokenizer pool. If None, will use synchronous tokenization.
    tokenizer_pool_config: Optional[TokenizerPoolConfig] = None

    # Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.
    ray_workers_use_nsight: bool = False

    # ray distributed model workers placement group.
    placement_group: Optional["PlacementGroup"] = None

    # Backend to use for distributed model
    # workers, either "ray" or "mp" (multiprocessing). If the product
    # of pipeline_parallel_size and tensor_parallel_size is less than
    # or equal to the number of GPUs available, "mp" will be used to
    # keep processing on a single host. Otherwise, this will default
    # to "ray" if Ray is installed and fail otherwise. Note that tpu
    # and hpu only support Ray for distributed inference.
    distributed_executor_backend: Optional[Union[str,
                                                 Type["ExecutorBase"]]] = None

    # the full name of the worker class to use. If "auto", the worker class
    # will be determined based on the platform.
    worker_cls: str = "auto"
    sd_worker_cls: str = "auto"

    world_size: int = field(init=False)

    rank: int = 0

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: List[Any] = []
        factors.append(self.pipeline_parallel_size)
        factors.append(self.tensor_parallel_size)
        return hashlib.sha256(str(factors).encode()).hexdigest()

    def __post_init__(self) -> None:
        self.world_size = self.pipeline_parallel_size * \
            self.tensor_parallel_size

        if self.worker_use_ray:
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            elif not self.use_ray:
                raise ValueError(f"worker-use-ray can't be used with "
                                 f"distributed executor backend "
                                 f"'{self.distributed_executor_backend}'.")
        ray_only_devices = ["tpu", "hpu"]
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
            backend = "mp"
            ray_found = ray_utils.ray_is_available()
            if (current_platform.is_cuda()
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

        self._verify_args()

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and self.distributed_executor_backend.uses_ray)

    def _verify_args(self) -> None:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase

        if self.distributed_executor_backend not in (
                "ray", "mp", None) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' or custom ExecutorBase subclass.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()
        if current_platform.is_rocm():
            self.disable_custom_all_reduce = True
            logger.info(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on AMD GPUs.")
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    runner_type: str = "generate"  # The runner type to launch for the model.

    # Maximum number of tokens to be processed in a single iteration.
    max_num_batched_tokens: int = field(default=None)  # type: ignore

    # Maximum number of sequences to be processed in a single iteration.
    max_num_seqs: int = 128

    # Maximum length of a sequence (including prompt and generated text).
    max_model_len: int = 8192

    # The number of slots to allocate per sequence per
    # step, beyond the known token ids. This is used in speculative
    # decoding to store KV activations of tokens which may or may not be
    # accepted.
    num_lookahead_slots: int = 0

    # Apply a delay (of delay factor multiplied by previous
    # prompt latency) before scheduling next prompt.
    delay_factor: float = 0.0

    # If True, prefill requests can be chunked based
    # on the remaining max_num_batched_tokens.
    enable_chunked_prefill: bool = False

    is_multimodal_model: bool = False

    # FIXME(woosuk & ywang96): Below are placeholder values. We need to
    # calculate the actual values from the configurations.
    # Multimodal encoder run compute budget, only used in V1
    max_num_encoder_input_tokens = 16384

    # Multimodal encoder cache size, only used in V1
    encoder_cache_size = 16384

    # Whether to perform preemption by swapping or
    # recomputation. If not specified, we determine the mode as follows:
    # We use recomputation by default since it incurs lower overhead than
    # swapping. However, when the sequence group has multiple sequences
    # (e.g., beam search), recomputation is not currently supported. In
    # such a case, we use swapping instead.
    preemption_mode: Optional[str] = None

    num_scheduler_steps: int = 1

    multi_step_stream_outputs: bool = False

    # Private API. If used, scheduler sends delta data to
    # workers instead of an entire data. It should be enabled only
    # when SPMD worker architecture is enabled. I.e.,
    # VLLM_USE_RAY_SPMD_WORKER=1
    send_delta_data: bool = False

    # The scheduling policy to use. "fcfs" (default) or "priority".
    policy: str = "fcfs"

    chunked_prefill_enabled: bool = field(init=False)

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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __post_init__(self) -> None:
        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                if self.num_scheduler_steps > 1:
                    # Multi-step Chunked-Prefill doesn't allow prompt-chunking
                    # for now. Have max_num_batched_tokens set to max_model_len
                    # so we don't reject sequences on account of a short
                    # max_num_batched_tokens.
                    self.max_num_batched_tokens = max(self.max_model_len, 2048)
                else:
                    # This value is chosen to have a balance between ITL
                    # and TTFT. Note it is not optimized for throughput.
                    self.max_num_batched_tokens = 2048
            else:
                # If max_model_len is too short, use 2048 as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(self.max_model_len, 2048)

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

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.chunked_prefill_enabled = self.enable_chunked_prefill
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

    @property
    def is_multi_step(self) -> bool:
        return self.num_scheduler_steps > 1


class DeviceConfig:
    device: Optional[torch.device]
    device_type: str

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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            # Automated device type detection
            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError("Failed to infer device type")
        else:
            # Device type is assigned explicitly
            self.device_type = device

        # Some device types require processing inputs on CPU
        if self.device_type in ["neuron", "openvino"]:
            self.device = torch.device("cpu")
        elif self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)


class SpeculativeConfig:
    """Configuration for speculative decoding.

    The configuration is currently specialized to draft-model speculative
    decoding with top-1 proposals.
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
        # spec decode does not use `torch.compile` yet.
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    @staticmethod
    def maybe_create_spec_config(
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
        target_dtype: str,
        speculative_model: Optional[str],
        speculative_model_quantization: Optional[str],
        speculative_draft_tensor_parallel_size: Optional[int],
        num_speculative_tokens: Optional[int],
        speculative_disable_mqa_scorer: Optional[bool],
        speculative_max_model_len: Optional[int],
        enable_chunked_prefill: bool,
        disable_log_stats: bool,
        speculative_disable_by_batch_size: Optional[int],
        ngram_prompt_lookup_max: Optional[int],
        ngram_prompt_lookup_min: Optional[int],
        draft_token_acceptance_method: str,
        typical_acceptance_sampler_posterior_threshold: Optional[float],
        typical_acceptance_sampler_posterior_alpha: Optional[float],
        disable_logprobs: Optional[bool],
    ) -> Optional["SpeculativeConfig"]:
        """Create a SpeculativeConfig if possible, else return None.

        This function attempts to create a SpeculativeConfig object based on the
        provided parameters. If the necessary conditions are met, it returns an
        instance of SpeculativeConfig. Otherwise, it returns None.

        Args:
            target_model_config (ModelConfig): The configuration of the target
                model.
            target_parallel_config (ParallelConfig): The parallel configuration
                for the target model.
            target_dtype (str): The data type used for the target model.
            speculative_model (Optional[str]): The name of the speculative
                model, if provided.
            speculative_model_quantization (Optional[str]): Quantization method
                that was used to quantize the speculative model weights. If
                None, we assume the model weights are not quantized.
            speculative_draft_tensor_parallel_size (Optional[int]): The degree
                of the tensor parallelism for the draft model.
            num_speculative_tokens (Optional[int]): The number of speculative
                tokens, if provided. Will default to the number in the draft
                model config if present, otherwise is required.
            speculative_disable_mqa_scorer (Optional[bool]): Disable the MQA
                scorer for the speculative model and fall back to batch
                expansion for scoring.
            speculative_max_model_len (Optional[int]): The maximum model len of
                the speculative model. Used when testing the ability to skip
                speculation for some sequences.
            enable_chunked_prefill (bool): Whether vLLM is configured to use
                chunked prefill or not. Used for raising an error since its not
                yet compatible with spec decode.
            speculative_disable_by_batch_size (Optional[int]): Disable
                speculative decoding for new incoming requests when the number
                of enqueue requests  is larger than this value, if provided.
            ngram_prompt_lookup_max (Optional[int]): Max size of ngram token
                window, if provided.
            ngram_prompt_lookup_min (Optional[int]): Min size of ngram token
                window, if provided.
            draft_token_acceptance_method (str): The method to use for
                accepting draft tokens. This can take two possible
                values 'rejection_sampler' and 'typical_acceptance_sampler'
                for RejectionSampler and TypicalAcceptanceSampler
                respectively.
            typical_acceptance_sampler_posterior_threshold (Optional[float]):
                A threshold value that sets a lower bound on the posterior
                probability of a token in the target model for it to be
                accepted. This threshold is used only when we use the
                TypicalAcceptanceSampler for token acceptance.
            typical_acceptance_sampler_posterior_alpha (Optional[float]):
                A scaling factor for the entropy-based threshold in the
                TypicalAcceptanceSampler.
            disable_logprobs (Optional[bool]): If set to True, token log
                probabilities are not returned during speculative decoding.
                If set to False, token log probabilities are returned
                according to the log probability settings in SamplingParams.
                If not specified, it defaults to True.

        Returns:
            Optional["SpeculativeConfig"]: An instance of SpeculativeConfig if
                the necessary conditions are met, else None.
        """

        if speculative_model is None:
            if num_speculative_tokens is not None:
                raise ValueError("num_speculative_tokens was provided without "
                                 "speculative_model.")
            return None

        if (speculative_disable_by_batch_size is not None
                and speculative_disable_by_batch_size < 2):
            raise ValueError("Expect the batch size threshold of disabling "
                             "speculative decoding is > 1, but got "
                             f"{speculative_disable_by_batch_size=}")

        # TODO: The user should be able to specify revision/max model len
        # for the draft model. It is not currently supported.
        draft_revision = None
        draft_code_revision = None
        draft_quantization = speculative_model_quantization

        if speculative_model == "[ngram]":
            if ngram_prompt_lookup_min is None:
                ngram_prompt_lookup_min = 1
            if ngram_prompt_lookup_max is None or ngram_prompt_lookup_max < 1:
                raise ValueError(f"{ngram_prompt_lookup_max=} must be > 0")
            if ngram_prompt_lookup_min < 1:
                raise ValueError(f"{ngram_prompt_lookup_min=} must be > 0")
            if ngram_prompt_lookup_min > ngram_prompt_lookup_max:
                raise ValueError(f"{ngram_prompt_lookup_min=} cannot be "
                                 f"larger than {ngram_prompt_lookup_max=}")

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            draft_model_config = target_model_config
            draft_parallel_config = target_parallel_config
        else:
            ngram_prompt_lookup_max = 0
            ngram_prompt_lookup_min = 0
            draft_model_config = ModelConfig(
                model=speculative_model,
                task="draft",
                tokenizer=target_model_config.tokenizer,
                tokenizer_mode=target_model_config.tokenizer_mode,
                trust_remote_code=target_model_config.trust_remote_code,
                allowed_local_media_path=target_model_config.
                allowed_local_media_path,
                dtype=target_model_config.dtype,
                seed=target_model_config.seed,
                revision=draft_revision,
                code_revision=draft_code_revision,
                tokenizer_revision=target_model_config.tokenizer_revision,
                max_model_len=None,
                spec_target_max_model_len=target_model_config.max_model_len,
                quantization=draft_quantization,
                enforce_eager=target_model_config.enforce_eager,
                max_seq_len_to_capture=target_model_config.
                max_seq_len_to_capture,
                max_logprobs=target_model_config.max_logprobs,
            )

            draft_hf_config = draft_model_config.hf_config

            if (num_speculative_tokens is not None
                    and hasattr(draft_hf_config, "num_lookahead_tokens")):
                draft_hf_config.num_lookahead_tokens = num_speculative_tokens

            n_predict = getattr(draft_hf_config, "n_predict", None)
            if n_predict is not None:
                if num_speculative_tokens is None:
                    # Default to max value defined in draft model config.
                    num_speculative_tokens = n_predict
                elif num_speculative_tokens > n_predict:
                    # Verify provided value doesn't exceed the maximum
                    # supported by the draft model.
                    raise ValueError(
                        "This speculative model supports a maximum of "
                        f"num_speculative_tokens={n_predict}, but "
                        f"{num_speculative_tokens=} was provided.")

            if enable_chunked_prefill and draft_hf_config.model_type in (
                    "medusa", "mlp_speculator", "eagle"):
                raise ValueError(
                    "Chunked prefill and hidden-state based draft models are "
                    "not compatible.")

            speculative_draft_tensor_parallel_size = \
                SpeculativeConfig._verify_and_get_draft_model_tensor_parallel_size(
                    target_parallel_config,
                    speculative_draft_tensor_parallel_size,
                    draft_hf_config
            )

            draft_model_config.max_model_len = (
                SpeculativeConfig._maybe_override_draft_max_model_len(
                    speculative_max_model_len,
                    draft_model_config.max_model_len,
                    target_model_config.max_model_len,
                ))

            draft_parallel_config = (
                SpeculativeConfig.create_draft_parallel_config(
                    target_parallel_config,
                    speculative_draft_tensor_parallel_size, draft_hf_config))

        if num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative_model unless the draft model config contains an "
                "n_predict parameter.")

        if typical_acceptance_sampler_posterior_threshold is None:
            typical_acceptance_sampler_posterior_threshold = 0.09
        if typical_acceptance_sampler_posterior_alpha is None:
            typical_acceptance_sampler_posterior_alpha = 0.3
        if disable_logprobs is None:
            disable_logprobs = True

        return SpeculativeConfig(
            draft_model_config,
            draft_parallel_config,
            num_speculative_tokens,
            speculative_disable_mqa_scorer,
            speculative_disable_by_batch_size,
            ngram_prompt_lookup_max,
            ngram_prompt_lookup_min,
            draft_token_acceptance_method=draft_token_acceptance_method,
            typical_acceptance_sampler_posterior_threshold=\
                typical_acceptance_sampler_posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=\
                typical_acceptance_sampler_posterior_alpha,
            disable_logprobs=disable_logprobs,
            disable_log_stats=disable_log_stats,
        )

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
    def _verify_and_get_draft_model_tensor_parallel_size(
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
                        "MLPSpeculator cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1")
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
        draft_hf_config: PretrainedConfig,
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
            tokenizer_pool_config=target_parallel_config.tokenizer_pool_config,
            ray_workers_use_nsight=target_parallel_config.
            ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    def __init__(
        self,
        draft_model_config: ModelConfig,
        draft_parallel_config: ParallelConfig,
        num_speculative_tokens: int,
        speculative_disable_mqa_scorer: Optional[bool],
        speculative_disable_by_batch_size: Optional[int],
        ngram_prompt_lookup_max: Optional[int],
        ngram_prompt_lookup_min: Optional[int],
        draft_token_acceptance_method: str,
        typical_acceptance_sampler_posterior_threshold: float,
        typical_acceptance_sampler_posterior_alpha: float,
        disable_logprobs: bool,
        disable_log_stats: bool,
    ):
        """Create a SpeculativeConfig object.

        Args:
            draft_model_config: ModelConfig for the draft model.
            draft_parallel_config: ParallelConfig for the draft model.
            num_speculative_tokens: The number of tokens to sample from the
                draft model before scoring with the target model.
            speculative_disable_by_batch_size: Disable speculative
                decoding for new incoming requests when the number of
                enqueue requests is larger than this value.
            ngram_prompt_lookup_max: Max size of ngram token window.
            ngram_prompt_lookup_min: Min size of ngram token window.
            draft_token_acceptance_method (str): The method to use for
                accepting draft tokens. This can take two possible
                values 'rejection_sampler' and 'typical_acceptance_sampler'
                for RejectionSampler and TypicalAcceptanceSampler
                respectively.
            typical_acceptance_sampler_posterior_threshold (Optional[float]):
                A threshold value that sets a lower bound on the posterior
                probability of a token in the target model for it to be
                accepted. This threshold is used only when we use the
                TypicalAcceptanceSampler for token acceptance.
            typical_acceptance_sampler_posterior_alpha (Optional[float]):
                A scaling factor for the entropy-based threshold in the
                TypicalAcceptanceSampler.
            disable_logprobs: If set to True, token log probabilities will not
                be returned even if requested by sampling parameters. This
                reduces latency by skipping logprob calculation in proposal
                sampling, target sampling, and after accepted tokens are
                determined. If set to False, log probabilities will be
                returned.
            disable_log_stats: Whether to disable periodic printing of stage
                times in speculative decoding.
        """
        self.draft_model_config = draft_model_config
        self.draft_parallel_config = draft_parallel_config
        self.num_speculative_tokens = num_speculative_tokens
        self.speculative_disable_mqa_scorer = speculative_disable_mqa_scorer
        self.speculative_disable_by_batch_size = \
            speculative_disable_by_batch_size
        self.ngram_prompt_lookup_max = ngram_prompt_lookup_max or 0
        self.ngram_prompt_lookup_min = ngram_prompt_lookup_min or 0
        self.draft_token_acceptance_method = draft_token_acceptance_method
        self.typical_acceptance_sampler_posterior_threshold = \
            typical_acceptance_sampler_posterior_threshold
        self.typical_acceptance_sampler_posterior_alpha = \
            typical_acceptance_sampler_posterior_alpha
        self.disable_logprobs = disable_logprobs
        self.disable_log_stats = disable_log_stats

        self._verify_args()

    def _verify_args(self) -> None:
        if self.num_speculative_tokens <= 0:
            raise ValueError("Expected num_speculative_tokens to be greater "
                             f"than zero ({self.num_speculative_tokens}).")

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config)
            # Validate and set draft token acceptance related settings.

        if (self.draft_token_acceptance_method is None):
            raise ValueError("draft_token_acceptance_method is not set. "
                             "Expected values are rejection_sampler or "
                             "typical_acceptance_sampler.")

        if (self.draft_token_acceptance_method != 'rejection_sampler'
                and self.draft_token_acceptance_method !=
                'typical_acceptance_sampler'):
            raise ValueError(
                "Expected draft_token_acceptance_method to be either "
                "rejection_sampler or typical_acceptance_sampler. Instead it "
                f"is {self.draft_token_acceptance_method}")

        if (self.typical_acceptance_sampler_posterior_threshold < 0
                or self.typical_acceptance_sampler_posterior_alpha < 0):
            raise ValueError(
                "Expected typical_acceptance_sampler_posterior_threshold "
                "and typical_acceptance_sampler_posterior_alpha to be > 0. "
                "Instead found "
                f"typical_acceptance_sampler_posterior_threshold = "
                f"{self.typical_acceptance_sampler_posterior_threshold} and "
                f"typical_acceptance_sampler_posterior_alpha = "
                f"{self.typical_acceptance_sampler_posterior_alpha}")

    @property
    def num_lookahead_slots(self) -> int:
        """The number of additional slots the scheduler should allocate per
        step, in addition to the slots allocated for each known token.

        This is equal to the number of speculative tokens, as each speculative
        token must be scored.
        """
        return self.num_speculative_tokens

    def __repr__(self) -> str:
        if self.ngram_prompt_lookup_max > 0:
            draft_model = "[ngram]"
        else:
            draft_model = self.draft_model_config.model
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({draft_model=}, {num_spec_tokens=})"


@dataclass
class LoRAConfig:
    max_lora_rank: int
    max_loras: int
    fully_sharded_loras: bool = False
    max_cpu_loras: Optional[int] = None
    lora_dtype: Optional[Union[torch.dtype, str]] = None
    lora_extra_vocab_size: int = 256
    # This is a constant.
    lora_vocab_padding_size: ClassVar[int] = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    bias_enabled: bool = False

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
        # LoRA is not compatible with `torch.compile` .
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __post_init__(self):
        # Setting the maximum rank to 256 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256)
        possible_lora_extra_vocab_size = (0, 256, 512)
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

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
        if model_config.quantization and model_config.quantization not in [
                "awq", "gptq"
        ]:
            # TODO support marlin
            logger.warning("%s quantization is not tested with LoRA yet.",
                           model_config.quantization)

    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        # Reminder: Please update docs/source/usage/compatibility_matrix.rst
        # If the feature combo become valid
        if scheduler_config.chunked_prefill_enabled:
            logger.warning("LoRA with chunked prefill is still experimental "
                           "and may be unstable.")


@dataclass
class PromptAdapterConfig:
    max_prompt_adapters: int
    max_prompt_adapter_token: int
    max_cpu_prompt_adapters: Optional[int] = None
    prompt_adapter_dtype: Optional[torch.dtype] = None

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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
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
        if self.prompt_adapter_dtype in (None, "auto"):
            self.prompt_adapter_dtype = model_config.dtype
        elif isinstance(self.prompt_adapter_dtype, str):
            self.prompt_adapter_dtype = getattr(torch,
                                                self.prompt_adapter_dtype)


@dataclass
class MultiModalConfig:
    """Controls the behavior of multimodal models."""

    limit_per_prompt: Mapping[str, int] = field(default_factory=dict)
    """
    The maximum number of multi-modal input instances allowed per prompt
    for each :class:`~vllm.multimodal.MultiModalPlugin`.
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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    # TODO: Add configs to init vision tower or not.


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

    returned_token_ids: Optional[List[int]] = None
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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
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

_ROCM_NOT_SUPPORTED_DTYPE: List[str] = []  #


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                if config.model_type == "gemma2":
                    logger.info(
                        "For Gemma 2, we downcast float32 to bfloat16 instead "
                        "of float16 by default. Please specify `dtype` if you "
                        "want to use float16.")
                    torch_dtype = torch.bfloat16
                else:
                    # Following the common practice, we use float16 for float32
                    # models.
                    torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype

            if (current_platform.is_cpu()
                    and current_platform.get_cpu_architecture()
                    == interface.CpuArchEnum.POWERPC
                    and (config_dtype == torch.float16
                         or config_dtype == torch.float32)):
                logger.info(
                    "For POWERPC, we cast models to bfloat16 instead of "
                    "using float16 by default. Float16 is not currently "
                    "supported for POWERPC.")
                torch_dtype = torch.bfloat16

            if current_platform.is_hpu() and config_dtype == torch.float16:
                logger.info(
                    "For HPU, we cast models to bfloat16 instead of"
                    "using float16 by default. Please specify `dtype` if you "
                    "want to use float16.")
                torch_dtype = torch.bfloat16
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
    sliding_window_len: Optional[Union[int, List[Optional[int]]]],
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
    if rope_scaling is not None:
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
        sliding_window: Union[int, List[Optional[int]]]) -> int:
    if isinstance(sliding_window, list):
        return min(s for s in sliding_window if s is not None)

    return sliding_window


def get_served_model_name(model: str,
                          served_model_name: Optional[Union[str, List[str]]]):
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


@dataclass
class DecodingConfig:
    """Dataclass which contains the decoding strategy of the engine"""

    # Which guided decoding algo to use.
    # 'outlines' / 'lm-format-enforcer' / 'xgrammar'
    guided_decoding_backend: str = 'xgrammar'

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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __post_init__(self):
        valid_guided_backends = ['outlines', 'lm-format-enforcer', 'xgrammar']
        backend = self.guided_decoding_backend
        if backend not in valid_guided_backends:
            raise ValueError(f"Invalid guided_decoding_backend '{backend},"
                             f"must be one of {valid_guided_backends}")


@dataclass
class ObservabilityConfig:
    """Configuration for observability."""
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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
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
        factors: List[Any] = []
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    @classmethod
    def from_cli(cls, cli_value: str) -> "KVTransferConfig":
        """Parse the CLI value for the kv cache transfer config."""
        return KVTransferConfig.model_validate_json(cli_value)

    def model_post_init(self, __context: Any) -> None:
        supported_kv_connector = ["PyNcclConnector", "MooncakeConnector"]
        if all([
                self.kv_connector is not None, self.kv_connector
                not in supported_kv_connector
        ]):
            raise ValueError(f"Unsupported kv_connector: {self.kv_connector}. "
                             f"Supported connectors are "
                             f"{supported_kv_connector}.")

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
    def need_kv_parallel_group(self) -> bool:
        # for those database-based connector, vLLM does not need to create
        # parallel group, and in that case the kv parallel size will be 1.
        return self.kv_connector is not None and self.kv_parallel_size > 1

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in ["kv_producer", "kv_both"]

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_connector is not None and \
            self.kv_role in ["kv_consumer", "kv_both"]


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
            - List[int]: capture sizes are specified as given.
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
                is compiled. In addition, compile for cudagraph sizes that are
                in candidate_compile_sizes, using configurations
                in inductor_compile_config.
        - candidate_compile_sizes: sizes to compile for inductor.
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
    custom_ops: List[str] = Field(default_factory=list)
    splitting_ops: List[str] = Field(default=None)  # type: ignore

    use_inductor: bool = True
    candidate_compile_sizes: Optional[List[int]] = Field(default=None)
    inductor_compile_config: Dict = Field(default_factory=dict)
    inductor_passes: Dict[str, str] = Field(default_factory=dict)

    use_cudagraph: bool = False
    cudagraph_num_of_warmups: int = 0
    cudagraph_capture_sizes: Optional[List[int]] = None
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
        - enable_reshape: whether to enable the custom reshape elimination pass.
            TODO better pass enabling system.
        """
        dump_graph_stages: List[str] = Field(default_factory=list)
        dump_graph_dir: Path = Field(default=Path("."))
        enable_fusion: bool = True
        enable_reshape: bool = True

        def uuid(self):
            """
            Produces a hash unique to the pass configuration.
            Any new fields that affect compilation should be added to the hash.
            Do not include dump_graph_* in the hash - they don't affect
            compilation.
            """
            dict_ = self.model_dump(
                include={"enable_fusion", "enable_reshape"})
            encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
            return hashlib.sha256(encoded).digest()

        def model_post_init(self, __context: Any) -> None:
            if not self.enable_reshape and self.enable_fusion:
                print_warning_once(
                    "Fusion enabled but reshape elimination disabled."
                    "RMSNorm + quant (fp8) fusion might not work")

    pass_config: PassConfig = Field(default_factory=PassConfig)

    # not configurable, computed after init
    compile_sizes: List[int] = PrivateAttr
    capture_sizes: List[int] = PrivateAttr
    max_capture_size: int = PrivateAttr
    # optimization:
    # Intuitively, bs_to_padded_graph_size should be Dict[int, int].
    # since we know all keys are in a range [0, max_capture_size],
    # we can optimize it to List[int] for better lookup performance.
    bs_to_padded_graph_size: List[int] = PrivateAttr

    # keep track of enabled and disabled custom ops
    enabled_custom_ops: Counter[str] = PrivateAttr
    disabled_custom_ops: Counter[str] = PrivateAttr
    compilation_time: float = PrivateAttr
    # should be InductorHashCache, but Pydantic does not support it
    inductor_hash_cache: Any = PrivateAttr

    # Per-model forward context
    # Mainly used to store attention cls
    # Map from layer name to the attention cls
    static_forward_context: Dict[str, Any] = PrivateAttr

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
        factors: List[Any] = []
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

        if self.splitting_ops is None:
            if envs.VLLM_USE_V1:
                # v1 must split the graph on attention ops
                # for piecewise cudagraph
                self.splitting_ops = [
                    "vllm.unified_attention",
                    "vllm.unified_attention_with_output",
                ]
            else:
                # v0 can use full graph compilation without splitting,
                # splitting is optional.
                # right now we still need it. kv cache shape
                # will be included in the graph if we don't split
                # the graph.
                # TODO: hide kv cache in static forward context
                # so that inductor does not see it.
                self.splitting_ops = [
                    "vllm.unified_attention",
                    "vllm.unified_attention_with_output",
                ]

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

        if not self.cache_dir:
            # no provided cache dir, generate one based on the known factors
            # that affects the compilation. if none of the factors change,
            # the cache dir will be the same so that we can reuse the compiled
            # graph.
            hash_key = vllm_config.compute_hash()
            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT, "torch_compile_cache", hash_key,
                f"rank_{vllm_config.parallel_config.rank}")
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir

            disabled = envs.VLLM_DISABLE_COMPILE_CACHE
            from vllm.compilation.backends import InductorHashCache
            self.inductor_hash_cache: InductorHashCache = InductorHashCache(
                self.cache_dir, disabled=disabled)
            if disabled:
                logger.info("vLLM's torch.compile cache is disabled.")
            else:
                logger.info(
                    "Using cache directory: %s for vLLM's torch.compile",
                    self.cache_dir)

        from vllm.compilation.backends import VllmBackend
        return VllmBackend(vllm_config)

    def init_with_cudagraph_sizes(self, sizes_to_specialize: List[int]):
        """To complete the initialization of config,
        we need to know the cudagraph sizes."""

        if self.cudagraph_capture_sizes is None:
            self.capture_sizes = sizes_to_specialize
        else:
            self.capture_sizes = self.cudagraph_capture_sizes
            logger.info(("cudagraph sizes specified by model runner"
                         " %s is overridden by config %s"),
                        sizes_to_specialize, self.cudagraph_capture_sizes)

        if self.candidate_compile_sizes is None:
            self.candidate_compile_sizes = []
        self.compile_sizes = [
            x for x in self.candidate_compile_sizes if x in self.capture_sizes
        ]
        ignored_sizes = [
            x for x in self.candidate_compile_sizes
            if x not in self.capture_sizes
        ]
        if ignored_sizes:
            logger.warning(("candidate_compile_sizes %s are ignored "
                            "because they are not cudagraph capture sizes."),
                           ignored_sizes)

        # sort to make sure cudagraph capture sizes are in descending order
        self.capture_sizes.sort(reverse=True)
        self.max_capture_size = self.capture_sizes[
            0] if self.capture_sizes else 0

        # pre-compute the mapping from batch size to padded graph size
        self.bs_to_padded_graph_size = [
            0 for i in range(self.max_capture_size + 1)
        ]
        for end, start in zip(self.capture_sizes,
                              self.capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.bs_to_padded_graph_size[bs] = start
                else:
                    self.bs_to_padded_graph_size[bs] = end
        self.bs_to_padded_graph_size[
            self.max_capture_size] = self.max_capture_size


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
    speculative_config: Optional[SpeculativeConfig] = None
    decoding_config: Optional[DecodingConfig] = None
    observability_config: Optional[ObservabilityConfig] = None
    prompt_adapter_config: Optional[PromptAdapterConfig] = None
    quant_config: Optional[QuantizationConfig] = None
    compilation_config: CompilationConfig = field(default=None,
                                                  init=True)  # type: ignore
    kv_transfer_config: KVTransferConfig = field(default=None,
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
        factors: List[Any] = []
        # summarize system state
        from torch._inductor.codecache import CacheBase
        system_factors = CacheBase.get_system()
        factors.append(system_factors)

        # summarize pytorch state
        from torch._inductor.codecache import torch_key
        torch_factors = torch_key()
        factors.append(torch_factors)

        # summarize vllm config
        vllm_factors: List[Any] = []
        from vllm import __version__
        vllm_factors.append(__version__)
        if self.model_config:
            vllm_factors.append(self.model_config.compute_hash())
        if self.cache_config:
            vllm_factors.append(self.cache_config.compute_hash())
        if self.parallel_config:
            vllm_factors.append(self.parallel_config.compute_hash())
        if self.scheduler_config:
            vllm_factors.append(self.scheduler_config.compute_hash())
        if self.device_config:
            vllm_factors.append(self.device_config.compute_hash())
        if self.load_config:
            vllm_factors.append(self.load_config.compute_hash())
        if self.lora_config:
            vllm_factors.append(self.lora_config.compute_hash())
        if self.speculative_config:
            vllm_factors.append(self.speculative_config.compute_hash())
        if self.decoding_config:
            vllm_factors.append(self.decoding_config.compute_hash())
        if self.observability_config:
            vllm_factors.append(self.observability_config.compute_hash())
        if self.prompt_adapter_config:
            vllm_factors.append(self.prompt_adapter_config.compute_hash())
        if self.quant_config:
            pass  # should be captured by model_config.quantization
        if self.compilation_config:
            vllm_factors.append(self.compilation_config.compute_hash())
        if self.kv_transfer_config:
            vllm_factors.append(self.kv_transfer_config.compute_hash())

        factors.append(vllm_factors)

        hash_str = hashlib.md5(str(factors).encode()).hexdigest()[:10]
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
            self.lora_config.verify_with_model_config(self.model_config)
            self.lora_config.verify_with_scheduler_config(
                self.scheduler_config)
        if self.prompt_adapter_config:
            self.prompt_adapter_config.verify_with_model_config(
                self.model_config)

        if self.quant_config is None and \
            self.model_config is not None and self.load_config is not None:
            self.quant_config = VllmConfig._get_quantization_config(
                self.model_config, self.load_config)

        if self.scheduler_config is not None and \
            self.model_config is not None and \
            self.scheduler_config.chunked_prefill_enabled and \
            self.model_config.dtype == torch.float32 and \
            current_platform.get_device_capability() == (7, 5):
            print_warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels.")

        if self.compilation_config is None:
            self.compilation_config = CompilationConfig()
        if envs.VLLM_USE_V1 and not self.model_config.enforce_eager:
            # NOTE(woosuk): Currently, we use inductor because the piecewise
            # CUDA graphs do not work properly with the custom CUDA kernels.
            # FIXME(woosuk): Disable inductor to reduce the compilation time
            # and avoid any potential issues with the inductor.
            self.compilation_config.custom_ops = ["none"]
            self.compilation_config.use_cudagraph = True
            self.compilation_config.use_inductor = True
            self.compilation_config.cudagraph_num_of_warmups = 1
            self.compilation_config.pass_config.enable_fusion = False
            self.compilation_config.pass_config.enable_reshape = False
            self.compilation_config.level = CompilationLevel.PIECEWISE

        self._set_cudagraph_sizes()

        if self.cache_config is not None and \
            self.cache_config.cpu_offload_gb > 0 and \
            self.compilation_config.level != CompilationLevel.NO_COMPILATION:
            logger.warning(
                "CPU offload is not supported with `torch.compile` yet."
                " Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        if self.lora_config is not None and self.compilation_config.level !=\
             CompilationLevel.NO_COMPILATION:
            logger.warning("LoRA is not supported with `torch.compile` yet. "
                           "Disabling `torch.compile`.")
            self.compilation_config.level = CompilationLevel.NO_COMPILATION

        current_platform.check_and_update_config(self)

        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

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

        In the end, `vllm_config.compilation_config.capture_sizes` will be the
        final sizes to capture cudagraph (in descending order).

        During runtime, if batchsize is larger than
        `vllm_config.compilation_config.capture_sizes`,
        no cudagraph will be used.
        If the batch size is no larger than
        `vllm_config.compilation_config.capture_sizes`,
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
            f"quantization_param_path={self.model_config.quantization_param_path},"
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
def set_current_vllm_config(vllm_config: VllmConfig):
    """
    Temporarily set the current VLLM config.
    Used during model initialization.
    We save the current VLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the VLLM config to determine how to dispatch.
    """
    global _current_vllm_config
    old_vllm_config = _current_vllm_config
    from vllm.compilation.counter import compilation_counter
    num_models_seen = compilation_counter.num_models_seen
    try:
        _current_vllm_config = vllm_config
        yield
    finally:
        logger.debug("enabled custom ops: %s",
                     vllm_config.compilation_config.enabled_custom_ops)
        logger.debug("disabled custom ops: %s",
                     vllm_config.compilation_config.disabled_custom_ops)
        if vllm_config.compilation_config.level == CompilationLevel.PIECEWISE \
            and compilation_counter.num_models_seen == num_models_seen:
            # If the model supports compilation,
            # compilation_counter.num_models_seen should be increased
            # by at least 1.
            # If it is not increased, it means the model does not support
            # compilation (does not have @support_torch_compile decorator).
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                "if you want it to be supported.",
                vllm_config.model_config.model)
        _current_vllm_config = old_vllm_config


def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the vllm config. In that case, we set a default
        # config.
        logger.warning("Current VLLM config is not set.")
        from vllm.config import VllmConfig
        return VllmConfig()
    return _current_vllm_config
