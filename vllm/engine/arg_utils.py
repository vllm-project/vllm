# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import dataclasses
import functools
import json
import sys
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields, is_dataclass
from itertools import permutations
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import huggingface_hub
import regex as re
import torch
from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import TypeIs

import vllm.envs as envs
from vllm.config import (
    AFDConfig,
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    ConfigType,
    DeviceConfig,
    ECTransferConfig,
    EPLBConfig,
    FaultToleranceConfig,
    KVEventsConfig,
    KVTransferConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    MultiModalConfig,
    ObservabilityConfig,
    ParallelConfig,
    PoolerConfig,
    ProfilerConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
    VllmConfig,
    WeightTransferConfig,
    get_attr_docs,
)
from vllm.config.cache import (
    BlockSize,
    CacheDType,
    KVOffloadingBackend,
    MambaCacheMode,
    MambaDType,
    PrefixCachingHashAlgo,
)
from vllm.config.device import Device
from vllm.config.model import (
    ConvertOption,
    HfOverrides,
    LogprobsMode,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode, MultiModalConfig
from vllm.config.observability import DetailedTraceModules
from vllm.config.parallel import DistributedExecutorBackend, ExpertPlacementStrategy
from vllm.config.scheduler import SchedulerPolicy
from vllm.config.utils import get_field
from vllm.config.vllm import OptimizationLevel
from vllm.logger import init_logger, suppress_logging
from vllm.platforms import CpuArchEnum, current_platform
from vllm.plugins import load_general_plugins
from vllm.ray.lazy_utils import is_in_ray_actor, is_ray_initialized
from vllm.transformers_utils.config import (
    is_interleaved,
    maybe_override_with_speculators,
)
from vllm.transformers_utils.gguf_utils import is_gguf
from vllm.transformers_utils.repo_utils import get_model_path
from vllm.transformers_utils.utils import is_cloud_storage
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.sample.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.model_loader import LoadFormats
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.executor import Executor
else:
    Executor = Any
    QuantizationMethods = Any
    LoadFormats = Any
    UsageContext = Any


logger = init_logger(__name__)

# object is used to allow for special typing forms
T = TypeVar("T")
TypeHint: TypeAlias = type[Any] | object
TypeHintT: TypeAlias = type[T] | object


def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:
    def _parse_type(val: str) -> T:
        try:
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to {return_type}."
            ) from e

    return _parse_type


def optional_type(return_type: Callable[[str], T]) -> Callable[[str], T | None]:
    def _optional_type(val: str) -> T | None:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)

    return _optional_type


def union_dict_and_str(val: str) -> str | dict[str, str] | None:
    if not re.match(r"(?s)^\s*{.*}\s*$", val):
        return str(val)
    return optional_type(json.loads)(val)


def is_type(type_hint: TypeHint, type: TypeHintT) -> TypeIs[TypeHintT]:
    """Check if the type hint is a specific type."""
    return type_hint is type or get_origin(type_hint) is type


def contains_type(type_hints: set[TypeHint], type: TypeHintT) -> bool:
    """Check if the type hints contain a specific type."""
    return any(is_type(type_hint, type) for type_hint in type_hints)


def get_type(type_hints: set[TypeHint], type: TypeHintT) -> TypeHintT:
    """Get the specific type from the type hints."""
    return next((th for th in type_hints if is_type(th, type)), None)


def literal_to_kwargs(type_hints: set[TypeHint]) -> dict[str, Any]:
    """Get the `type` and `choices` from a `Literal` type hint in `type_hints`.

    If `type_hints` also contains `str`, we use `metavar` instead of `choices`.
    """
    type_hint = get_type(type_hints, Literal)
    options = get_args(type_hint)
    option_type = type(options[0])
    if not all(isinstance(option, option_type) for option in options):
        raise ValueError(
            "All options must be of the same type. "
            f"Got {options} with types {[type(c) for c in options]}"
        )
    kwarg = "metavar" if contains_type(type_hints, str) else "choices"
    return {"type": option_type, kwarg: sorted(options)}


def collection_to_kwargs(type_hints: set[TypeHint], type: TypeHint) -> dict[str, Any]:
    type_hint = get_type(type_hints, type)
    types = get_args(type_hint)
    elem_type = types[0]

    # Handle Ellipsis
    assert all(t is elem_type for t in types if t is not Ellipsis), (
        f"All non-Ellipsis elements must be of the same type. Got {types}."
    )

    # Handle Union types
    if get_origin(elem_type) in {Union, UnionType}:
        # Union for Union[X, Y] and UnionType for X | Y
        assert str in get_args(elem_type), (
            "If element can have multiple types, one must be 'str' "
            f"(i.e. 'list[int | str]'). Got {elem_type}."
        )
        elem_type = str

    return {
        "type": elem_type,
        "nargs": "+" if type is not tuple or Ellipsis in types else len(types),
    }


def is_not_builtin(type_hint: TypeHint) -> bool:
    """Check if the class is not a built-in type."""
    return type_hint.__module__ != "builtins"


def get_type_hints(type_hint: TypeHint) -> set[TypeHint]:
    """Extract type hints from Annotated or Union type hints."""
    type_hints: set[TypeHint] = set()
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Annotated:
        type_hints.update(get_type_hints(args[0]))
    elif origin in {Union, UnionType}:
        # Union for Union[X, Y] and UnionType for X | Y
        for arg in args:
            type_hints.update(get_type_hints(arg))
    else:
        type_hints.add(type_hint)

    return type_hints


NEEDS_HELP = (
    any("--help" in arg for arg in sys.argv)  # vllm SUBCOMMAND --help
    or (argv0 := sys.argv[0]).endswith("mkdocs")  # mkdocs SUBCOMMAND
    or argv0.endswith("mkdocs/__main__.py")  # python -m mkdocs SUBCOMMAND
)


@functools.lru_cache(maxsize=30)
def _compute_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    # Save time only getting attr docs if we're generating help text
    cls_docs = get_attr_docs(cls) if NEEDS_HELP else {}
    kwargs = {}
    for field in fields(cls):
        # Get the set of possible types for the field
        type_hints: set[TypeHint] = get_type_hints(field.type)

        # If the field is a dataclass, we can use the model_validate_json
        generator = (th for th in type_hints if is_dataclass(th))
        dataclass_cls = next(generator, None)

        # Get the default value of the field
        if field.default is not MISSING:
            default = field.default
            # Handle pydantic.Field defaults
            if isinstance(default, FieldInfo):
                if default.default_factory is None:
                    default = default.default
                else:
                    # VllmConfig's Fields have default_factory set to config classes.
                    # These could emit logs on init, which would be confusing.
                    with suppress_logging():
                        default = default.default_factory()
        elif field.default_factory is not MISSING:
            default = field.default_factory()

        # Get the help text for the field
        name = field.name
        help = cls_docs.get(name, "").strip()
        # Escape % for argparse
        help = help.replace("%", "%%")

        # Initialise the kwargs dictionary for the field
        kwargs[name] = {"default": default, "help": help}

        # Set other kwargs based on the type hints
        json_tip = (
            "Should either be a valid JSON string or JSON keys passed individually."
        )
        if dataclass_cls is not None:

            def parse_dataclass(val: str, cls=dataclass_cls) -> Any:
                try:
                    return TypeAdapter(cls).validate_json(val)
                except ValidationError as e:
                    raise argparse.ArgumentTypeError(repr(e)) from e

            kwargs[name]["type"] = parse_dataclass
            kwargs[name]["help"] += f"\n\n{json_tip}"
        elif contains_type(type_hints, bool):
            # Creates --no-<name> and --<name> flags
            kwargs[name]["action"] = argparse.BooleanOptionalAction
        elif contains_type(type_hints, Literal):
            kwargs[name].update(literal_to_kwargs(type_hints))
        elif contains_type(type_hints, tuple):
            kwargs[name].update(collection_to_kwargs(type_hints, tuple))
        elif contains_type(type_hints, list):
            kwargs[name].update(collection_to_kwargs(type_hints, list))
        elif contains_type(type_hints, set):
            kwargs[name].update(collection_to_kwargs(type_hints, set))
        elif contains_type(type_hints, int):
            if name == "max_model_len":
                kwargs[name]["type"] = human_readable_int_or_auto
                kwargs[name]["help"] += f"\n\n{human_readable_int_or_auto.__doc__}"
            elif name in ("max_num_batched_tokens", "kv_cache_memory_bytes"):
                kwargs[name]["type"] = human_readable_int
                kwargs[name]["help"] += f"\n\n{human_readable_int.__doc__}"
            else:
                kwargs[name]["type"] = int
        elif contains_type(type_hints, float):
            kwargs[name]["type"] = float
        elif contains_type(type_hints, dict) and (
            contains_type(type_hints, str)
            or any(is_not_builtin(th) for th in type_hints)
        ):
            kwargs[name]["type"] = union_dict_and_str
        elif contains_type(type_hints, dict):
            kwargs[name]["type"] = parse_type(json.loads)
            kwargs[name]["help"] += f"\n\n{json_tip}"
        elif contains_type(type_hints, str) or any(
            is_not_builtin(th) for th in type_hints
        ):
            kwargs[name]["type"] = str
        else:
            raise ValueError(f"Unsupported type {type_hints} for argument {name}.")

        # If the type hint was a sequence of literals, use the helper function
        # to update the type and choices
        if get_origin(kwargs[name].get("type")) is Literal:
            kwargs[name].update(literal_to_kwargs({kwargs[name]["type"]}))

        # If None is in type_hints, make the argument optional.
        # But not if it's a bool, argparse will handle this better.
        if type(None) in type_hints and not contains_type(type_hints, bool):
            kwargs[name]["type"] = optional_type(kwargs[name]["type"])
            if kwargs[name].get("choices"):
                kwargs[name]["choices"].append("None")
    return kwargs


def get_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    """Return argparse kwargs for the given Config dataclass.

    If `--help` or `mkdocs` are not present in the command line command, the
    attribute documentation will not be included in the help output.

    The heavy computation is cached via functools.lru_cache, and a deep copy
    is returned so callers can mutate the dictionary without affecting the
    cached version.
    """
    return copy.deepcopy(_compute_kwargs(cls))


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""

    model: str = ModelConfig.model
    enable_return_routed_experts: bool = ModelConfig.enable_return_routed_experts
    model_weights: str = ModelConfig.model_weights
    served_model_name: str | list[str] | None = ModelConfig.served_model_name
    tokenizer: str | None = ModelConfig.tokenizer
    hf_config_path: str | None = ModelConfig.hf_config_path
    runner: RunnerOption = ModelConfig.runner
    convert: ConvertOption = ModelConfig.convert
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds
    tokenizer_mode: TokenizerMode | str = ModelConfig.tokenizer_mode
    trust_remote_code: bool = ModelConfig.trust_remote_code
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path
    allowed_media_domains: list[str] | None = ModelConfig.allowed_media_domains
    download_dir: str | None = LoadConfig.download_dir
    safetensors_load_strategy: str = LoadConfig.safetensors_load_strategy
    load_format: str | LoadFormats = LoadConfig.load_format
    config_format: str = ModelConfig.config_format
    dtype: ModelDType = ModelConfig.dtype
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    seed: int = ModelConfig.seed
    max_model_len: int | None = ModelConfig.max_model_len
    cudagraph_capture_sizes: list[int] | None = (
        CompilationConfig.cudagraph_capture_sizes
    )
    max_cudagraph_capture_size: int | None = get_field(
        CompilationConfig, "max_cudagraph_capture_size"
    )
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = ParallelConfig.distributed_executor_backend
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    master_addr: str = ParallelConfig.master_addr
    master_port: int = ParallelConfig.master_port
    nnodes: int = ParallelConfig.nnodes
    node_rank: int = ParallelConfig.node_rank
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    prefill_context_parallel_size: int = ParallelConfig.prefill_context_parallel_size
    decode_context_parallel_size: int = ParallelConfig.decode_context_parallel_size
    dcp_kv_cache_interleave_size: int = ParallelConfig.dcp_kv_cache_interleave_size
    cp_kv_cache_interleave_size: int = ParallelConfig.cp_kv_cache_interleave_size
    data_parallel_size: int = ParallelConfig.data_parallel_size
    data_parallel_rank: int | None = None
    data_parallel_start_rank: int | None = None
    data_parallel_size_local: int | None = None
    data_parallel_address: str | None = None
    data_parallel_rpc_port: int | None = None
    data_parallel_hybrid_lb: bool = False
    data_parallel_external_lb: bool = False
    data_parallel_backend: str = ParallelConfig.data_parallel_backend
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    all2all_backend: str = ParallelConfig.all2all_backend
    enable_dbo: bool = ParallelConfig.enable_dbo
    ubatch_size: int = ParallelConfig.ubatch_size
    dbo_decode_token_threshold: int = ParallelConfig.dbo_decode_token_threshold
    dbo_prefill_token_threshold: int = ParallelConfig.dbo_prefill_token_threshold
    disable_nccl_for_dp_synchronization: bool | None = (
        ParallelConfig.disable_nccl_for_dp_synchronization
    )
    eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")
    enable_eplb: bool = ParallelConfig.enable_eplb
    expert_placement_strategy: ExpertPlacementStrategy = (
        ParallelConfig.expert_placement_strategy
    )
    _api_process_count: int = ParallelConfig._api_process_count
    _api_process_rank: int = ParallelConfig._api_process_rank
    max_parallel_loading_workers: int | None = (
        ParallelConfig.max_parallel_loading_workers
    )
    block_size: BlockSize | None = CacheConfig.block_size
    enable_prefix_caching: bool | None = None
    prefix_caching_hash_algo: PrefixCachingHashAlgo = (
        CacheConfig.prefix_caching_hash_algo
    )
    disable_sliding_window: bool = ModelConfig.disable_sliding_window
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn
    swap_space: float = CacheConfig.swap_space
    cpu_offload_gb: float = CacheConfig.cpu_offload_gb
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    kv_cache_memory_bytes: int | None = CacheConfig.kv_cache_memory_bytes
    max_num_batched_tokens: int | None = None
    max_num_partial_prefills: int = SchedulerConfig.max_num_partial_prefills
    max_long_partial_prefills: int = SchedulerConfig.max_long_partial_prefills
    long_prefill_token_threshold: int = SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: int | None = None
    max_logprobs: int = ModelConfig.max_logprobs
    logprobs_mode: LogprobsMode = ModelConfig.logprobs_mode
    disable_log_stats: bool = False
    aggregate_engine_logging: bool = False
    revision: str | None = ModelConfig.revision
    code_revision: str | None = ModelConfig.code_revision
    hf_token: bool | str | None = ModelConfig.hf_token
    hf_overrides: HfOverrides = get_field(ModelConfig, "hf_overrides")
    tokenizer_revision: str | None = ModelConfig.tokenizer_revision
    quantization: QuantizationMethods | None = ModelConfig.quantization
    allow_deprecated_quantization: bool = ModelConfig.allow_deprecated_quantization
    enforce_eager: bool = ModelConfig.enforce_eager
    disable_custom_all_reduce: bool = ParallelConfig.disable_custom_all_reduce
    limit_mm_per_prompt: dict[str, int | dict[str, int]] = get_field(
        MultiModalConfig, "limit_per_prompt"
    )
    enable_mm_embeds: bool = MultiModalConfig.enable_mm_embeds
    interleave_mm_strings: bool = MultiModalConfig.interleave_mm_strings
    media_io_kwargs: dict[str, dict[str, Any]] = get_field(
        MultiModalConfig, "media_io_kwargs"
    )
    mm_processor_kwargs: dict[str, Any] | None = MultiModalConfig.mm_processor_kwargs
    mm_processor_cache_gb: float = MultiModalConfig.mm_processor_cache_gb
    mm_processor_cache_type: MMCacheType | None = (
        MultiModalConfig.mm_processor_cache_type
    )
    mm_shm_cache_max_object_size_mb: int = (
        MultiModalConfig.mm_shm_cache_max_object_size_mb
    )
    mm_encoder_only: bool = MultiModalConfig.mm_encoder_only
    mm_encoder_tp_mode: MMEncoderTPMode = MultiModalConfig.mm_encoder_tp_mode
    mm_encoder_attn_backend: AttentionBackendEnum | str | None = (
        MultiModalConfig.mm_encoder_attn_backend
    )
    io_processor_plugin: str | None = None
    skip_mm_profiling: bool = MultiModalConfig.skip_mm_profiling
    video_pruning_rate: float = MultiModalConfig.video_pruning_rate
    # LoRA fields
    enable_lora: bool = False
    max_loras: int = LoRAConfig.max_loras
    max_lora_rank: int = LoRAConfig.max_lora_rank
    default_mm_loras: dict[str, str] | None = LoRAConfig.default_mm_loras
    fully_sharded_loras: bool = LoRAConfig.fully_sharded_loras
    max_cpu_loras: int | None = LoRAConfig.max_cpu_loras
    lora_dtype: str | torch.dtype | None = LoRAConfig.lora_dtype
    enable_tower_connector_lora: bool = LoRAConfig.enable_tower_connector_lora
    specialize_active_lora: bool = LoRAConfig.specialize_active_lora

    ray_workers_use_nsight: bool = ParallelConfig.ray_workers_use_nsight
    num_gpu_blocks_override: int | None = CacheConfig.num_gpu_blocks_override
    model_loader_extra_config: dict = get_field(LoadConfig, "model_loader_extra_config")
    ignore_patterns: str | list[str] = get_field(LoadConfig, "ignore_patterns")

    enable_chunked_prefill: bool | None = None
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input

    disable_hybrid_kv_cache_manager: bool | None = (
        SchedulerConfig.disable_hybrid_kv_cache_manager
    )

    structured_outputs_config: StructuredOutputsConfig = get_field(
        VllmConfig, "structured_outputs_config"
    )
    reasoning_parser: str = StructuredOutputsConfig.reasoning_parser
    reasoning_parser_plugin: str | None = None

    logits_processor_pattern: str | None = ModelConfig.logits_processor_pattern

    speculative_config: dict[str, Any] | None = None

    show_hidden_metrics_for_version: str | None = (
        ObservabilityConfig.show_hidden_metrics_for_version
    )
    otlp_traces_endpoint: str | None = ObservabilityConfig.otlp_traces_endpoint
    collect_detailed_traces: list[DetailedTraceModules] | None = (
        ObservabilityConfig.collect_detailed_traces
    )
    kv_cache_metrics: bool = ObservabilityConfig.kv_cache_metrics
    kv_cache_metrics_sample: float = get_field(
        ObservabilityConfig, "kv_cache_metrics_sample"
    )
    cudagraph_metrics: bool = ObservabilityConfig.cudagraph_metrics
    enable_layerwise_nvtx_tracing: bool = (
        ObservabilityConfig.enable_layerwise_nvtx_tracing
    )
    enable_mfu_metrics: bool = ObservabilityConfig.enable_mfu_metrics
    enable_logging_iteration_details: bool = (
        ObservabilityConfig.enable_logging_iteration_details
    )
    enable_mm_processor_stats: bool = ObservabilityConfig.enable_mm_processor_stats
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
    scheduler_cls: str | type[object] | None = SchedulerConfig.scheduler_cls

    pooler_config: PoolerConfig | None = ModelConfig.pooler_config
    compilation_config: CompilationConfig = get_field(VllmConfig, "compilation_config")
    attention_config: AttentionConfig = get_field(VllmConfig, "attention_config")
    worker_cls: str = ParallelConfig.worker_cls
    worker_extension_cls: str = ParallelConfig.worker_extension_cls

    profiler_config: ProfilerConfig = get_field(VllmConfig, "profiler_config")

    kv_transfer_config: KVTransferConfig | None = None
    kv_events_config: KVEventsConfig | None = None

    ec_transfer_config: ECTransferConfig | None = None

    generation_config: str = ModelConfig.generation_config
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode
    override_generation_config: dict[str, Any] = get_field(
        ModelConfig, "override_generation_config"
    )
    model_impl: str = ModelConfig.model_impl
    override_attention_dtype: str = ModelConfig.override_attention_dtype
    attention_backend: AttentionBackendEnum | None = AttentionConfig.backend

    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales
    mamba_cache_dtype: MambaDType = CacheConfig.mamba_cache_dtype
    mamba_ssm_cache_dtype: MambaDType = CacheConfig.mamba_ssm_cache_dtype
    mamba_block_size: int | None = get_field(CacheConfig, "mamba_block_size")
    mamba_cache_mode: MambaCacheMode = CacheConfig.mamba_cache_mode

    additional_config: dict[str, Any] = get_field(VllmConfig, "additional_config")

    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load
    pt_load_map_location: str = LoadConfig.pt_load_map_location

    logits_processors: list[str | type[LogitsProcessor]] | None = (
        ModelConfig.logits_processors
    )
    """Custom logitproc types"""

    async_scheduling: bool | None = SchedulerConfig.async_scheduling

    stream_interval: int = SchedulerConfig.stream_interval

    kv_sharing_fast_prefill: bool = CacheConfig.kv_sharing_fast_prefill
    optimization_level: OptimizationLevel = VllmConfig.optimization_level

    # fault tolerance fields
    enable_fault_tolerance: bool = FaultToleranceConfig.enable_fault_tolerance
    engine_recovery_timeout: int = FaultToleranceConfig.engine_recovery_timeout
    internal_fault_report_port: int = FaultToleranceConfig.internal_fault_report_port
    external_fault_notify_port: int = FaultToleranceConfig.external_fault_notify_port
    gloo_comm_timeout: int = FaultToleranceConfig.gloo_comm_timeout
    shutdown_on_fault_tolerance_failure: bool = (
        FaultToleranceConfig.shutdown_on_fault_tolerance_failure
    )

    kv_offloading_size: float | None = CacheConfig.kv_offloading_size
    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend
    tokens_only: bool = False
    # AFD config
    afd_config: AFDConfig | None = None

    weight_transfer_config: WeightTransferConfig | None = None
    """Configuration for weight transfer during RL training. 
    Accepts a JSON string or dict with backend-specific options.
    Example: '{"backend": "nccl"}'"""

    def __post_init__(self):
        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, dict):
            self.compilation_config = CompilationConfig(**self.compilation_config)
        if isinstance(self.attention_config, dict):
            self.attention_config = AttentionConfig(**self.attention_config)
        if isinstance(self.eplb_config, dict):
            self.eplb_config = EPLBConfig(**self.eplb_config)
        if isinstance(self.weight_transfer_config, dict):
            self.weight_transfer_config = WeightTransferConfig(
                **self.weight_transfer_config
            )
        # Setup plugins
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        # when use hf offline,replace model and tokenizer id to local model path
        if huggingface_hub.constants.HF_HUB_OFFLINE:
            model_id = self.model
            self.model = get_model_path(self.model, self.revision)
            if model_id is not self.model:
                logger.info(
                    "HF_HUB_OFFLINE is True, replace model_id [%s] to model_path [%s]",
                    model_id,
                    self.model,
                )
            if self.tokenizer is not None:
                tokenizer_id = self.tokenizer
                self.tokenizer = get_model_path(self.tokenizer, self.tokenizer_revision)
                if tokenizer_id is not self.tokenizer:
                    logger.info(
                        "HF_HUB_OFFLINE is True, replace tokenizer_id [%s] "
                        "to tokenizer_path [%s]",
                        tokenizer_id,
                        self.tokenizer,
                    )

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # Model arguments
        model_kwargs = get_kwargs(ModelConfig)
        model_group = parser.add_argument_group(
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        if not ("serve" in sys.argv[1:] and "--help" in sys.argv[1:]):
            model_group.add_argument("--model", **model_kwargs["model"])
        model_group.add_argument("--runner", **model_kwargs["runner"])
        model_group.add_argument("--convert", **model_kwargs["convert"])
        model_group.add_argument("--tokenizer", **model_kwargs["tokenizer"])
        model_group.add_argument("--tokenizer-mode", **model_kwargs["tokenizer_mode"])
        model_group.add_argument(
            "--trust-remote-code", **model_kwargs["trust_remote_code"]
        )
        model_group.add_argument("--dtype", **model_kwargs["dtype"])
        model_group.add_argument("--seed", **model_kwargs["seed"])
        model_group.add_argument("--hf-config-path", **model_kwargs["hf_config_path"])
        model_group.add_argument(
            "--allowed-local-media-path", **model_kwargs["allowed_local_media_path"]
        )
        model_group.add_argument(
            "--allowed-media-domains", **model_kwargs["allowed_media_domains"]
        )
        model_group.add_argument("--revision", **model_kwargs["revision"])
        model_group.add_argument("--code-revision", **model_kwargs["code_revision"])
        model_group.add_argument(
            "--tokenizer-revision", **model_kwargs["tokenizer_revision"]
        )
        model_group.add_argument("--max-model-len", **model_kwargs["max_model_len"])
        model_group.add_argument("--quantization", "-q", **model_kwargs["quantization"])
        model_group.add_argument(
            "--allow-deprecated-quantization",
            **model_kwargs["allow_deprecated_quantization"],
        )
        model_group.add_argument("--enforce-eager", **model_kwargs["enforce_eager"])
        model_group.add_argument(
            "--enable-return-routed-experts",
            **model_kwargs["enable_return_routed_experts"],
        )
        model_group.add_argument("--max-logprobs", **model_kwargs["max_logprobs"])
        model_group.add_argument("--logprobs-mode", **model_kwargs["logprobs_mode"])
        model_group.add_argument(
            "--disable-sliding-window", **model_kwargs["disable_sliding_window"]
        )
        model_group.add_argument(
            "--disable-cascade-attn", **model_kwargs["disable_cascade_attn"]
        )
        model_group.add_argument(
            "--skip-tokenizer-init", **model_kwargs["skip_tokenizer_init"]
        )
        model_group.add_argument(
            "--enable-prompt-embeds", **model_kwargs["enable_prompt_embeds"]
        )
        model_group.add_argument(
            "--served-model-name", **model_kwargs["served_model_name"]
        )
        model_group.add_argument("--config-format", **model_kwargs["config_format"])
        # This one is a special case because it can bool
        # or str. TODO: Handle this in get_kwargs
        model_group.add_argument(
            "--hf-token",
            type=str,
            nargs="?",
            const=True,
            default=model_kwargs["hf_token"]["default"],
            help=model_kwargs["hf_token"]["help"],
        )
        model_group.add_argument("--hf-overrides", **model_kwargs["hf_overrides"])
        model_group.add_argument("--pooler-config", **model_kwargs["pooler_config"])
        model_group.add_argument(
            "--logits-processor-pattern", **model_kwargs["logits_processor_pattern"]
        )
        model_group.add_argument(
            "--generation-config", **model_kwargs["generation_config"]
        )
        model_group.add_argument(
            "--override-generation-config", **model_kwargs["override_generation_config"]
        )
        model_group.add_argument(
            "--enable-sleep-mode", **model_kwargs["enable_sleep_mode"]
        )
        model_group.add_argument("--model-impl", **model_kwargs["model_impl"])
        model_group.add_argument(
            "--override-attention-dtype", **model_kwargs["override_attention_dtype"]
        )
        model_group.add_argument(
            "--logits-processors", **model_kwargs["logits_processors"]
        )
        model_group.add_argument(
            "--io-processor-plugin", **model_kwargs["io_processor_plugin"]
        )

        # Model loading arguments
        load_kwargs = get_kwargs(LoadConfig)
        load_group = parser.add_argument_group(
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument("--load-format", **load_kwargs["load_format"])
        load_group.add_argument("--download-dir", **load_kwargs["download_dir"])
        load_group.add_argument(
            "--safetensors-load-strategy", **load_kwargs["safetensors_load_strategy"]
        )
        load_group.add_argument(
            "--model-loader-extra-config", **load_kwargs["model_loader_extra_config"]
        )
        load_group.add_argument("--ignore-patterns", **load_kwargs["ignore_patterns"])
        load_group.add_argument("--use-tqdm-on-load", **load_kwargs["use_tqdm_on_load"])
        load_group.add_argument(
            "--pt-load-map-location", **load_kwargs["pt_load_map_location"]
        )

        # Attention arguments
        attention_kwargs = get_kwargs(AttentionConfig)
        attention_group = parser.add_argument_group(
            title="AttentionConfig",
            description=AttentionConfig.__doc__,
        )
        attention_group.add_argument(
            "--attention-backend", **attention_kwargs["backend"]
        )

        # Structured outputs arguments
        structured_outputs_kwargs = get_kwargs(StructuredOutputsConfig)
        structured_outputs_group = parser.add_argument_group(
            title="StructuredOutputsConfig",
            description=StructuredOutputsConfig.__doc__,
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser",
            # Choices need to be validated after parsing to include plugins
            **structured_outputs_kwargs["reasoning_parser"],
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser-plugin",
            **structured_outputs_kwargs["reasoning_parser_plugin"],
        )

        # Parallel arguments
        parallel_kwargs = get_kwargs(ParallelConfig)
        parallel_group = parser.add_argument_group(
            title="ParallelConfig",
            description=ParallelConfig.__doc__,
        )
        parallel_group.add_argument(
            "--distributed-executor-backend",
            **parallel_kwargs["distributed_executor_backend"],
        )
        parallel_group.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            **parallel_kwargs["pipeline_parallel_size"],
        )
        parallel_group.add_argument("--master-addr", **parallel_kwargs["master_addr"])
        parallel_group.add_argument("--master-port", **parallel_kwargs["master_port"])
        parallel_group.add_argument("--nnodes", "-n", **parallel_kwargs["nnodes"])
        parallel_group.add_argument("--node-rank", "-r", **parallel_kwargs["node_rank"])
        parallel_group.add_argument(
            "--tensor-parallel-size", "-tp", **parallel_kwargs["tensor_parallel_size"]
        )
        parallel_group.add_argument(
            "--decode-context-parallel-size",
            "-dcp",
            **parallel_kwargs["decode_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--dcp-kv-cache-interleave-size",
            **parallel_kwargs["dcp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--cp-kv-cache-interleave-size",
            **parallel_kwargs["cp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--prefill-context-parallel-size",
            "-pcp",
            **parallel_kwargs["prefill_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--data-parallel-size", "-dp", **parallel_kwargs["data_parallel_size"]
        )
        parallel_group.add_argument(
            "--data-parallel-rank",
            "-dpn",
            type=int,
            help="Data parallel rank of this instance. "
            "When set, enables external load balancer mode.",
        )
        parallel_group.add_argument(
            "--data-parallel-start-rank",
            "-dpr",
            type=int,
            help="Starting data parallel rank for secondary nodes.",
        )
        parallel_group.add_argument(
            "--data-parallel-size-local",
            "-dpl",
            type=int,
            help="Number of data parallel replicas to run on this node.",
        )
        parallel_group.add_argument(
            "--data-parallel-address",
            "-dpa",
            type=str,
            help="Address of data parallel cluster head-node.",
        )
        parallel_group.add_argument(
            "--data-parallel-rpc-port",
            "-dpp",
            type=int,
            help="Port for data parallel RPC communication.",
        )
        parallel_group.add_argument(
            "--data-parallel-backend",
            "-dpb",
            type=str,
            default="mp",
            help='Backend for data parallel, either "mp" or "ray".',
        )
        parallel_group.add_argument(
            "--data-parallel-hybrid-lb",
            "-dph",
            **parallel_kwargs["data_parallel_hybrid_lb"],
        )
        parallel_group.add_argument(
            "--data-parallel-external-lb",
            "-dpe",
            **parallel_kwargs["data_parallel_external_lb"],
        )
        parallel_group.add_argument(
            "--enable-expert-parallel",
            "-ep",
            **parallel_kwargs["enable_expert_parallel"],
        )
        parallel_group.add_argument(
            "--all2all-backend", **parallel_kwargs["all2all_backend"]
        )
        parallel_group.add_argument("--enable-dbo", **parallel_kwargs["enable_dbo"])
        parallel_group.add_argument(
            "--ubatch-size",
            **parallel_kwargs["ubatch_size"],
        )
        parallel_group.add_argument(
            "--dbo-decode-token-threshold",
            **parallel_kwargs["dbo_decode_token_threshold"],
        )
        parallel_group.add_argument(
            "--dbo-prefill-token-threshold",
            **parallel_kwargs["dbo_prefill_token_threshold"],
        )
        parallel_group.add_argument(
            "--disable-nccl-for-dp-synchronization",
            **parallel_kwargs["disable_nccl_for_dp_synchronization"],
        )
        parallel_group.add_argument("--enable-eplb", **parallel_kwargs["enable_eplb"])
        parallel_group.add_argument("--eplb-config", **parallel_kwargs["eplb_config"])
        parallel_group.add_argument(
            "--expert-placement-strategy",
            **parallel_kwargs["expert_placement_strategy"],
        )

        parallel_group.add_argument(
            "--max-parallel-loading-workers",
            **parallel_kwargs["max_parallel_loading_workers"],
        )
        parallel_group.add_argument(
            "--ray-workers-use-nsight", **parallel_kwargs["ray_workers_use_nsight"]
        )
        parallel_group.add_argument(
            "--disable-custom-all-reduce",
            **parallel_kwargs["disable_custom_all_reduce"],
        )
        parallel_group.add_argument("--worker-cls", **parallel_kwargs["worker_cls"])
        parallel_group.add_argument(
            "--worker-extension-cls", **parallel_kwargs["worker_extension_cls"]
        )

        # KV cache arguments
        cache_kwargs = get_kwargs(CacheConfig)
        cache_group = parser.add_argument_group(
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
        cache_group.add_argument(
            "--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"]
        )
        cache_group.add_argument(
            "--kv-cache-memory-bytes", **cache_kwargs["kv_cache_memory_bytes"]
        )
        cache_group.add_argument("--swap-space", **cache_kwargs["swap_space"])
        cache_group.add_argument("--kv-cache-dtype", **cache_kwargs["cache_dtype"])
        cache_group.add_argument(
            "--num-gpu-blocks-override", **cache_kwargs["num_gpu_blocks_override"]
        )
        cache_group.add_argument(
            "--enable-prefix-caching",
            **{
                **cache_kwargs["enable_prefix_caching"],
                "default": None,
            },
        )
        cache_group.add_argument(
            "--prefix-caching-hash-algo", **cache_kwargs["prefix_caching_hash_algo"]
        )
        cache_group.add_argument("--cpu-offload-gb", **cache_kwargs["cpu_offload_gb"])
        cache_group.add_argument(
            "--calculate-kv-scales", **cache_kwargs["calculate_kv_scales"]
        )
        cache_group.add_argument(
            "--kv-sharing-fast-prefill", **cache_kwargs["kv_sharing_fast_prefill"]
        )
        cache_group.add_argument(
            "--mamba-cache-dtype", **cache_kwargs["mamba_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-ssm-cache-dtype", **cache_kwargs["mamba_ssm_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-block-size", **cache_kwargs["mamba_block_size"]
        )
        cache_group.add_argument(
            "--mamba-cache-mode", **cache_kwargs["mamba_cache_mode"]
        )
        cache_group.add_argument(
            "--kv-offloading-size", **cache_kwargs["kv_offloading_size"]
        )
        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )

        # Multimodal related configs
        multimodal_kwargs = get_kwargs(MultiModalConfig)
        multimodal_group = parser.add_argument_group(
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument(
            "--limit-mm-per-prompt", **multimodal_kwargs["limit_per_prompt"]
        )
        multimodal_group.add_argument(
            "--enable-mm-embeds", **multimodal_kwargs["enable_mm_embeds"]
        )
        multimodal_group.add_argument(
            "--media-io-kwargs", **multimodal_kwargs["media_io_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-kwargs", **multimodal_kwargs["mm_processor_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-gb", **multimodal_kwargs["mm_processor_cache_gb"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-type", **multimodal_kwargs["mm_processor_cache_type"]
        )
        multimodal_group.add_argument(
            "--mm-shm-cache-max-object-size-mb",
            **multimodal_kwargs["mm_shm_cache_max_object_size_mb"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-only", **multimodal_kwargs["mm_encoder_only"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-tp-mode", **multimodal_kwargs["mm_encoder_tp_mode"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-attn-backend",
            **multimodal_kwargs["mm_encoder_attn_backend"],
        )
        multimodal_group.add_argument(
            "--interleave-mm-strings", **multimodal_kwargs["interleave_mm_strings"]
        )
        multimodal_group.add_argument(
            "--skip-mm-profiling", **multimodal_kwargs["skip_mm_profiling"]
        )

        multimodal_group.add_argument(
            "--video-pruning-rate", **multimodal_kwargs["video_pruning_rate"]
        )

        # LoRA related configs
        lora_kwargs = get_kwargs(LoRAConfig)
        lora_group = parser.add_argument_group(
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            "--enable-lora",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of LoRA adapters.",
        )
        lora_group.add_argument("--max-loras", **lora_kwargs["max_loras"])
        lora_group.add_argument("--max-lora-rank", **lora_kwargs["max_lora_rank"])
        lora_group.add_argument(
            "--lora-dtype",
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument(
            "--enable-tower-connector-lora",
            **lora_kwargs["enable_tower_connector_lora"],
        )
        lora_group.add_argument("--max-cpu-loras", **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument(
            "--fully-sharded-loras", **lora_kwargs["fully_sharded_loras"]
        )
        lora_group.add_argument("--default-mm-loras", **lora_kwargs["default_mm_loras"])
        lora_group.add_argument(
            "--specialize-active-lora", **lora_kwargs["specialize_active_lora"]
        )

        # Observability arguments
        observability_kwargs = get_kwargs(ObservabilityConfig)
        observability_group = parser.add_argument_group(
            title="ObservabilityConfig",
            description=ObservabilityConfig.__doc__,
        )
        observability_group.add_argument(
            "--show-hidden-metrics-for-version",
            **observability_kwargs["show_hidden_metrics_for_version"],
        )
        observability_group.add_argument(
            "--otlp-traces-endpoint", **observability_kwargs["otlp_traces_endpoint"]
        )
        # TODO: generalise this special case
        choices = observability_kwargs["collect_detailed_traces"]["choices"]
        metavar = f"{{{','.join(choices)}}}"
        observability_kwargs["collect_detailed_traces"]["metavar"] = metavar
        observability_kwargs["collect_detailed_traces"]["choices"] += [
            ",".join(p) for p in permutations(get_args(DetailedTraceModules), r=2)
        ]
        observability_group.add_argument(
            "--collect-detailed-traces",
            **observability_kwargs["collect_detailed_traces"],
        )
        observability_group.add_argument(
            "--kv-cache-metrics", **observability_kwargs["kv_cache_metrics"]
        )
        observability_group.add_argument(
            "--kv-cache-metrics-sample",
            **observability_kwargs["kv_cache_metrics_sample"],
        )
        observability_group.add_argument(
            "--cudagraph-metrics",
            **observability_kwargs["cudagraph_metrics"],
        )
        observability_group.add_argument(
            "--enable-layerwise-nvtx-tracing",
            **observability_kwargs["enable_layerwise_nvtx_tracing"],
        )
        observability_group.add_argument(
            "--enable-mfu-metrics",
            **observability_kwargs["enable_mfu_metrics"],
        )
        observability_group.add_argument(
            "--enable-logging-iteration-details",
            **observability_kwargs["enable_logging_iteration_details"],
        )

        # Scheduler arguments
        scheduler_kwargs = get_kwargs(SchedulerConfig)
        scheduler_group = parser.add_argument_group(
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            "--max-num-batched-tokens",
            **{
                **scheduler_kwargs["max_num_batched_tokens"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-seqs",
            **{
                **scheduler_kwargs["max_num_seqs"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-partial-prefills", **scheduler_kwargs["max_num_partial_prefills"]
        )
        scheduler_group.add_argument(
            "--max-long-partial-prefills",
            **scheduler_kwargs["max_long_partial_prefills"],
        )
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"],
        )
        # multi-step scheduling has been removed; corresponding arguments
        # are no longer supported.
        scheduler_group.add_argument(
            "--scheduling-policy", **scheduler_kwargs["policy"]
        )
        scheduler_group.add_argument(
            "--enable-chunked-prefill",
            **{
                **scheduler_kwargs["enable_chunked_prefill"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--disable-chunked-mm-input", **scheduler_kwargs["disable_chunked_mm_input"]
        )
        scheduler_group.add_argument(
            "--scheduler-cls", **scheduler_kwargs["scheduler_cls"]
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
            **scheduler_kwargs["disable_hybrid_kv_cache_manager"],
        )
        scheduler_group.add_argument(
            "--async-scheduling", **scheduler_kwargs["async_scheduling"]
        )
        scheduler_group.add_argument(
            "--stream-interval", **scheduler_kwargs["stream_interval"]
        )

        # Compilation arguments
        compilation_kwargs = get_kwargs(CompilationConfig)
        compilation_group = parser.add_argument_group(
            title="CompilationConfig",
            description=CompilationConfig.__doc__,
        )
        compilation_group.add_argument(
            "--cudagraph-capture-sizes", **compilation_kwargs["cudagraph_capture_sizes"]
        )
        compilation_group.add_argument(
            "--max-cudagraph-capture-size",
            **compilation_kwargs["max_cudagraph_capture_size"],
        )

        # vLLM arguments
        vllm_kwargs = get_kwargs(VllmConfig)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        # We construct SpeculativeConfig using fields from other configs in
        # create_engine_config. So we set the type to a JSON string here to
        # delay the Pydantic validation that comes with SpeculativeConfig.
        vllm_kwargs["speculative_config"]["type"] = optional_type(json.loads)
        vllm_group.add_argument(
            "--speculative-config", **vllm_kwargs["speculative_config"]
        )
        vllm_group.add_argument(
            "--kv-transfer-config", **vllm_kwargs["kv_transfer_config"]
        )
        vllm_group.add_argument("--kv-events-config", **vllm_kwargs["kv_events_config"])
        vllm_group.add_argument(
            "--ec-transfer-config", **vllm_kwargs["ec_transfer_config"]
        )
        vllm_group.add_argument(
            "--compilation-config", "-cc", **vllm_kwargs["compilation_config"]
        )
        vllm_group.add_argument(
            "--attention-config", "-ac", **vllm_kwargs["attention_config"]
        )
        vllm_group.add_argument(
            "--additional-config", **vllm_kwargs["additional_config"]
        )
        vllm_group.add_argument(
            "--structured-outputs-config", **vllm_kwargs["structured_outputs_config"]
        )
        vllm_group.add_argument("--profiler-config", **vllm_kwargs["profiler_config"])
        vllm_group.add_argument("--afd-config", **vllm_kwargs["afd_config"])

        vllm_group.add_argument(
            "--optimization-level", **vllm_kwargs["optimization_level"]
        )
        vllm_group.add_argument(
            "--weight-transfer-config", **vllm_kwargs["weight_transfer_config"]
        )

        # fault tolerance arguments
        fault_tolerance_kwargs = get_kwargs(FaultToleranceConfig)
        fault_tolerance_group = parser.add_argument_group(
            title="FaultToleranceConfig",
            description=FaultToleranceConfig.__doc__,
        )
        fault_tolerance_group.add_argument(
            "--enable-fault-tolerance",
            **fault_tolerance_kwargs["enable_fault_tolerance"],
        )
        fault_tolerance_group.add_argument(
            "--shutdown-on-fault-tolerance-failure",
            **fault_tolerance_kwargs["shutdown_on_fault_tolerance_failure"],
        )
        fault_tolerance_group.add_argument(
            "--engine-recovery-timeout",
            **fault_tolerance_kwargs["engine_recovery_timeout"],
        )
        fault_tolerance_group.add_argument(
            "--internal-fault-report-port",
            **fault_tolerance_kwargs["internal_fault_report_port"],
        )
        fault_tolerance_group.add_argument(
            "--external-fault-notify-port",
            **fault_tolerance_kwargs["external_fault_notify_port"],
        )
        fault_tolerance_group.add_argument(
            "--gloo-comm-timeout",
            **fault_tolerance_kwargs["gloo_comm_timeout"],
        )

        # Other arguments
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="Disable logging statistics.",
        )

        parser.add_argument(
            "--aggregate-engine-logging",
            action="store_true",
            help="Log aggregate rather than per-engine statistics "
            "when using data parallelism.",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args

    def create_model_config(self) -> ModelConfig:
        # gguf file needs a specific model loader
        if is_gguf(self.model):
            self.quantization = self.load_format = "gguf"

        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        return ModelConfig(
            model=self.model,
            model_weights=self.model_weights,
            hf_config_path=self.hf_config_path,
            runner=self.runner,
            convert=self.convert,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            allowed_media_domains=self.allowed_media_domains,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            allow_deprecated_quantization=self.allow_deprecated_quantization,
            enforce_eager=self.enforce_eager,
            enable_return_routed_experts=self.enable_return_routed_experts,
            max_logprobs=self.max_logprobs,
            logprobs_mode=self.logprobs_mode,
            disable_sliding_window=self.disable_sliding_window,
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            enable_mm_embeds=self.enable_mm_embeds,
            interleave_mm_strings=self.interleave_mm_strings,
            media_io_kwargs=self.media_io_kwargs,
            skip_mm_profiling=self.skip_mm_profiling,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            mm_processor_cache_gb=self.mm_processor_cache_gb,
            mm_processor_cache_type=self.mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=self.mm_shm_cache_max_object_size_mb,
            mm_encoder_only=self.mm_encoder_only,
            mm_encoder_tp_mode=self.mm_encoder_tp_mode,
            mm_encoder_attn_backend=self.mm_encoder_attn_backend,
            pooler_config=self.pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
            override_attention_dtype=self.override_attention_dtype,
            logits_processors=self.logits_processors,
            video_pruning_rate=self.video_pruning_rate,
            io_processor_plugin=self.io_processor_plugin,
        )

    def validate_tensorizer_args(self):
        from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

        for key in self.model_loader_extra_config:
            if key in TensorizerConfig._fields:
                self.model_loader_extra_config["tensorizer_config"][key] = (
                    self.model_loader_extra_config[key]
                )

    def create_load_config(self) -> LoadConfig:
        if self.quantization == "bitsandbytes":
            self.load_format = "bitsandbytes"

        if self.load_format == "tensorizer":
            if hasattr(self.model_loader_extra_config, "to_serializable"):
                self.model_loader_extra_config = (
                    self.model_loader_extra_config.to_serializable()
                )
            self.model_loader_extra_config["tensorizer_config"] = {}
            self.model_loader_extra_config["tensorizer_config"]["tensorizer_dir"] = (
                self.model
            )
            self.validate_tensorizer_args()

        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            safetensors_load_strategy=self.safetensors_load_strategy,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
            use_tqdm_on_load=self.use_tqdm_on_load,
            pt_load_map_location=self.pt_load_map_location,
        )

    def create_speculative_config(
        self,
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
    ) -> SpeculativeConfig | None:
        """Initializes and returns a SpeculativeConfig object based on
        `speculative_config`.

        This function utilizes `speculative_config` to create a
        SpeculativeConfig object. The `speculative_config` can either be
        provided as a JSON string input via CLI arguments or directly as a
        dictionary from the engine.
        """
        if self.speculative_config is None:
            return None

        # Note(Shangming): These parameters are not obtained from the cli arg
        # '--speculative-config' and must be passed in when creating the engine
        # config.
        self.speculative_config.update(
            {
                "target_model_config": target_model_config,
                "target_parallel_config": target_parallel_config,
            }
        )
        return SpeculativeConfig(**self.speculative_config)

    def create_engine_config(
        self,
        usage_context: UsageContext | None = None,
        headless: bool = False,
    ) -> VllmConfig:
        """
        Create the VllmConfig.

        NOTE: If VllmConfig is incompatible, we raise an error.
        """
        current_platform.pre_register_and_update()

        device_config = DeviceConfig(device=cast(Device, current_platform.device_type))

        # Check if the model is a speculator and override model/tokenizer/config
        # BEFORE creating ModelConfig, so the config is created with the target model
        # Skip speculator detection for cloud storage models (eg: S3, GCS) since
        # HuggingFace cannot load configs directly from S3 URLs. S3 models can still
        # use speculators with explicit --speculative-config.
        if not is_cloud_storage(self.model):
            (self.model, self.tokenizer, self.speculative_config) = (
                maybe_override_with_speculators(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    vllm_speculative_config=self.speculative_config,
                )
            )

        model_config = self.create_model_config()
        self.model = model_config.model
        self.model_weights = model_config.model_weights
        self.tokenizer = model_config.tokenizer

        self._check_feature_supported(model_config)
        self._set_default_chunked_prefill_and_prefix_caching_args(model_config)
        self._set_default_max_num_seqs_and_batched_tokens_args(
            usage_context, model_config
        )

        sliding_window: int | None = None
        if not is_interleaved(model_config.hf_text_config):
            # Only set CacheConfig.sliding_window if the model is all sliding
            # window. Otherwise CacheConfig.sliding_window will override the
            # global layers in interleaved sliding window models.
            sliding_window = model_config.get_sliding_window()

        # Resolve "auto" kv_cache_dtype to actual value from model config
        resolved_cache_dtype = resolve_kv_cache_dtype_string(
            self.kv_cache_dtype, model_config
        )

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            swap_space=self.swap_space,
            cache_dtype=resolved_cache_dtype,
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=sliding_window,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            cpu_offload_gb=self.cpu_offload_gb,
            calculate_kv_scales=self.calculate_kv_scales,
            kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
            mamba_cache_dtype=self.mamba_cache_dtype,
            mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
            mamba_block_size=self.mamba_block_size,
            mamba_cache_mode=self.mamba_cache_mode,
            kv_offloading_size=self.kv_offloading_size,
            kv_offloading_backend=self.kv_offloading_backend,
        )

        ray_runtime_env = None
        if is_ray_initialized():
            # Ray Serve LLM calls `create_engine_config` in the context
            # of a Ray task, therefore we check is_ray_initialized()
            # as opposed to is_in_ray_actor().
            import ray

            ray_runtime_env = ray.get_runtime_context().runtime_env
            # Avoid logging sensitive environment variables
            sanitized_env = ray_runtime_env.to_dict() if ray_runtime_env else {}
            if "env_vars" in sanitized_env:
                sanitized_env["env_vars"] = {
                    k: "***" for k in sanitized_env["env_vars"]
                }
            logger.info("Using ray runtime env (env vars redacted): %s", sanitized_env)

        # Get the current placement group if Ray is initialized and
        # we are in a Ray actor. If so, then the placement group will be
        # passed to spawned processes.
        placement_group = None
        if is_in_ray_actor():
            import ray

            # This call initializes Ray automatically if it is not initialized,
            # but we should not do this here.
            placement_group = ray.util.get_current_placement_group()

        assert not headless or not self.data_parallel_hybrid_lb, (
            "data_parallel_hybrid_lb is not applicable in headless mode"
        )
        assert not (self.data_parallel_hybrid_lb and self.data_parallel_external_lb), (
            "data_parallel_hybrid_lb and data_parallel_external_lb cannot both be True."
        )
        assert self.data_parallel_backend == "mp" or self.nnodes == 1, (
            "nnodes > 1 is only supported with data_parallel_backend=mp"
        )
        inferred_data_parallel_rank = 0
        if self.nnodes > 1:
            world_size = (
                self.data_parallel_size
                * self.pipeline_parallel_size
                * self.tensor_parallel_size
            )
            world_size_within_dp = (
                self.pipeline_parallel_size * self.tensor_parallel_size
            )
            local_world_size = world_size // self.nnodes
            assert world_size % self.nnodes == 0, (
                f"world_size={world_size} must be divisible by nnodes={self.nnodes}."
            )
            assert self.node_rank < self.nnodes, (
                f"node_rank={self.node_rank} must be less than nnodes={self.nnodes}."
            )
            inferred_data_parallel_rank = (
                self.node_rank * local_world_size
            ) // world_size_within_dp
            if self.data_parallel_size > 1 and self.data_parallel_external_lb:
                self.data_parallel_rank = inferred_data_parallel_rank
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d for external lb",
                    self.data_parallel_rank,
                    self.node_rank,
                )
            elif self.data_parallel_size_local is None:
                # Infer data parallel size local for internal dplb:
                self.data_parallel_size_local = max(
                    local_world_size // world_size_within_dp, 1
                )
        data_parallel_external_lb = (
            self.data_parallel_external_lb or self.data_parallel_rank is not None
        )
        # Local DP rank = 1, use pure-external LB.
        if data_parallel_external_lb:
            assert self.data_parallel_rank is not None, (
                "data_parallel_rank or node_rank must be specified if "
                "data_parallel_external_lb is enable."
            )
            assert self.data_parallel_size_local in (1, None), (
                "data_parallel_size_local must be 1 or None when data_parallel_rank "
                "is set"
            )
            data_parallel_size_local = 1
            # Use full external lb if we have local_size of 1.
            self.data_parallel_hybrid_lb = False
        elif self.data_parallel_size_local is not None:
            data_parallel_size_local = self.data_parallel_size_local

            if self.data_parallel_start_rank and not headless:
                # Infer hybrid LB mode.
                self.data_parallel_hybrid_lb = True

            if self.data_parallel_hybrid_lb and data_parallel_size_local == 1:
                # Use full external lb if we have local_size of 1.
                logger.warning(
                    "data_parallel_hybrid_lb is not eligible when "
                    "data_parallel_size_local = 1, autoswitch to "
                    "data_parallel_external_lb."
                )
                data_parallel_external_lb = True
                self.data_parallel_hybrid_lb = False

            if data_parallel_size_local == self.data_parallel_size:
                # Disable hybrid LB mode if set for a single node
                self.data_parallel_hybrid_lb = False

            self.data_parallel_rank = (
                self.data_parallel_start_rank or inferred_data_parallel_rank
            )
            if self.nnodes > 1:
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d",
                    self.data_parallel_rank,
                    self.node_rank,
                )
        else:
            assert not self.data_parallel_hybrid_lb, (
                "data_parallel_size_local must be set to use data_parallel_hybrid_lb."
            )

            if self.data_parallel_backend == "ray" and (
                envs.VLLM_RAY_DP_PACK_STRATEGY == "span"
            ):
                # Data parallel size defaults to 1 if DP ranks are spanning
                # multiple nodes
                data_parallel_size_local = 1
            else:
                # Otherwise local DP size defaults to global DP size if not set
                data_parallel_size_local = self.data_parallel_size

        # DP address, used in multi-node case for torch distributed group
        # and ZMQ sockets.
        if self.data_parallel_address is None:
            if self.data_parallel_backend == "ray":
                host_ip = get_ip()
                logger.info(
                    "Using host IP %s as ray-based data parallel address", host_ip
                )
                data_parallel_address = host_ip
            else:
                assert self.data_parallel_backend == "mp", (
                    "data_parallel_backend can only be ray or mp, got %s",
                    self.data_parallel_backend,
                )
                data_parallel_address = (
                    self.master_addr or ParallelConfig.data_parallel_master_ip
                )
        else:
            data_parallel_address = self.data_parallel_address

        # This port is only used when there are remote data parallel engines,
        # otherwise the local IPC transport is used.
        data_parallel_rpc_port = (
            self.data_parallel_rpc_port
            if (self.data_parallel_rpc_port is not None)
            else ParallelConfig.data_parallel_rpc_port
        )

        if self.tokens_only and not model_config.skip_tokenizer_init:
            model_config.skip_tokenizer_init = True
            logger.info("Skipping tokenizer initialization for tokens-only mode.")

        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            prefill_context_parallel_size=self.prefill_context_parallel_size,
            data_parallel_size=self.data_parallel_size,
            data_parallel_rank=self.data_parallel_rank or 0,
            data_parallel_external_lb=data_parallel_external_lb,
            data_parallel_size_local=data_parallel_size_local,
            master_addr=self.master_addr,
            master_port=self.master_port,
            nnodes=self.nnodes,
            node_rank=self.node_rank,
            data_parallel_master_ip=data_parallel_address,
            data_parallel_rpc_port=data_parallel_rpc_port,
            data_parallel_backend=self.data_parallel_backend,
            data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,
            is_moe_model=model_config.is_moe,
            enable_expert_parallel=self.enable_expert_parallel,
            all2all_backend=self.all2all_backend,
            enable_dbo=self.enable_dbo,
            ubatch_size=self.ubatch_size,
            dbo_decode_token_threshold=self.dbo_decode_token_threshold,
            dbo_prefill_token_threshold=self.dbo_prefill_token_threshold,
            disable_nccl_for_dp_synchronization=self.disable_nccl_for_dp_synchronization,
            enable_eplb=self.enable_eplb,
            eplb_config=self.eplb_config,
            expert_placement_strategy=self.expert_placement_strategy,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            ray_runtime_env=ray_runtime_env,
            placement_group=placement_group,
            distributed_executor_backend=self.distributed_executor_backend,
            worker_cls=self.worker_cls,
            worker_extension_cls=self.worker_extension_cls,
            decode_context_parallel_size=self.decode_context_parallel_size,
            dcp_kv_cache_interleave_size=self.dcp_kv_cache_interleave_size,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            _api_process_count=self._api_process_count,
            _api_process_rank=self._api_process_rank,
        )

        speculative_config = self.create_speculative_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )

        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
            is_multimodal_model=model_config.is_multimodal_model,
            is_encoder_decoder=model_config.is_encoder_decoder,
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
            async_scheduling=self.async_scheduling,
            stream_interval=self.stream_interval,
        )

        if not model_config.is_multimodal_model and self.default_mm_loras:
            raise ValueError(
                "Default modality-specific LoRA(s) were provided for a "
                "non multimodal model"
            )

        lora_config = (
            LoRAConfig(
                max_lora_rank=self.max_lora_rank,
                max_loras=self.max_loras,
                default_mm_loras=self.default_mm_loras,
                fully_sharded_loras=self.fully_sharded_loras,
                lora_dtype=self.lora_dtype,
                enable_tower_connector_lora=self.enable_tower_connector_lora,
                specialize_active_lora=self.specialize_active_lora,
                max_cpu_loras=self.max_cpu_loras
                if self.max_cpu_loras and self.max_cpu_loras > 0
                else None,
            )
            if self.enable_lora
            else None
        )

        if (
            lora_config is not None
            and speculative_config is not None
            and scheduler_config.max_num_batched_tokens
            < (
                scheduler_config.max_num_seqs
                * (speculative_config.num_speculative_tokens + 1)
            )
        ):
            raise ValueError(
                "Consider increasing max_num_batched_tokens or "
                "decreasing num_speculative_tokens"
            )

        # bitsandbytes pre-quantized model need a specific model loader
        if model_config.quantization == "bitsandbytes":
            self.quantization = self.load_format = "bitsandbytes"

        # Attention config overrides
        attention_config = copy.deepcopy(self.attention_config)
        if self.attention_backend is not None:
            if attention_config.backend is not None:
                raise ValueError(
                    "attention_backend and attention_config.backend "
                    "are mutually exclusive"
                )
            # Convert string to enum if needed (CLI parsing returns a string)
            if isinstance(self.attention_backend, str):
                attention_config.backend = AttentionBackendEnum[
                    self.attention_backend.upper()
                ]
            else:
                attention_config.backend = self.attention_backend

        load_config = self.create_load_config()

        # Pass reasoning_parser into StructuredOutputsConfig
        if self.reasoning_parser:
            self.structured_outputs_config.reasoning_parser = self.reasoning_parser

        if self.reasoning_parser_plugin:
            self.structured_outputs_config.reasoning_parser_plugin = (
                self.reasoning_parser_plugin
            )

        observability_config = ObservabilityConfig(
            show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_detailed_traces=self.collect_detailed_traces,
            kv_cache_metrics=self.kv_cache_metrics,
            kv_cache_metrics_sample=self.kv_cache_metrics_sample,
            cudagraph_metrics=self.cudagraph_metrics,
            enable_layerwise_nvtx_tracing=self.enable_layerwise_nvtx_tracing,
            enable_mfu_metrics=self.enable_mfu_metrics,
            enable_mm_processor_stats=self.enable_mm_processor_stats,
            enable_logging_iteration_details=self.enable_logging_iteration_details,
        )

        fault_tolerance_config = FaultToleranceConfig(
            enable_fault_tolerance=self.enable_fault_tolerance,
            shutdown_on_fault_tolerance_failure=self.shutdown_on_fault_tolerance_failure,
            engine_recovery_timeout=self.engine_recovery_timeout,
            internal_fault_report_port=self.internal_fault_report_port,
            external_fault_notify_port=self.external_fault_notify_port,
            gloo_comm_timeout=self.gloo_comm_timeout,
        )

        # Compilation config overrides
        compilation_config = copy.deepcopy(self.compilation_config)
        if self.cudagraph_capture_sizes is not None:
            if compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "cudagraph_capture_sizes and compilation_config."
                    "cudagraph_capture_sizes are mutually exclusive"
                )
            compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
        if self.max_cudagraph_capture_size is not None:
            if compilation_config.max_cudagraph_capture_size is not None:
                raise ValueError(
                    "max_cudagraph_capture_size and compilation_config."
                    "max_cudagraph_capture_size are mutually exclusive"
                )
            compilation_config.max_cudagraph_capture_size = (
                self.max_cudagraph_capture_size
            )
        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_config,
            attention_config=attention_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            structured_outputs_config=self.structured_outputs_config,
            observability_config=observability_config,
            compilation_config=compilation_config,
            kv_transfer_config=self.kv_transfer_config,
            kv_events_config=self.kv_events_config,
            ec_transfer_config=self.ec_transfer_config,
            profiler_config=self.profiler_config,
            fault_tolerance_config=fault_tolerance_config,
            additional_config=self.additional_config,
            optimization_level=self.optimization_level,
            weight_transfer_config=self.weight_transfer_config,
            afd_config=self.afd_config,
        )

        return config

    def _check_feature_supported(self, model_config: ModelConfig):
        """Raise an error if the feature is not supported."""
        if self.logits_processor_pattern != EngineArgs.logits_processor_pattern:
            _raise_unsupported_error(feature_name="--logits-processor-pattern")

        # No Concurrent Partial Prefills so far.
        if (
            self.max_num_partial_prefills != SchedulerConfig.max_num_partial_prefills
            or self.max_long_partial_prefills
            != SchedulerConfig.max_long_partial_prefills
        ):
            _raise_unsupported_error(feature_name="Concurrent Partial Prefill")

        if self.pipeline_parallel_size > 1:
            supports_pp = getattr(
                self.distributed_executor_backend, "supports_pp", False
            )
            if not supports_pp and self.distributed_executor_backend not in (
                ParallelConfig.distributed_executor_backend,
                "ray",
                "mp",
                "external_launcher",
            ):
                name = (
                    "Pipeline Parallelism without Ray distributed "
                    "executor or multiprocessing executor or external "
                    "launcher"
                )
                _raise_unsupported_error(feature_name=name)

    @classmethod
    def get_batch_defaults(
        cls,
        world_size: int,
    ) -> tuple[dict[UsageContext | None, int], dict[UsageContext | None, int]]:
        from vllm.usage.usage_lib import UsageContext

        default_max_num_batched_tokens: dict[UsageContext | None, int]
        default_max_num_seqs: dict[UsageContext | None, int]

        # When no user override, set the default values based on the usage
        # context.
        # Use different default values for different hardware.

        # Try to query the device name on the current platform. If it fails,
        # it may be because the platform that imports vLLM is not the same
        # as the platform that vLLM is running on (e.g. the case of scaling
        # vLLM with Ray) and has no GPUs. In this case we use the default
        # values for non-H100/H200 GPUs.
        try:
            device_memory = current_platform.get_device_total_memory()
            device_name = current_platform.get_device_name().lower()
        except Exception:
            # This is only used to set default_max_num_batched_tokens
            device_memory = 0
            device_name = ""

        # NOTE(Kuntai): Setting large `max_num_batched_tokens` for A100 reduces
        # throughput, see PR #17885 for more details.
        # So here we do an extra device name check to prevent such regression.
        if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
            # For GPUs like H100 and MI300x, use larger default values.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 1024,
                UsageContext.OPENAI_API_SERVER: 1024,
            }
        else:
            # TODO(woosuk): Tune the default values for other hardware.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256,
                UsageContext.OPENAI_API_SERVER: 256,
            }

        # tpu specific default values.
        if current_platform.is_tpu():
            chip_name = current_platform.get_device_name()

            if chip_name == "V6E":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 2048,
                    UsageContext.OPENAI_API_SERVER: 1024,
                }
            elif chip_name == "V5E":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 1024,
                    UsageContext.OPENAI_API_SERVER: 512,
                }
            elif chip_name == "V5P":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 512,
                    UsageContext.OPENAI_API_SERVER: 256,
                }

        # cpu specific default values.
        if current_platform.is_cpu():
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 4096 * world_size,
                UsageContext.OPENAI_API_SERVER: 2048 * world_size,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256 * world_size,
                UsageContext.OPENAI_API_SERVER: 128 * world_size,
            }

        return default_max_num_batched_tokens, default_max_num_seqs

    def _set_default_chunked_prefill_and_prefix_caching_args(
        self, model_config: ModelConfig
    ) -> None:
        default_chunked_prefill = model_config.is_chunked_prefill_supported
        default_prefix_caching = model_config.is_prefix_caching_supported

        if self.enable_chunked_prefill is None:
            self.enable_chunked_prefill = default_chunked_prefill

            logger.debug(
                "%s chunked prefill by default",
                "Enabling" if default_chunked_prefill else "Disabling",
            )
        elif (
            model_config.runner_type == "generate"
            and not self.enable_chunked_prefill
            and default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support disabling chunked prefill. "
                "Disabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_chunked_prefill
            and not default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support chunked prefill. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )

        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = default_prefix_caching

            logger.debug(
                "%s prefix caching by default",
                "Enabling" if default_prefix_caching else "Disabling",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_prefix_caching
            and not default_prefix_caching
        ):
            logger.warning_once(
                "This model does not officially support prefix caching. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )

        # Disable chunked prefill and prefix caching for:
        # POWER (ppc64le)/s390x/RISCV CPUs in V1
        if current_platform.is_cpu() and current_platform.get_cpu_architecture() in (
            CpuArchEnum.POWERPC,
            CpuArchEnum.S390X,
            CpuArchEnum.RISCV,
        ):
            logger.info(
                "Chunked prefill is not supported for POWER, "
                "S390X and RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_chunked_prefill = False
            logger.info(
                "Prefix caching is not supported for POWER, "
                "S390X and RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_prefix_caching = False

    def _set_default_max_num_seqs_and_batched_tokens_args(
        self,
        usage_context: UsageContext | None,
        model_config: ModelConfig,
    ):
        world_size = self.pipeline_parallel_size * self.tensor_parallel_size
        (
            default_max_num_batched_tokens,
            default_max_num_seqs,
        ) = self.get_batch_defaults(world_size)

        orig_max_num_batched_tokens = self.max_num_batched_tokens
        orig_max_num_seqs = self.max_num_seqs

        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = default_max_num_batched_tokens.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

        if self.max_num_seqs is None:
            self.max_num_seqs = default_max_num_seqs.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_SEQS,
            )

        if orig_max_num_batched_tokens is None:
            if not self.enable_chunked_prefill:
                # If max_model_len is too short, use the default for higher throughput.
                self.max_num_batched_tokens = max(
                    model_config.max_model_len,
                    self.max_num_batched_tokens,
                )

            # When using default settings,
            # Ensure max_num_batched_tokens does not exceed model limit.
            # Some models (e.g., Whisper) have embeddings tied to max length.
            self.max_num_batched_tokens = min(
                self.max_num_seqs * model_config.max_model_len,
                self.max_num_batched_tokens,
            )

            logger.debug(
                "Defaulting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens,
                usage_context.value if usage_context else None,
            )

        if orig_max_num_seqs is None:
            assert self.max_num_batched_tokens is not None  # For type checking
            self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

            logger.debug(
                "Defaulting max_num_seqs to %d for %s usage context.",
                self.max_num_seqs,
                usage_context.value if usage_context else None,
            )


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    enable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
        parser: FlexibleArgumentParser, async_args_only: bool = False
    ) -> FlexibleArgumentParser:
        # Initialize plugin to update the parser, for example, The plugin may
        # add a new kind of quantization method to --quantization argument or
        # a new device to --device argument.
        load_general_plugins()
        if not async_args_only:
            parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--enable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=AsyncEngineArgs.enable_log_requests,
            help="Enable logging requests.",
        )
        parser.add_argument(
            "--disable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=not AsyncEngineArgs.enable_log_requests,
            help="[DEPRECATED] Disable logging requests.",
            deprecated=True,
        )
        current_platform.pre_register_and_update(parser)
        return parser


def _raise_unsupported_error(feature_name: str):
    msg = (
        f"{feature_name} is not supported. We recommend to "
        f"remove {feature_name} from your config."
    )
    raise NotImplementedError(msg)


def human_readable_int(value: str) -> int:
    """Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    """
    value = value.strip()

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgGtT])", value)
    if match:
        decimal_multiplier = {
            "k": 10**3,
            "m": 10**6,
            "g": 10**9,
            "t": 10**12,
        }
        binary_multiplier = {
            "K": 2**10,
            "M": 2**20,
            "G": 2**30,
            "T": 2**40,
        }

        number, suffix = match.groups()
        if suffix in decimal_multiplier:
            mult = decimal_multiplier[suffix]
            return int(float(number) * mult)
        elif suffix in binary_multiplier:
            mult = binary_multiplier[suffix]
            # Do not allow decimals with binary multipliers
            try:
                return int(number) * mult
            except ValueError as e:
                raise argparse.ArgumentTypeError(
                    "Decimals are not allowed "
                    f"with binary suffixes like {suffix}. Did you mean to use "
                    f"{number}{suffix.lower()} instead?"
                ) from e

    # Regular plain number.
    return int(value)


def human_readable_int_or_auto(value: str) -> int:
    """Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.
    Also accepts -1 or 'auto' as a special value for auto-detection.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    - '-1' or 'auto' -> -1 (special value for auto-detection)
    """
    value = value.strip()

    if value == "-1" or value.lower() == "auto":
        return -1

    return human_readable_int(value)
