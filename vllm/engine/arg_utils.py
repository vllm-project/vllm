# SPDX-License-Identifier: Apache-2.0

# yapf: disable
import argparse
import dataclasses
import json
import re
import sys
import threading
import warnings
from dataclasses import MISSING, dataclass, fields, is_dataclass
from itertools import permutations
from typing import (Annotated, Any, Callable, Dict, List, Literal, Optional,
                    Type, TypeVar, Union, cast, get_args, get_origin)

import torch
from typing_extensions import TypeIs, deprecated

import vllm.envs as envs
from vllm.config import (BlockSize, CacheConfig, CacheDType, CompilationConfig,
                         ConfigFormat, ConfigType, DecodingConfig,
                         DetailedTraceModules, Device, DeviceConfig,
                         DistributedExecutorBackend, GuidedDecodingBackend,
                         GuidedDecodingBackendV1, HfOverrides, KVEventsConfig,
                         KVTransferConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ModelDType, ModelImpl, MultiModalConfig,
                         ObservabilityConfig, ParallelConfig, PoolerConfig,
                         PrefixCachingHashAlgo, PromptAdapterConfig,
                         SchedulerConfig, SchedulerPolicy, SpeculativeConfig,
                         TaskOption, TokenizerMode, TokenizerPoolConfig,
                         VllmConfig, get_attr_docs, get_field)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.plugins import load_general_plugins
from vllm.reasoning import ReasoningParserManager
from vllm.test_utils import MODEL_WEIGHTS_S3_BUCKET, MODELS_ON_S3
from vllm.transformers_utils.utils import check_gguf_file
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (STR_DUAL_CHUNK_FLASH_ATTN_VAL, FlexibleArgumentParser,
                        GiB_bytes, is_in_doc_build, is_in_ray_actor)

# yapf: enable

logger = init_logger(__name__)

# object is used to allow for special typing forms
T = TypeVar("T")
TypeHint = Union[type[Any], object]
TypeHintT = Union[type[T], object]


def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:

    def _parse_type(val: str) -> T:
        try:
            if return_type is json.loads and not re.match("^{.*}$", val):
                return cast(T, nullable_kvs(val))
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to {return_type}.") from e

    return _parse_type


def optional_type(
        return_type: Callable[[str], T]) -> Callable[[str], Optional[T]]:

    def _optional_type(val: str) -> Optional[T]:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)

    return _optional_type


def union_dict_and_str(val: str) -> Optional[Union[str, dict[str, str]]]:
    if not re.match("^{.*}$", val):
        return str(val)
    return optional_type(json.loads)(val)


@deprecated(
    "Passing a JSON argument as a string containing comma separated key=value "
    "pairs is deprecated. This will be removed in v0.10.0. Please use a JSON "
    "string instead.")
def nullable_kvs(val: str) -> dict[str, int]:
    """Parses a string containing comma separate key [str] to value [int]
    pairs into a dictionary.

    Args:
        val: String value to be parsed.

    Returns:
        Dictionary with parsed values.
    """
    out_dict: dict[str, int] = {}
    for item in val.split(","):
        kv_parts = [part.lower().strip() for part in item.split("=")]
        if len(kv_parts) != 2:
            raise argparse.ArgumentTypeError(
                "Each item should be in the form KEY=VALUE")
        key, value = kv_parts

        try:
            parsed_value = int(value)
        except ValueError as exc:
            msg = f"Failed to parse value of item {key}={value}"
            raise argparse.ArgumentTypeError(msg) from exc

        if key in out_dict and out_dict[key] != parsed_value:
            raise argparse.ArgumentTypeError(
                f"Conflicting values specified for key: {key}")
        out_dict[key] = parsed_value

    return out_dict


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
    """Convert Literal type hints to argparse kwargs."""
    type_hint = get_type(type_hints, Literal)
    choices = get_args(type_hint)
    choice_type = type(choices[0])
    if not all(isinstance(choice, choice_type) for choice in choices):
        raise ValueError(
            "All choices must be of the same type. "
            f"Got {choices} with types {[type(c) for c in choices]}")
    return {"type": choice_type, "choices": sorted(choices)}


def is_not_builtin(type_hint: TypeHint) -> bool:
    """Check if the class is not a built-in type."""
    return type_hint.__module__ != "builtins"


def get_kwargs(cls: ConfigType) -> dict[str, Any]:
    cls_docs = get_attr_docs(cls)
    kwargs = {}
    for field in fields(cls):
        # Get the set of possible types for the field
        type_hints: set[TypeHint] = set()
        if get_origin(field.type) in {Union, Annotated}:
            type_hints.update(get_args(field.type))
        else:
            type_hints.add(field.type)

        # If the field is a dataclass, we can use the model_validate_json
        generator = (th for th in type_hints if is_dataclass(th))
        dataclass_cls = next(generator, None)

        # Get the default value of the field
        if field.default is not MISSING:
            default = field.default
        elif field.default_factory is not MISSING:
            if is_dataclass(field.default_factory) and is_in_doc_build():
                default = {}
            else:
                default = field.default_factory()

        # Get the help text for the field
        name = field.name
        help = cls_docs[name].strip()
        # Escape % for argparse
        help = help.replace("%", "%%")

        # Initialise the kwargs dictionary for the field
        kwargs[name] = {"default": default, "help": help}

        # Set other kwargs based on the type hints
        json_tip = "\n\nShould be a valid JSON string."
        if dataclass_cls is not None:
            dataclass_init = lambda x, f=dataclass_cls: f(**json.loads(x))
            # Special case for configs with a from_cli method
            if hasattr(dataclass_cls, "from_cli"):
                from_cli = dataclass_cls.from_cli
                dataclass_init = lambda x, f=from_cli: f(x)
            kwargs[name]["type"] = dataclass_init
            kwargs[name]["help"] += json_tip
        elif contains_type(type_hints, bool):
            # Creates --no-<name> and --<name> flags
            kwargs[name]["action"] = argparse.BooleanOptionalAction
        elif contains_type(type_hints, Literal):
            kwargs[name].update(literal_to_kwargs(type_hints))
        elif contains_type(type_hints, tuple):
            type_hint = get_type(type_hints, tuple)
            types = get_args(type_hint)
            tuple_type = types[0]
            assert all(t is tuple_type for t in types if t is not Ellipsis), (
                "All non-Ellipsis tuple elements must be of the same "
                f"type. Got {types}.")
            kwargs[name]["type"] = tuple_type
            kwargs[name]["nargs"] = "+" if Ellipsis in types else len(types)
        elif contains_type(type_hints, list):
            type_hint = get_type(type_hints, list)
            types = get_args(type_hint)
            assert len(types) == 1, (
                "List type must have exactly one type. Got "
                f"{type_hint} with types {types}")
            kwargs[name]["type"] = types[0]
            kwargs[name]["nargs"] = "+"
        elif contains_type(type_hints, int):
            kwargs[name]["type"] = int
            # Special case for large integers
            if name in {"max_model_len"}:
                kwargs[name]["type"] = human_readable_int
        elif contains_type(type_hints, float):
            kwargs[name]["type"] = float
        elif contains_type(type_hints,
                           dict) and (contains_type(type_hints, str) or any(
                               is_not_builtin(th) for th in type_hints)):
            kwargs[name]["type"] = union_dict_and_str
        elif contains_type(type_hints, dict):
            # Dict arguments will always be optional
            kwargs[name]["type"] = parse_type(json.loads)
            kwargs[name]["help"] += json_tip
        elif (contains_type(type_hints, str)
              or any(is_not_builtin(th) for th in type_hints)):
            kwargs[name]["type"] = str
        else:
            raise ValueError(
                f"Unsupported type {type_hints} for argument {name}.")

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


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = ModelConfig.model
    served_model_name: Optional[Union[
        str, List[str]]] = ModelConfig.served_model_name
    tokenizer: Optional[str] = ModelConfig.tokenizer
    hf_config_path: Optional[str] = ModelConfig.hf_config_path
    task: TaskOption = ModelConfig.task
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds
    tokenizer_mode: TokenizerMode = ModelConfig.tokenizer_mode
    trust_remote_code: bool = ModelConfig.trust_remote_code
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path
    download_dir: Optional[str] = LoadConfig.download_dir
    load_format: str = LoadConfig.load_format
    config_format: str = ModelConfig.config_format
    dtype: ModelDType = ModelConfig.dtype
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    seed: Optional[int] = ModelConfig.seed
    max_model_len: Optional[int] = ModelConfig.max_model_len
    cuda_graph_sizes: list[int] = get_field(SchedulerConfig,
                                            "cuda_graph_sizes")
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[
        DistributedExecutorBackend,
        Type[ExecutorBase]]] = ParallelConfig.distributed_executor_backend
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    data_parallel_size: int = ParallelConfig.data_parallel_size
    data_parallel_size_local: Optional[int] = None
    data_parallel_address: Optional[str] = None
    data_parallel_rpc_port: Optional[int] = None
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    max_parallel_loading_workers: Optional[
        int] = ParallelConfig.max_parallel_loading_workers
    block_size: Optional[BlockSize] = CacheConfig.block_size
    enable_prefix_caching: Optional[bool] = CacheConfig.enable_prefix_caching
    prefix_caching_hash_algo: PrefixCachingHashAlgo = \
        CacheConfig.prefix_caching_hash_algo
    disable_sliding_window: bool = ModelConfig.disable_sliding_window
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn
    use_v2_block_manager: bool = True
    swap_space: float = CacheConfig.swap_space
    cpu_offload_gb: float = CacheConfig.cpu_offload_gb
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    max_num_batched_tokens: Optional[
        int] = SchedulerConfig.max_num_batched_tokens
    max_num_partial_prefills: int = SchedulerConfig.max_num_partial_prefills
    max_long_partial_prefills: int = SchedulerConfig.max_long_partial_prefills
    long_prefill_token_threshold: int = \
        SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: Optional[int] = SchedulerConfig.max_num_seqs
    max_logprobs: int = ModelConfig.max_logprobs
    disable_log_stats: bool = False
    revision: Optional[str] = ModelConfig.revision
    code_revision: Optional[str] = ModelConfig.code_revision
    rope_scaling: dict[str, Any] = get_field(ModelConfig, "rope_scaling")
    rope_theta: Optional[float] = ModelConfig.rope_theta
    hf_token: Optional[Union[bool, str]] = ModelConfig.hf_token
    hf_overrides: Optional[HfOverrides] = \
        get_field(ModelConfig, "hf_overrides")
    tokenizer_revision: Optional[str] = ModelConfig.tokenizer_revision
    quantization: Optional[QuantizationMethods] = ModelConfig.quantization
    enforce_eager: bool = ModelConfig.enforce_eager
    max_seq_len_to_capture: int = ModelConfig.max_seq_len_to_capture
    disable_custom_all_reduce: bool = ParallelConfig.disable_custom_all_reduce
    # The following three fields are deprecated and will be removed in a future
    # release. Setting them will have no effect. Please remove them from your
    # configurations.
    tokenizer_pool_size: int = TokenizerPoolConfig.pool_size
    tokenizer_pool_type: str = TokenizerPoolConfig.pool_type
    tokenizer_pool_extra_config: dict = \
        get_field(TokenizerPoolConfig, "extra_config")
    limit_mm_per_prompt: dict[str, int] = \
        get_field(MultiModalConfig, "limit_per_prompt")
    mm_processor_kwargs: Optional[Dict[str, Any]] = \
        MultiModalConfig.mm_processor_kwargs
    disable_mm_preprocessor_cache: bool = \
        MultiModalConfig.disable_mm_preprocessor_cache
    # LoRA fields
    enable_lora: bool = False
    enable_lora_bias: bool = LoRAConfig.bias_enabled
    max_loras: int = LoRAConfig.max_loras
    max_lora_rank: int = LoRAConfig.max_lora_rank
    fully_sharded_loras: bool = LoRAConfig.fully_sharded_loras
    max_cpu_loras: Optional[int] = LoRAConfig.max_cpu_loras
    lora_dtype: Optional[Union[str, torch.dtype]] = LoRAConfig.lora_dtype
    lora_extra_vocab_size: int = LoRAConfig.lora_extra_vocab_size
    long_lora_scaling_factors: Optional[tuple[float, ...]] = \
        LoRAConfig.long_lora_scaling_factors
    # PromptAdapter fields
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = PromptAdapterConfig.max_prompt_adapters
    max_prompt_adapter_token: int = \
        PromptAdapterConfig.max_prompt_adapter_token

    device: Device = DeviceConfig.device
    num_scheduler_steps: int = SchedulerConfig.num_scheduler_steps
    multi_step_stream_outputs: bool = SchedulerConfig.multi_step_stream_outputs
    ray_workers_use_nsight: bool = ParallelConfig.ray_workers_use_nsight
    num_gpu_blocks_override: Optional[
        int] = CacheConfig.num_gpu_blocks_override
    num_lookahead_slots: int = SchedulerConfig.num_lookahead_slots
    model_loader_extra_config: dict = \
        get_field(LoadConfig, "model_loader_extra_config")
    ignore_patterns: Optional[Union[str,
                                    List[str]]] = LoadConfig.ignore_patterns
    preemption_mode: Optional[str] = SchedulerConfig.preemption_mode

    scheduler_delay_factor: float = SchedulerConfig.delay_factor
    enable_chunked_prefill: Optional[
        bool] = SchedulerConfig.enable_chunked_prefill
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input

    guided_decoding_backend: GuidedDecodingBackend = DecodingConfig.backend
    guided_decoding_disable_fallback: bool = DecodingConfig.disable_fallback
    guided_decoding_disable_any_whitespace: bool = \
        DecodingConfig.disable_any_whitespace
    guided_decoding_disable_additional_properties: bool = \
        DecodingConfig.disable_additional_properties
    logits_processor_pattern: Optional[
        str] = ModelConfig.logits_processor_pattern

    speculative_config: Optional[Dict[str, Any]] = None

    qlora_adapter_name_or_path: Optional[str] = None
    show_hidden_metrics_for_version: Optional[str] = \
        ObservabilityConfig.show_hidden_metrics_for_version
    otlp_traces_endpoint: Optional[str] = \
        ObservabilityConfig.otlp_traces_endpoint
    collect_detailed_traces: Optional[list[DetailedTraceModules]] = \
        ObservabilityConfig.collect_detailed_traces
    disable_async_output_proc: bool = not ModelConfig.use_async_output_proc
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
    scheduler_cls: Union[str, Type[object]] = SchedulerConfig.scheduler_cls

    override_neuron_config: dict[str, Any] = \
        get_field(ModelConfig, "override_neuron_config")
    override_pooler_config: Optional[Union[dict, PoolerConfig]] = \
        ModelConfig.override_pooler_config
    compilation_config: Optional[CompilationConfig] = None
    worker_cls: str = ParallelConfig.worker_cls
    worker_extension_cls: str = ParallelConfig.worker_extension_cls

    kv_transfer_config: Optional[KVTransferConfig] = None
    kv_events_config: Optional[KVEventsConfig] = None

    generation_config: str = ModelConfig.generation_config
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode
    override_generation_config: dict[str, Any] = \
        get_field(ModelConfig, "override_generation_config")
    model_impl: str = ModelConfig.model_impl

    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales

    additional_config: Optional[Dict[str, Any]] = None
    enable_reasoning: Optional[bool] = None  # DEPRECATED
    reasoning_parser: str = DecodingConfig.reasoning_backend

    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load
    pt_load_map_location: str = LoadConfig.pt_load_map_location

    def __post_init__(self):
        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, (int, dict)):
            self.compilation_config = CompilationConfig.from_cli(
                str(self.compilation_config))
        if self.qlora_adapter_name_or_path is not None:
            warnings.warn(
                "The `qlora_adapter_name_or_path` is deprecated "
                "and will be removed in v0.10.0. ",
                DeprecationWarning,
                stacklevel=2,
            )
        # Setup plugins
        from vllm.plugins import load_general_plugins
        load_general_plugins()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # Model arguments
        model_kwargs = get_kwargs(ModelConfig)
        model_group = parser.add_argument_group(
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        if 'serve' not in sys.argv[1:] and '--help' not in sys.argv[1:]:
            model_group.add_argument("--model", **model_kwargs["model"])
        model_group.add_argument("--task", **model_kwargs["task"])
        model_group.add_argument("--tokenizer", **model_kwargs["tokenizer"])
        model_group.add_argument("--tokenizer-mode",
                                 **model_kwargs["tokenizer_mode"])
        model_group.add_argument("--trust-remote-code",
                                 **model_kwargs["trust_remote_code"])
        model_group.add_argument("--dtype", **model_kwargs["dtype"])
        model_group.add_argument("--seed", **model_kwargs["seed"])
        model_group.add_argument("--hf-config-path",
                                 **model_kwargs["hf_config_path"])
        model_group.add_argument("--allowed-local-media-path",
                                 **model_kwargs["allowed_local_media_path"])
        model_group.add_argument("--revision", **model_kwargs["revision"])
        model_group.add_argument("--code-revision",
                                 **model_kwargs["code_revision"])
        model_group.add_argument("--rope-scaling",
                                 **model_kwargs["rope_scaling"])
        model_group.add_argument("--rope-theta", **model_kwargs["rope_theta"])
        model_group.add_argument("--tokenizer-revision",
                                 **model_kwargs["tokenizer_revision"])
        model_group.add_argument("--max-model-len",
                                 **model_kwargs["max_model_len"])
        model_group.add_argument("--quantization", "-q",
                                 **model_kwargs["quantization"])
        model_group.add_argument("--enforce-eager",
                                 **model_kwargs["enforce_eager"])
        model_group.add_argument("--max-seq-len-to-capture",
                                 **model_kwargs["max_seq_len_to_capture"])
        model_group.add_argument("--max-logprobs",
                                 **model_kwargs["max_logprobs"])
        model_group.add_argument("--disable-sliding-window",
                                 **model_kwargs["disable_sliding_window"])
        model_group.add_argument("--disable-cascade-attn",
                                 **model_kwargs["disable_cascade_attn"])
        model_group.add_argument("--skip-tokenizer-init",
                                 **model_kwargs["skip_tokenizer_init"])
        model_group.add_argument("--enable-prompt-embeds",
                                 **model_kwargs["enable_prompt_embeds"])
        model_group.add_argument("--served-model-name",
                                 **model_kwargs["served_model_name"])
        # This one is a special case because it is the
        # opposite of ModelConfig.use_async_output_proc
        model_group.add_argument(
            "--disable-async-output-proc",
            action="store_true",
            default=EngineArgs.disable_async_output_proc,
            help="Disable async output processing. This may result in "
            "lower performance.")
        model_group.add_argument("--config-format",
                                 choices=[f.value for f in ConfigFormat],
                                 **model_kwargs["config_format"])
        # This one is a special case because it can bool
        # or str. TODO: Handle this in get_kwargs
        model_group.add_argument("--hf-token",
                                 type=str,
                                 nargs="?",
                                 const=True,
                                 default=model_kwargs["hf_token"]["default"],
                                 help=model_kwargs["hf_token"]["help"])
        model_group.add_argument("--hf-overrides",
                                 **model_kwargs["hf_overrides"])
        model_group.add_argument("--override-neuron-config",
                                 **model_kwargs["override_neuron_config"])
        model_group.add_argument("--override-pooler-config",
                                 **model_kwargs["override_pooler_config"])
        model_group.add_argument("--logits-processor-pattern",
                                 **model_kwargs["logits_processor_pattern"])
        model_group.add_argument("--generation-config",
                                 **model_kwargs["generation_config"])
        model_group.add_argument("--override-generation-config",
                                 **model_kwargs["override_generation_config"])
        model_group.add_argument("--enable-sleep-mode",
                                 **model_kwargs["enable_sleep_mode"])
        model_group.add_argument("--model-impl",
                                 choices=[f.value for f in ModelImpl],
                                 **model_kwargs["model_impl"])

        # Model loading arguments
        load_kwargs = get_kwargs(LoadConfig)
        load_group = parser.add_argument_group(
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument("--load-format",
                                choices=[f.value for f in LoadFormat],
                                **load_kwargs["load_format"])
        load_group.add_argument("--download-dir",
                                **load_kwargs["download_dir"])
        load_group.add_argument("--model-loader-extra-config",
                                **load_kwargs["model_loader_extra_config"])
        load_group.add_argument("--ignore-patterns",
                                **load_kwargs["ignore_patterns"])
        load_group.add_argument("--use-tqdm-on-load",
                                **load_kwargs["use_tqdm_on_load"])
        load_group.add_argument(
            "--qlora-adapter-name-or-path",
            type=str,
            default=None,
            help="The `--qlora-adapter-name-or-path` has no effect, do not set"
            " it, and it  will be removed in v0.10.0.",
            deprecated=True,
        )
        load_group.add_argument('--pt-load-map-location',
                                **load_kwargs["pt_load_map_location"])

        # Guided decoding arguments
        guided_decoding_kwargs = get_kwargs(DecodingConfig)
        guided_decoding_group = parser.add_argument_group(
            title="DecodingConfig",
            description=DecodingConfig.__doc__,
        )
        guided_decoding_group.add_argument("--guided-decoding-backend",
                                           **guided_decoding_kwargs["backend"])
        guided_decoding_group.add_argument(
            "--guided-decoding-disable-fallback",
            **guided_decoding_kwargs["disable_fallback"])
        guided_decoding_group.add_argument(
            "--guided-decoding-disable-any-whitespace",
            **guided_decoding_kwargs["disable_any_whitespace"])
        guided_decoding_group.add_argument(
            "--guided-decoding-disable-additional-properties",
            **guided_decoding_kwargs["disable_additional_properties"])
        guided_decoding_group.add_argument(
            "--enable-reasoning",
            action=argparse.BooleanOptionalAction,
            deprecated=True,
            help="[DEPRECATED] The `--enable-reasoning` flag is deprecated as "
            "of v0.8.6. Use `--reasoning-parser` to specify the reasoning "
            "parser backend instead. This flag (`--enable-reasoning`) will be "
            "removed in v0.10.0. When `--reasoning-parser` is specified, "
            "reasoning mode is automatically enabled.")
        guided_decoding_group.add_argument(
            "--reasoning-parser",
            # This choices is a special case because it's not static
            choices=list(ReasoningParserManager.reasoning_parsers),
            **guided_decoding_kwargs["reasoning_backend"])

        # Parallel arguments
        parallel_kwargs = get_kwargs(ParallelConfig)
        parallel_group = parser.add_argument_group(
            title="ParallelConfig",
            description=ParallelConfig.__doc__,
        )
        parallel_group.add_argument(
            "--distributed-executor-backend",
            **parallel_kwargs["distributed_executor_backend"])
        parallel_group.add_argument(
            "--pipeline-parallel-size", "-pp",
            **parallel_kwargs["pipeline_parallel_size"])
        parallel_group.add_argument("--tensor-parallel-size", "-tp",
                                    **parallel_kwargs["tensor_parallel_size"])
        parallel_group.add_argument("--data-parallel-size", "-dp",
                                    **parallel_kwargs["data_parallel_size"])
        parallel_group.add_argument('--data-parallel-size-local',
                                    '-dpl',
                                    type=int,
                                    help='Number of data parallel replicas '
                                    'to run on this node.')
        parallel_group.add_argument('--data-parallel-address',
                                    '-dpa',
                                    type=str,
                                    help='Address of data parallel cluster '
                                    'head-node.')
        parallel_group.add_argument('--data-parallel-rpc-port',
                                    '-dpp',
                                    type=int,
                                    help='Port for data parallel RPC '
                                    'communication.')
        parallel_group.add_argument(
            "--enable-expert-parallel",
            **parallel_kwargs["enable_expert_parallel"])
        parallel_group.add_argument(
            "--max-parallel-loading-workers",
            **parallel_kwargs["max_parallel_loading_workers"])
        parallel_group.add_argument(
            "--ray-workers-use-nsight",
            **parallel_kwargs["ray_workers_use_nsight"])
        parallel_group.add_argument(
            "--disable-custom-all-reduce",
            **parallel_kwargs["disable_custom_all_reduce"])
        parallel_group.add_argument("--worker-cls",
                                    **parallel_kwargs["worker_cls"])
        parallel_group.add_argument("--worker-extension-cls",
                                    **parallel_kwargs["worker_extension_cls"])

        # KV cache arguments
        cache_kwargs = get_kwargs(CacheConfig)
        cache_group = parser.add_argument_group(
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
        cache_group.add_argument("--gpu-memory-utilization",
                                 **cache_kwargs["gpu_memory_utilization"])
        cache_group.add_argument("--swap-space", **cache_kwargs["swap_space"])
        cache_group.add_argument("--kv-cache-dtype",
                                 **cache_kwargs["cache_dtype"])
        cache_group.add_argument("--num-gpu-blocks-override",
                                 **cache_kwargs["num_gpu_blocks_override"])
        cache_group.add_argument("--enable-prefix-caching",
                                 **cache_kwargs["enable_prefix_caching"])
        cache_group.add_argument("--prefix-caching-hash-algo",
                                 **cache_kwargs["prefix_caching_hash_algo"])
        cache_group.add_argument("--cpu-offload-gb",
                                 **cache_kwargs["cpu_offload_gb"])
        cache_group.add_argument("--calculate-kv-scales",
                                 **cache_kwargs["calculate_kv_scales"])

        # Tokenizer arguments
        tokenizer_kwargs = get_kwargs(TokenizerPoolConfig)
        tokenizer_group = parser.add_argument_group(
            title="TokenizerPoolConfig",
            description=TokenizerPoolConfig.__doc__,
        )
        tokenizer_group.add_argument("--tokenizer-pool-size",
                                     **tokenizer_kwargs["pool_size"])
        tokenizer_group.add_argument("--tokenizer-pool-type",
                                     **tokenizer_kwargs["pool_type"])
        tokenizer_group.add_argument("--tokenizer-pool-extra-config",
                                     **tokenizer_kwargs["extra_config"])

        # Multimodal related configs
        multimodal_kwargs = get_kwargs(MultiModalConfig)
        multimodal_group = parser.add_argument_group(
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument("--limit-mm-per-prompt",
                                      **multimodal_kwargs["limit_per_prompt"])
        multimodal_group.add_argument(
            "--mm-processor-kwargs",
            **multimodal_kwargs["mm_processor_kwargs"])
        multimodal_group.add_argument(
            "--disable-mm-preprocessor-cache",
            **multimodal_kwargs["disable_mm_preprocessor_cache"])

        # LoRA related configs
        lora_kwargs = get_kwargs(LoRAConfig)
        lora_group = parser.add_argument_group(
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            "--enable-lora",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of LoRA adapters.")
        lora_group.add_argument("--enable-lora-bias",
                                **lora_kwargs["bias_enabled"])
        lora_group.add_argument("--max-loras", **lora_kwargs["max_loras"])
        lora_group.add_argument("--max-lora-rank",
                                **lora_kwargs["max_lora_rank"])
        lora_group.add_argument("--lora-extra-vocab-size",
                                **lora_kwargs["lora_extra_vocab_size"])
        lora_group.add_argument(
            "--lora-dtype",
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument("--long-lora-scaling-factors",
                                **lora_kwargs["long_lora_scaling_factors"])
        lora_group.add_argument("--max-cpu-loras",
                                **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument("--fully-sharded-loras",
                                **lora_kwargs["fully_sharded_loras"])

        # PromptAdapter related configs
        prompt_adapter_kwargs = get_kwargs(PromptAdapterConfig)
        prompt_adapter_group = parser.add_argument_group(
            title="PromptAdapterConfig",
            description=PromptAdapterConfig.__doc__,
        )
        prompt_adapter_group.add_argument(
            "--enable-prompt-adapter",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of PromptAdapters.")
        prompt_adapter_group.add_argument(
            "--max-prompt-adapters",
            **prompt_adapter_kwargs["max_prompt_adapters"])
        prompt_adapter_group.add_argument(
            "--max-prompt-adapter-token",
            **prompt_adapter_kwargs["max_prompt_adapter_token"])

        # Device arguments
        device_kwargs = get_kwargs(DeviceConfig)
        device_group = parser.add_argument_group(
            title="DeviceConfig",
            description=DeviceConfig.__doc__,
        )
        device_group.add_argument("--device", **device_kwargs["device"])

        # Speculative arguments
        speculative_group = parser.add_argument_group(
            title="SpeculativeConfig",
            description=SpeculativeConfig.__doc__,
        )
        speculative_group.add_argument(
            "--speculative-config",
            type=json.loads,
            default=None,
            help="The configurations for speculative decoding. Should be a "
            "JSON string.")

        # Observability arguments
        observability_kwargs = get_kwargs(ObservabilityConfig)
        observability_group = parser.add_argument_group(
            title="ObservabilityConfig",
            description=ObservabilityConfig.__doc__,
        )
        observability_group.add_argument(
            "--show-hidden-metrics-for-version",
            **observability_kwargs["show_hidden_metrics_for_version"])
        observability_group.add_argument(
            "--otlp-traces-endpoint",
            **observability_kwargs["otlp_traces_endpoint"])
        # TODO: generalise this special case
        choices = observability_kwargs["collect_detailed_traces"]["choices"]
        metavar = f"{{{','.join(choices)}}}"
        observability_kwargs["collect_detailed_traces"]["metavar"] = metavar
        observability_kwargs["collect_detailed_traces"]["choices"] += [
            ",".join(p)
            for p in permutations(get_args(DetailedTraceModules), r=2)
        ]
        observability_group.add_argument(
            "--collect-detailed-traces",
            **observability_kwargs["collect_detailed_traces"])

        # Scheduler arguments
        scheduler_kwargs = get_kwargs(SchedulerConfig)
        scheduler_group = parser.add_argument_group(
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            "--max-num-batched-tokens",
            **scheduler_kwargs["max_num_batched_tokens"])
        scheduler_group.add_argument("--max-num-seqs",
                                     **scheduler_kwargs["max_num_seqs"])
        scheduler_group.add_argument(
            "--max-num-partial-prefills",
            **scheduler_kwargs["max_num_partial_prefills"])
        scheduler_group.add_argument(
            "--max-long-partial-prefills",
            **scheduler_kwargs["max_long_partial_prefills"])
        scheduler_group.add_argument('--cuda-graph-sizes',
                                     **scheduler_kwargs["cuda_graph_sizes"])
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"])
        scheduler_group.add_argument("--num-lookahead-slots",
                                     **scheduler_kwargs["num_lookahead_slots"])
        scheduler_group.add_argument("--scheduler-delay-factor",
                                     **scheduler_kwargs["delay_factor"])
        scheduler_group.add_argument("--preemption-mode",
                                     **scheduler_kwargs["preemption_mode"])
        scheduler_group.add_argument("--num-scheduler-steps",
                                     **scheduler_kwargs["num_scheduler_steps"])
        scheduler_group.add_argument(
            "--multi-step-stream-outputs",
            **scheduler_kwargs["multi_step_stream_outputs"])
        scheduler_group.add_argument("--scheduling-policy",
                                     **scheduler_kwargs["policy"])
        scheduler_group.add_argument(
            "--enable-chunked-prefill",
            **scheduler_kwargs["enable_chunked_prefill"])
        scheduler_group.add_argument(
            "--disable-chunked-mm-input",
            **scheduler_kwargs["disable_chunked_mm_input"])
        scheduler_group.add_argument("--scheduler-cls",
                                     **scheduler_kwargs["scheduler_cls"])

        # vLLM arguments
        vllm_kwargs = get_kwargs(VllmConfig)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        vllm_group.add_argument("--kv-transfer-config",
                                **vllm_kwargs["kv_transfer_config"])
        vllm_group.add_argument('--kv-events-config',
                                **vllm_kwargs["kv_events_config"])
        vllm_group.add_argument("--compilation-config", "-O",
                                **vllm_kwargs["compilation_config"])
        vllm_group.add_argument("--additional-config",
                                **vllm_kwargs["additional_config"])

        # Other arguments
        parser.add_argument('--use-v2-block-manager',
                            action='store_true',
                            default=True,
                            deprecated=True,
                            help='[DEPRECATED] block manager v1 has been '
                            'removed and SelfAttnBlockSpaceManager (i.e. '
                            'block manager v2) is now the default. '
                            'Setting this flag to True or False'
                            ' has no effect on vLLM behavior.')
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='Disable logging statistics.')

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_model_config(self) -> ModelConfig:
        # gguf file needs a specific model loader and doesn't use hf_repo
        if check_gguf_file(self.model):
            self.quantization = self.load_format = "gguf"

        # NOTE: This is to allow model loading from S3 in CI
        if (not isinstance(self, AsyncEngineArgs) and envs.VLLM_CI_USE_S3
                and self.model in MODELS_ON_S3
                and self.load_format == LoadFormat.AUTO):  # noqa: E501
            self.model = f"{MODEL_WEIGHTS_S3_BUCKET}/{self.model}"
            self.load_format = LoadFormat.RUNAI_STREAMER

        return ModelConfig(
            model=self.model,
            hf_config_path=self.hf_config_path,
            task=self.task,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            enforce_eager=self.enforce_eager,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            use_async_output_proc=not self.disable_async_output_proc,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            override_neuron_config=self.override_neuron_config,
            override_pooler_config=self.override_pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
        )

    def create_load_config(self) -> LoadConfig:

        if self.quantization == "bitsandbytes":
            self.load_format = "bitsandbytes"

        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
            use_tqdm_on_load=self.use_tqdm_on_load,
            pt_load_map_location=self.pt_load_map_location,
        )

    def create_speculative_config(
        self,
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
        enable_chunked_prefill: bool,
        disable_log_stats: bool,
    ) -> Optional["SpeculativeConfig"]:
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
        self.speculative_config.update({
            "target_model_config": target_model_config,
            "target_parallel_config": target_parallel_config,
            "enable_chunked_prefill": enable_chunked_prefill,
            "disable_log_stats": disable_log_stats,
        })
        speculative_config = SpeculativeConfig.from_dict(
            self.speculative_config)

        return speculative_config

    def create_engine_config(
        self,
        usage_context: Optional[UsageContext] = None,
    ) -> VllmConfig:
        """
        Create the VllmConfig.

        NOTE: for autoselection of V0 vs V1 engine, we need to
        create the ModelConfig first, since ModelConfig's attrs
        (e.g. the model arch) are needed to make the decision.

        This function set VLLM_USE_V1=X if VLLM_USE_V1 is
        unspecified by the user.

        If VLLM_USE_V1 is specified by the user but the VllmConfig
        is incompatible, we raise an error.
        """
        from vllm.platforms import current_platform
        current_platform.pre_register_and_update()

        device_config = DeviceConfig(device=self.device)
        model_config = self.create_model_config()

        # * If VLLM_USE_V1 is unset, we enable V1 for "supported features"
        #   and fall back to V0 for experimental or unsupported features.
        # * If VLLM_USE_V1=1, we enable V1 for supported + experimental
        #   features and raise error for unsupported features.
        # * If VLLM_USE_V1=0, we disable V1.
        use_v1 = False
        try_v1 = envs.VLLM_USE_V1 or not envs.is_set("VLLM_USE_V1")
        if try_v1 and self._is_v1_supported_oracle(model_config):
            use_v1 = True

        # If user explicitly set VLLM_USE_V1, sanity check we respect it.
        if envs.is_set("VLLM_USE_V1"):
            assert use_v1 == envs.VLLM_USE_V1
        # Otherwise, set the VLLM_USE_V1 variable globally.
        else:
            envs.set_vllm_use_v1(use_v1)

        # Set default arguments for V0 or V1 Engine.
        if use_v1:
            self._set_default_args_v1(usage_context)
        else:
            self._set_default_args_v0(model_config)

        assert self.enable_chunked_prefill is not None

        if envs.VLLM_ATTENTION_BACKEND in [STR_DUAL_CHUNK_FLASH_ATTN_VAL]:
            assert self.enforce_eager, (
                "Cuda graph is not supported with DualChunkFlashAttention. "
                "To run the model in eager mode, set 'enforce_eager=True' "
                "or use '--enforce-eager' in the CLI.")
            assert current_platform.is_cuda(), (
                "DualChunkFlashAttention is only supported on CUDA platform.")
            assert not use_v1, (
                "DualChunkFlashAttention is not supported on V1 engine. "
                "To run the model in V0 engine, try set 'VLLM_USE_V1=0'")

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            cpu_offload_gb=self.cpu_offload_gb,
            calculate_kv_scales=self.calculate_kv_scales,
        )

        # Get the current placement group if Ray is initialized and
        # we are in a Ray actor. If so, then the placement group will be
        # passed to spawned processes.
        placement_group = None
        if is_in_ray_actor():
            import ray

            # This call initializes Ray automatically if it is not initialized,
            # but we should not do this here.
            placement_group = ray.util.get_current_placement_group()

        # Local DP size defaults to global DP size if not set.
        data_parallel_size_local = self.data_parallel_size if (
            self.data_parallel_size_local
            is None) else self.data_parallel_size_local

        # DP address, used in multi-node case for torch distributed group
        # and ZMQ sockets.
        data_parallel_address = self.data_parallel_address if (
            self.data_parallel_address
            is not None) else ParallelConfig.data_parallel_master_ip

        # This port is only used when there are remote data parallel engines,
        # otherwise the local IPC transport is used.
        data_parallel_rpc_port = self.data_parallel_rpc_port if (
            self.data_parallel_rpc_port
            is not None) else ParallelConfig.data_parallel_rpc_port

        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            data_parallel_size=self.data_parallel_size,
            data_parallel_size_local=data_parallel_size_local,
            data_parallel_master_ip=data_parallel_address,
            data_parallel_rpc_port=data_parallel_rpc_port,
            enable_expert_parallel=self.enable_expert_parallel,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            placement_group=placement_group,
            distributed_executor_backend=self.distributed_executor_backend,
            worker_cls=self.worker_cls,
            worker_extension_cls=self.worker_extension_cls,
        )

        speculative_config = self.create_speculative_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_log_stats=self.disable_log_stats,
        )

        # Reminder: Please update docs/source/features/compatibility_matrix.md
        # If the feature combo become valid
        if self.num_scheduler_steps > 1:
            if speculative_config is not None:
                raise ValueError("Speculative decoding is not supported with "
                                 "multi-step (--num-scheduler-steps > 1)")
            if self.enable_chunked_prefill and self.pipeline_parallel_size > 1:
                raise ValueError("Multi-Step Chunked-Prefill is not supported "
                                 "for pipeline-parallel-size > 1")
            from vllm.platforms import current_platform
            if current_platform.is_cpu():
                logger.warning("Multi-Step (--num-scheduler-steps > 1) is "
                               "currently not supported for CPUs and has been "
                               "disabled.")
                self.num_scheduler_steps = 1

        # make sure num_lookahead_slots is set the higher value depending on
        # if we are using speculative decoding or multi-step
        num_lookahead_slots = max(self.num_lookahead_slots,
                                  self.num_scheduler_steps - 1)
        num_lookahead_slots = num_lookahead_slots \
            if speculative_config is None \
            else speculative_config.num_lookahead_slots

        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            cuda_graph_sizes=self.cuda_graph_sizes,
            num_lookahead_slots=num_lookahead_slots,
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
            is_multimodal_model=model_config.is_multimodal_model,
            preemption_mode=self.preemption_mode,
            num_scheduler_steps=self.num_scheduler_steps,
            multi_step_stream_outputs=self.multi_step_stream_outputs,
            send_delta_data=(envs.VLLM_USE_RAY_SPMD_WORKER
                             and parallel_config.use_ray),
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
        )

        lora_config = LoRAConfig(
            bias_enabled=self.enable_lora_bias,
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            fully_sharded_loras=self.fully_sharded_loras,
            lora_extra_vocab_size=self.lora_extra_vocab_size,
            long_lora_scaling_factors=self.long_lora_scaling_factors,
            lora_dtype=self.lora_dtype,
            max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras
            and self.max_cpu_loras > 0 else None) if self.enable_lora else None

        # bitsandbytes pre-quantized model need a specific model loader
        if model_config.quantization == "bitsandbytes":
            self.quantization = self.load_format = "bitsandbytes"

        load_config = self.create_load_config()

        prompt_adapter_config = PromptAdapterConfig(
            max_prompt_adapters=self.max_prompt_adapters,
            max_prompt_adapter_token=self.max_prompt_adapter_token) \
                                        if self.enable_prompt_adapter else None

        decoding_config = DecodingConfig(
            backend=self.guided_decoding_backend,
            disable_fallback=self.guided_decoding_disable_fallback,
            disable_any_whitespace=self.guided_decoding_disable_any_whitespace,
            disable_additional_properties=\
                self.guided_decoding_disable_additional_properties,
            reasoning_backend=self.reasoning_parser
        )

        observability_config = ObservabilityConfig(
            show_hidden_metrics_for_version=self.
            show_hidden_metrics_for_version,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_detailed_traces=self.collect_detailed_traces,
        )

        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            load_config=load_config,
            decoding_config=decoding_config,
            observability_config=observability_config,
            prompt_adapter_config=prompt_adapter_config,
            compilation_config=self.compilation_config,
            kv_transfer_config=self.kv_transfer_config,
            kv_events_config=self.kv_events_config,
            additional_config=self.additional_config,
        )

        return config

    def _is_v1_supported_oracle(self, model_config: ModelConfig) -> bool:
        """Oracle for whether to use V0 or V1 Engine by default."""

        #############################################################
        # Unsupported Feature Flags on V1.

        if (self.load_format == LoadFormat.TENSORIZER.value
                or self.load_format == LoadFormat.SHARDED_STATE.value):
            _raise_or_fallback(
                feature_name=f"--load_format {self.load_format}",
                recommend_to_remove=False)
            return False

        if (self.logits_processor_pattern
                != EngineArgs.logits_processor_pattern):
            _raise_or_fallback(feature_name="--logits-processor-pattern",
                               recommend_to_remove=False)
            return False

        if self.preemption_mode != SchedulerConfig.preemption_mode:
            _raise_or_fallback(feature_name="--preemption-mode",
                               recommend_to_remove=True)
            return False

        if (self.disable_async_output_proc
                != EngineArgs.disable_async_output_proc):
            _raise_or_fallback(feature_name="--disable-async-output-proc",
                               recommend_to_remove=True)
            return False

        if self.scheduling_policy != SchedulerConfig.policy:
            _raise_or_fallback(feature_name="--scheduling-policy",
                               recommend_to_remove=False)
            return False

        if self.num_scheduler_steps != SchedulerConfig.num_scheduler_steps:
            _raise_or_fallback(feature_name="--num-scheduler-steps",
                               recommend_to_remove=True)
            return False

        if self.scheduler_delay_factor != SchedulerConfig.delay_factor:
            _raise_or_fallback(feature_name="--scheduler-delay-factor",
                               recommend_to_remove=True)
            return False

        if self.guided_decoding_backend not in get_args(
                GuidedDecodingBackendV1):
            _raise_or_fallback(
                feature_name=
                f"--guided-decoding-backend={self.guided_decoding_backend}",
                recommend_to_remove=False)
            return False

        # Need at least Ampere for now (FA support required).
        # Skip this check if we are running on a non-GPU platform,
        # or if the device capability is not available
        # (e.g. in a Ray actor without GPUs).
        from vllm.platforms import current_platform
        if (current_platform.is_cuda()
                and current_platform.get_device_capability()
                and current_platform.get_device_capability().major < 8):
            _raise_or_fallback(feature_name="Compute Capability < 8.0",
                               recommend_to_remove=False)
            return False

        # No Fp8 KV cache so far.
        if self.kv_cache_dtype != "auto":
            fp8_attention = self.kv_cache_dtype.startswith("fp8")
            will_use_fa = (
                current_platform.is_cuda()
                and not envs.is_set("VLLM_ATTENTION_BACKEND")
            ) or envs.VLLM_ATTENTION_BACKEND == "FLASH_ATTN_VLLM_V1"
            supported = False
            if current_platform.is_rocm():
                supported = True
            elif fp8_attention and will_use_fa:
                from vllm.attention.utils.fa_utils import (
                    flash_attn_supports_fp8)
                supported = flash_attn_supports_fp8()
            if not supported:
                _raise_or_fallback(feature_name="--kv-cache-dtype",
                                   recommend_to_remove=False)
                return False

        # No Prompt Adapter so far.
        if self.enable_prompt_adapter:
            _raise_or_fallback(feature_name="--enable-prompt-adapter",
                               recommend_to_remove=False)
            return False

        # No text embedding inputs so far.
        if self.enable_prompt_embeds:
            _raise_or_fallback(feature_name="--enable-prompt-embeds",
                               recommend_to_remove=False)
            return False

        # Only Fp16 and Bf16 dtypes since we only support FA.
        V1_SUPPORTED_DTYPES = [torch.bfloat16, torch.float16]
        if model_config.dtype not in V1_SUPPORTED_DTYPES:
            _raise_or_fallback(feature_name=f"--dtype {model_config.dtype}",
                               recommend_to_remove=False)
            return False

        # Some quantization is not compatible with torch.compile.
        V1_UNSUPPORTED_QUANT = ["gguf"]
        if model_config.quantization in V1_UNSUPPORTED_QUANT:
            _raise_or_fallback(
                feature_name=f"--quantization {model_config.quantization}",
                recommend_to_remove=False)
            return False

        # No Embedding Models so far.
        if model_config.task not in ["generate"]:
            _raise_or_fallback(feature_name=f"--task {model_config.task}",
                               recommend_to_remove=False)
            return False

        # No Mamba or Encoder-Decoder so far.
        if not model_config.is_v1_compatible:
            _raise_or_fallback(feature_name=model_config.architectures,
                               recommend_to_remove=False)
            return False

        # No Concurrent Partial Prefills so far.
        if (self.max_num_partial_prefills
                != SchedulerConfig.max_num_partial_prefills
                or self.max_long_partial_prefills
                != SchedulerConfig.max_long_partial_prefills):
            _raise_or_fallback(feature_name="Concurrent Partial Prefill",
                               recommend_to_remove=False)
            return False

        # No OTLP observability so far.
        if (self.otlp_traces_endpoint or self.collect_detailed_traces):
            _raise_or_fallback(feature_name="--otlp-traces-endpoint",
                               recommend_to_remove=False)
            return False

        # Only Ngram speculative decoding so far.
        is_ngram_enabled = False
        is_eagle_enabled = False
        if self.speculative_config is not None:
            # This is supported but experimental (handled below).
            speculative_method = self.speculative_config.get("method")
            if speculative_method:
                if speculative_method in ("ngram", "[ngram]"):
                    is_ngram_enabled = True
                elif speculative_method in ("eagle", "eagle3"):
                    is_eagle_enabled = True
            else:
                speculative_model = self.speculative_config.get("model")
                if speculative_model in ("ngram", "[ngram]"):
                    is_ngram_enabled = True
            if not (is_ngram_enabled or is_eagle_enabled):
                # Other speculative decoding methods are not supported yet.
                _raise_or_fallback(feature_name="Speculative Decoding",
                                   recommend_to_remove=False)
                return False

        # No XFormers so far.
        V1_BACKENDS = [
            "FLASH_ATTN_VLLM_V1",
            "FLASH_ATTN",
            "PALLAS",
            "PALLAS_VLLM_V1",
            "TRITON_ATTN_VLLM_V1",
            "TRITON_MLA",
            "FLASHMLA",
            "FLASHINFER",
            "FLASHINFER_VLLM_V1",
            "ROCM_AITER_MLA",
        ]
        if (envs.is_set("VLLM_ATTENTION_BACKEND")
                and envs.VLLM_ATTENTION_BACKEND not in V1_BACKENDS):
            name = f"VLLM_ATTENTION_BACKEND={envs.VLLM_ATTENTION_BACKEND}"
            _raise_or_fallback(feature_name=name, recommend_to_remove=True)
            return False

        # Platforms must decide if they can support v1 for this model
        if not current_platform.supports_v1(model_config=model_config):
            _raise_or_fallback(
                feature_name=f"device type={current_platform.device_type}",
                recommend_to_remove=False)
            return False
        #############################################################
        # Experimental Features - allow users to opt in.

        # Signal Handlers requires running in main thread.
        if (threading.current_thread() != threading.main_thread()
                and _warn_or_fallback("Engine in background thread")):
            return False

        if (self.pipeline_parallel_size > 1
                and self.distributed_executor_backend not in ["ray", "mp"]):
            name = "Pipeline Parallelism without Ray distributed executor " \
                    "or multiprocessing executor"
            _raise_or_fallback(feature_name=name, recommend_to_remove=False)
            return False

        # ngram is supported on V1, but off by default for now.
        if is_ngram_enabled and _warn_or_fallback("ngram"):
            return False

        # Eagle is under development, so we don't support it yet.
        if is_eagle_enabled and _warn_or_fallback("Eagle"):
            return False

        # Non-[CUDA, TPU] may be supported on V1, but off by default for now.
        v0_hardware = not any(
            (current_platform.is_cuda(), current_platform.is_tpu()))
        if v0_hardware and _warn_or_fallback(  # noqa: SIM103
                current_platform.device_name):
            return False
        #############################################################

        return True

    def _set_default_args_v0(self, model_config: ModelConfig) -> None:
        """Set Default Arguments for V0 Engine."""

        max_model_len = model_config.max_model_len
        use_long_context = max_model_len > 32768
        if self.enable_chunked_prefill is None:
            # Chunked prefill not supported for Multimodal or MLA in V0.
            if model_config.is_multimodal_model or model_config.use_mla:
                self.enable_chunked_prefill = False

            # Enable chunked prefill by default for long context (> 32K)
            # models to avoid OOM errors in initial memory profiling phase.
            elif use_long_context:
                from vllm.platforms import current_platform
                is_gpu = current_platform.is_cuda()
                use_sliding_window = (model_config.get_sliding_window()
                                      is not None)
                use_spec_decode = self.speculative_config is not None

                if (is_gpu and not use_sliding_window and not use_spec_decode
                        and not self.enable_lora
                        and not self.enable_prompt_adapter
                        and model_config.runner_type != "pooling"):
                    self.enable_chunked_prefill = True
                    logger.warning(
                        "Chunked prefill is enabled by default for models "
                        "with max_model_len > 32K. Chunked prefill might "
                        "not work with some features or models. If you "
                        "encounter any issues, please disable by launching "
                        "with --enable-chunked-prefill=False.")

            if self.enable_chunked_prefill is None:
                self.enable_chunked_prefill = False

        if not self.enable_chunked_prefill and use_long_context:
            logger.warning(
                "The model has a long context length (%s). This may cause"
                "OOM during the initial memory profiling phase, or result "
                "in low performance due to small KV cache size. Consider "
                "setting --max-model-len to a smaller value.", max_model_len)
        elif (self.enable_chunked_prefill
              and model_config.runner_type == "pooling"):
            msg = "Chunked prefill is not supported for pooling models"
            raise ValueError(msg)

        # if using prefix caching, we must set a hash algo
        if self.enable_prefix_caching:
            # Disable prefix caching for multimodal models for VLLM_V0.
            if model_config.is_multimodal_model:
                logger.warning(
                    "--enable-prefix-caching is not supported for multimodal "
                    "models in V0 and has been disabled.")
                self.enable_prefix_caching = False

            # VLLM_V0 only supports builtin hash algo for prefix caching.
            if self.prefix_caching_hash_algo == "sha256":
                raise ValueError(
                    "sha256 is not supported for prefix caching in V0 engine. "
                    "Please use 'builtin'.")

        # Set max_num_seqs to 256 for VLLM_V0.
        if self.max_num_seqs is None:
            self.max_num_seqs = 256

    def _set_default_args_v1(self, usage_context: UsageContext) -> None:
        """Set Default Arguments for V1 Engine."""

        # V1 always uses chunked prefills.
        self.enable_chunked_prefill = True

        # V1 enables prefix caching by default.
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = True

        # V1 should use the new scheduler by default.
        # Swap it only if this arg is set to the original V0 default
        if self.scheduler_cls == EngineArgs.scheduler_cls:
            self.scheduler_cls = "vllm.v1.core.sched.scheduler.Scheduler"

        # When no user override, set the default values based on the usage
        # context.
        # Use different default values for different hardware.

        # Try to query the device name on the current platform. If it fails,
        # it may be because the platform that imports vLLM is not the same
        # as the platform that vLLM is running on (e.g. the case of scaling
        # vLLM with Ray) and has no GPUs. In this case we use the default
        # values for non-H100/H200 GPUs.
        from vllm.platforms import current_platform
        try:
            device_memory = current_platform.get_device_total_memory()
            device_name = current_platform.get_device_name().lower()
        except Exception:
            # This is only used to set default_max_num_batched_tokens
            device_memory = 0

        # NOTE(Kuntai): Setting large `max_num_batched_tokens` for A100 reduces
        # throughput, see PR #17885 for more details.
        # So here we do an extra device name check to prevent such regression.
        if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
            # For GPUs like H100 and MI300x, use larger default values.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
            default_max_num_seqs = 1024
        else:
            # TODO(woosuk): Tune the default values for other hardware.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
            default_max_num_seqs = 256

        # tpu specific default values.
        if current_platform.is_tpu():
            default_max_num_batched_tokens_tpu = {
                UsageContext.LLM_CLASS: {
                    'V6E': 2048,
                    'V5E': 1024,
                    'V5P': 512,
                },
                UsageContext.OPENAI_API_SERVER: {
                    'V6E': 1024,
                    'V5E': 512,
                    'V5P': 256,
                }
            }

        use_context_value = usage_context.value if usage_context else None
        if (self.max_num_batched_tokens is None
                and usage_context in default_max_num_batched_tokens):
            if current_platform.is_tpu():
                chip_name = current_platform.get_device_name()
                if chip_name in default_max_num_batched_tokens_tpu[
                        usage_context]:
                    self.max_num_batched_tokens = \
                        default_max_num_batched_tokens_tpu[
                            usage_context][chip_name]
                else:
                    self.max_num_batched_tokens = \
                        default_max_num_batched_tokens[usage_context]
            else:
                self.max_num_batched_tokens = default_max_num_batched_tokens[
                    usage_context]
            logger.debug(
                "Setting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens, use_context_value)

        if self.max_num_seqs is None:
            self.max_num_seqs = default_max_num_seqs

            logger.debug("Setting max_num_seqs to %d for %s usage context.",
                         self.max_num_seqs, use_context_value)


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser,
                     async_args_only: bool = False) -> FlexibleArgumentParser:
        # Initialize plugin to update the parser, for example, The plugin may
        # adding a new kind of quantization method to --quantization argument or
        # a new device to --device argument.
        load_general_plugins()
        if not async_args_only:
            parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='Disable logging requests.')
        from vllm.platforms import current_platform
        current_platform.pre_register_and_update(parser)
        return parser


def _raise_or_fallback(feature_name: str, recommend_to_remove: bool):
    if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
        raise NotImplementedError(
            f"VLLM_USE_V1=1 is not supported with {feature_name}.")
    msg = f"{feature_name} is not supported by the V1 Engine. "
    msg += "Falling back to V0. "
    if recommend_to_remove:
        msg += f"We recommend to remove {feature_name} from your config "
        msg += "in favor of the V1 Engine."
    logger.warning(msg)


def _warn_or_fallback(feature_name: str) -> bool:
    if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
        logger.warning(
            "Detected VLLM_USE_V1=1 with %s. Usage should "
            "be considered experimental. Please report any "
            "issues on Github.", feature_name)
        should_exit = False
    else:
        logger.info(
            "%s is experimental on VLLM_USE_V1=1. "
            "Falling back to V0 Engine.", feature_name)
        should_exit = True
    return should_exit


def human_readable_int(value):
    """Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    """
    value = value.strip()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([kKmMgGtT])', value)
    if match:
        decimal_multiplier = {
            'k': 10**3,
            'm': 10**6,
            'g': 10**9,
        }
        binary_multiplier = {
            'K': 2**10,
            'M': 2**20,
            'G': 2**30,
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
                raise argparse.ArgumentTypeError("Decimals are not allowed " \
                f"with binary suffixes like {suffix}. Did you mean to use " \
                f"{number}{suffix.lower()} instead?") from e

    # Regular plain number.
    return int(value)


# These functions are used by sphinx to build the documentation
def _engine_args_parser():
    return EngineArgs.add_cli_args(FlexibleArgumentParser())


def _async_engine_args_parser():
    return AsyncEngineArgs.add_cli_args(FlexibleArgumentParser(),
                                        async_args_only=True)
