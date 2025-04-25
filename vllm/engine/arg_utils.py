# SPDX-License-Identifier: Apache-2.0

# yapf: disable
import argparse
import dataclasses
import json
import re
import threading
from dataclasses import MISSING, dataclass, fields
from typing import (Any, Callable, Dict, List, Literal, Optional, Type,
                    TypeVar, Union, cast, get_args, get_origin)

import torch
from typing_extensions import TypeIs

import vllm.envs as envs
from vllm import version
from vllm.config import (BlockSize, CacheConfig, CacheDType, CompilationConfig,
                         ConfigFormat, ConfigType, DecodingConfig, Device,
                         DeviceConfig, DistributedExecutorBackend,
                         GuidedDecodingBackendV1, HfOverrides,
                         KVTransferConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ModelImpl, MultiModalConfig,
                         ObservabilityConfig, ParallelConfig, PoolerConfig,
                         PrefixCachingHashAlgo, PromptAdapterConfig,
                         SchedulerConfig, SchedulerPolicy, SpeculativeConfig,
                         TaskOption, TokenizerPoolConfig, VllmConfig,
                         get_attr_docs, get_field)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.plugins import load_general_plugins
from vllm.reasoning import ReasoningParserManager
from vllm.test_utils import MODEL_WEIGHTS_S3_BUCKET, MODELS_ON_S3
from vllm.transformers_utils.utils import check_gguf_file
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, GiB_bytes, is_in_ray_actor

# yapf: enable

logger = init_logger(__name__)

ALLOWED_DETAILED_TRACE_MODULES = ["model", "worker", "all"]

# object is used to allow for special typing forms
T = TypeVar("T")
TypeHint = Union[type[Any], object]
TypeHintT = Union[type[T], object]


def optional_arg(val: str, return_type: Callable[[str], T]) -> Optional[T]:
    if val == "" or val == "None":
        return None
    try:
        return return_type(val)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Value {val} cannot be converted to {return_type}.") from e


def optional_str(val: str) -> Optional[str]:
    return optional_arg(val, str)


def optional_int(val: str) -> Optional[int]:
    return optional_arg(val, int)


def optional_float(val: str) -> Optional[float]:
    return optional_arg(val, float)


def nullable_kvs(val: str) -> Optional[dict[str, int]]:
    """NOTE: This function is deprecated, args should be passed as JSON
    strings instead.
    
    Parses a string containing comma separate key [str] to value [int]
    pairs into a dictionary.

    Args:
        val: String value to be parsed.

    Returns:
        Dictionary with parsed values.
    """
    if len(val) == 0:
        return None

    out_dict: Dict[str, int] = {}
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


def optional_dict(val: str) -> Optional[dict[str, int]]:
    if re.match("^{.*}$", val):
        return optional_arg(val, json.loads)

    logger.warning(
        "Failed to parse JSON string. Attempting to parse as "
        "comma-separated key=value pairs. This will be deprecated in a "
        "future release.")
    return nullable_kvs(val)


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = 'facebook/opt-125m'
    served_model_name: Optional[Union[str, List[str]]] = None
    tokenizer: Optional[str] = None
    hf_config_path: Optional[str] = None
    task: TaskOption = "auto"
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    download_dir: Optional[str] = LoadConfig.download_dir
    load_format: str = LoadConfig.load_format
    config_format: ConfigFormat = ConfigFormat.AUTO
    dtype: str = 'auto'
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    seed: Optional[int] = None
    max_model_len: Optional[int] = None
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
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    max_parallel_loading_workers: Optional[
        int] = ParallelConfig.max_parallel_loading_workers
    block_size: Optional[BlockSize] = CacheConfig.block_size
    enable_prefix_caching: Optional[bool] = CacheConfig.enable_prefix_caching
    prefix_caching_hash_algo: PrefixCachingHashAlgo = \
        CacheConfig.prefix_caching_hash_algo
    disable_sliding_window: bool = False
    disable_cascade_attn: bool = False
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
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = None
    hf_token: Optional[Union[bool, str]] = None
    hf_overrides: Optional[HfOverrides] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
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
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    disable_mm_preprocessor_cache: bool = False
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

    guided_decoding_backend: str = DecodingConfig.guided_decoding_backend
    logits_processor_pattern: Optional[str] = None

    speculative_config: Optional[Dict[str, Any]] = None

    qlora_adapter_name_or_path: Optional[str] = None
    show_hidden_metrics_for_version: Optional[str] = None
    otlp_traces_endpoint: Optional[str] = None
    collect_detailed_traces: Optional[str] = None
    disable_async_output_proc: bool = False
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
    scheduler_cls: Union[str, Type[object]] = SchedulerConfig.scheduler_cls

    override_neuron_config: Optional[Dict[str, Any]] = None
    override_pooler_config: Optional[PoolerConfig] = None
    compilation_config: Optional[CompilationConfig] = None
    worker_cls: str = ParallelConfig.worker_cls
    worker_extension_cls: str = ParallelConfig.worker_extension_cls

    kv_transfer_config: Optional[KVTransferConfig] = None

    generation_config: Optional[str] = "auto"
    override_generation_config: Optional[Dict[str, Any]] = None
    enable_sleep_mode: bool = False
    model_impl: str = "auto"

    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales

    additional_config: Optional[Dict[str, Any]] = None
    enable_reasoning: Optional[bool] = None
    reasoning_parser: Optional[str] = DecodingConfig.reasoning_backend
    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = self.model

        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, (int, dict)):
            self.compilation_config = CompilationConfig.from_cli(
                str(self.compilation_config))

        # Setup plugins
        from vllm.plugins import load_general_plugins
        load_general_plugins()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        def is_type_in_union(cls: TypeHint, type: TypeHint) -> bool:
            """Check if the class is a type in a union type."""
            is_union = get_origin(cls) is Union
            type_in_union = type in [get_origin(a) or a for a in get_args(cls)]
            return is_union and type_in_union

        def get_type_from_union(cls: TypeHint, type: TypeHintT) -> TypeHintT:
            """Get the type in a union type."""
            for arg in get_args(cls):
                if (get_origin(arg) or arg) is type:
                    return arg
            raise ValueError(f"Type {type} not found in union type {cls}.")

        def is_optional(cls: TypeHint) -> TypeIs[Union[Any, None]]:
            """Check if the class is an optional type."""
            return is_type_in_union(cls, type(None))

        def can_be_type(cls: TypeHint, type: TypeHintT) -> TypeIs[TypeHintT]:
            """Check if the class can be of type."""
            return cls is type or get_origin(cls) is type or is_type_in_union(
                cls, type)

        def is_custom_type(cls: TypeHint) -> bool:
            """Check if the class is a custom type."""
            return cls.__module__ != "builtins"

        def get_kwargs(cls: ConfigType) -> dict[str, Any]:
            cls_docs = get_attr_docs(cls)
            kwargs = {}
            for field in fields(cls):
                # Get the default value of the field
                default = field.default
                if field.default_factory is not MISSING:
                    default = field.default_factory()

                # Get the help text for the field
                name = field.name
                help = cls_docs[name]
                # Escape % for argparse
                help = help.replace("%", "%%")

                # Initialise the kwargs dictionary for the field
                kwargs[name] = {"default": default, "help": help}

                # Make note of if the field is optional and get the actual
                # type of the field if it is
                optional = is_optional(field.type)
                field_type = get_args(
                    field.type)[0] if optional else field.type

                # Set type, action and choices for the field depending on the
                # type of the field
                if can_be_type(field_type, bool):
                    # Creates --no-<name> and --<name> flags
                    kwargs[name]["action"] = argparse.BooleanOptionalAction
                    kwargs[name]["type"] = bool
                elif can_be_type(field_type, Literal):
                    # Creates choices from Literal arguments
                    if is_type_in_union(field_type, Literal):
                        field_type = get_type_from_union(field_type, Literal)
                    choices = get_args(field_type)
                    kwargs[name]["choices"] = choices
                    choice_type = type(choices[0])
                    assert all(type(c) is choice_type for c in choices), (
                        "All choices must be of the same type. "
                        f"Got {choices} with types {[type(c) for c in choices]}"
                    )
                    kwargs[name]["type"] = choice_type
                elif can_be_type(field_type, tuple):
                    if is_type_in_union(field_type, tuple):
                        field_type = get_type_from_union(field_type, tuple)
                    dtypes = get_args(field_type)
                    dtype = dtypes[0]
                    assert all(
                        d is dtype for d in dtypes if d is not Ellipsis
                    ), ("All non-Ellipsis tuple elements must be of the same "
                        f"type. Got {dtypes}.")
                    kwargs[name]["type"] = dtype
                    kwargs[name]["nargs"] = "+"
                elif can_be_type(field_type, int):
                    kwargs[name]["type"] = optional_int if optional else int
                elif can_be_type(field_type, float):
                    kwargs[name][
                        "type"] = optional_float if optional else float
                elif can_be_type(field_type, dict):
                    kwargs[name]["type"] = optional_dict
                elif (can_be_type(field_type, str)
                      or is_custom_type(field_type)):
                    kwargs[name]["type"] = optional_str if optional else str
                else:
                    raise ValueError(
                        f"Unsupported type {field.type} for argument {name}. ")
            return kwargs

        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default=EngineArgs.model,
            help='Name or path of the huggingface model to use.')
        parser.add_argument(
            '--task',
            default=EngineArgs.task,
            choices=get_args(TaskOption),
            help='The task to use the model for. Each vLLM instance only '
            'supports one task, even if the same model can be used for '
            'multiple tasks. When the model only supports one task, ``"auto"`` '
            'can be used to select it; otherwise, you must specify explicitly '
            'which task to use.')
        parser.add_argument(
            '--tokenizer',
            type=optional_str,
            default=EngineArgs.tokenizer,
            help='Name or path of the huggingface tokenizer to use. '
            'If unspecified, model name or path will be used.')
        parser.add_argument(
            "--hf-config-path",
            type=optional_str,
            default=EngineArgs.hf_config_path,
            help='Name or path of the huggingface config to use. '
            'If unspecified, model name or path will be used.')
        parser.add_argument(
            '--skip-tokenizer-init',
            action='store_true',
            help='Skip initialization of tokenizer and detokenizer. '
            'Expects valid prompt_token_ids and None for prompt from '
            'the input. The generated output will contain token ids.')
        parser.add_argument(
            '--revision',
            type=optional_str,
            default=None,
            help='The specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--code-revision',
            type=optional_str,
            default=None,
            help='The specific revision to use for the model code on '
            'Hugging Face Hub. It can be a branch name, a tag name, or a '
            'commit id. If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=optional_str,
            default=None,
            help='Revision of the huggingface tokenizer to use. '
            'It can be a branch name, a tag name, or a commit id. '
            'If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-mode',
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=['auto', 'slow', 'mistral', 'custom'],
            help='The tokenizer mode.\n\n* "auto" will use the '
            'fast tokenizer if available.\n* "slow" will '
            'always use the slow tokenizer. \n* '
            '"mistral" will always use the `mistral_common` tokenizer. \n* '
            '"custom" will use --tokenizer to select the '
            'preregistered tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='Trust remote code from huggingface.')
        parser.add_argument(
            '--allowed-local-media-path',
            type=str,
            help="Allowing API requests to read local images or videos "
            "from directories specified by the server file system. "
            "This is a security risk. "
            "Should only be enabled in trusted environments.")
        # Model loading arguments
        load_kwargs = get_kwargs(LoadConfig)
        load_group = parser.add_argument_group(
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument('--load-format',
                                choices=[f.value for f in LoadFormat],
                                **load_kwargs["load_format"])
        load_group.add_argument('--download-dir',
                                **load_kwargs["download_dir"])
        load_group.add_argument('--model-loader-extra-config',
                                **load_kwargs["model_loader_extra_config"])
        load_group.add_argument('--use-tqdm-on-load',
                                **load_kwargs["use_tqdm_on_load"])

        parser.add_argument(
            '--config-format',
            default=EngineArgs.config_format,
            choices=[f.value for f in ConfigFormat],
            help='The format of the model config to load.\n\n'
            '* "auto" will try to load the config in hf format '
            'if available else it will try to load in mistral format ')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='Data type for model weights and activations.\n\n'
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            'BF16 precision for BF16 models.\n'
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.')
        parser.add_argument('--max-model-len',
                            type=human_readable_int,
                            default=EngineArgs.max_model_len,
                            help='Model context length. If unspecified, will '
                            'be automatically derived from the model config. '
                            'Supports k/m/g/K/M/G in human-readable format.\n'
                            'Examples:\n'
                            '- 1k → 1000\n'
                            '- 1K → 1024\n')

        # Guided decoding arguments
        guided_decoding_kwargs = get_kwargs(DecodingConfig)
        guided_decoding_group = parser.add_argument_group(
            title="DecodingConfig",
            description=DecodingConfig.__doc__,
        )
        guided_decoding_group.add_argument(
            '--guided-decoding-backend',
            **guided_decoding_kwargs["guided_decoding_backend"])
        guided_decoding_group.add_argument(
            "--reasoning-parser",
            # This choices is a special case because it's not static
            choices=list(ReasoningParserManager.reasoning_parsers),
            **guided_decoding_kwargs["reasoning_backend"])

        parser.add_argument(
            '--logits-processor-pattern',
            type=optional_str,
            default=None,
            help='Optional regex pattern specifying valid logits processor '
            'qualified names that can be passed with the `logits_processors` '
            'extra completion argument. Defaults to None, which allows no '
            'processors.')
        parser.add_argument(
            '--model-impl',
            type=str,
            default=EngineArgs.model_impl,
            choices=[f.value for f in ModelImpl],
            help='Which implementation of the model to use.\n\n'
            '* "auto" will try to use the vLLM implementation if it exists '
            'and fall back to the Transformers implementation if no vLLM '
            'implementation is available.\n'
            '* "vllm" will use the vLLM model implementation.\n'
            '* "transformers" will use the Transformers model '
            'implementation.\n')
        # Parallel arguments
        parallel_kwargs = get_kwargs(ParallelConfig)
        parallel_group = parser.add_argument_group(
            title="ParallelConfig",
            description=ParallelConfig.__doc__,
        )
        parallel_group.add_argument(
            '--distributed-executor-backend',
            **parallel_kwargs["distributed_executor_backend"])
        parallel_group.add_argument(
            '--pipeline-parallel-size', '-pp',
            **parallel_kwargs["pipeline_parallel_size"])
        parallel_group.add_argument('--tensor-parallel-size', '-tp',
                                    **parallel_kwargs["tensor_parallel_size"])
        parallel_group.add_argument('--data-parallel-size', '-dp',
                                    **parallel_kwargs["data_parallel_size"])
        parallel_group.add_argument(
            '--enable-expert-parallel',
            **parallel_kwargs["enable_expert_parallel"])
        parallel_group.add_argument(
            '--max-parallel-loading-workers',
            **parallel_kwargs["max_parallel_loading_workers"])
        parallel_group.add_argument(
            '--ray-workers-use-nsight',
            **parallel_kwargs["ray_workers_use_nsight"])
        parallel_group.add_argument(
            '--disable-custom-all-reduce',
            **parallel_kwargs["disable_custom_all_reduce"])

        # KV cache arguments
        cache_kwargs = get_kwargs(CacheConfig)
        cache_group = parser.add_argument_group(
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument('--block-size', **cache_kwargs["block_size"])
        cache_group.add_argument('--gpu-memory-utilization',
                                 **cache_kwargs["gpu_memory_utilization"])
        cache_group.add_argument('--swap-space', **cache_kwargs["swap_space"])
        cache_group.add_argument('--kv-cache-dtype',
                                 **cache_kwargs["cache_dtype"])
        cache_group.add_argument('--num-gpu-blocks-override',
                                 **cache_kwargs["num_gpu_blocks_override"])
        cache_group.add_argument("--enable-prefix-caching",
                                 **cache_kwargs["enable_prefix_caching"])
        cache_group.add_argument("--prefix-caching-hash-algo",
                                 **cache_kwargs["prefix_caching_hash_algo"])
        cache_group.add_argument('--cpu-offload-gb',
                                 **cache_kwargs["cpu_offload_gb"])
        cache_group.add_argument('--calculate-kv-scales',
                                 **cache_kwargs["calculate_kv_scales"])

        parser.add_argument('--disable-sliding-window',
                            action='store_true',
                            help='Disables sliding window, '
                            'capping to sliding window size.')
        parser.add_argument('--use-v2-block-manager',
                            action='store_true',
                            default=True,
                            help='[DEPRECATED] block manager v1 has been '
                            'removed and SelfAttnBlockSpaceManager (i.e. '
                            'block manager v2) is now the default. '
                            'Setting this flag to True or False'
                            ' has no effect on vLLM behavior.')

        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='Random seed for operations.')
        parser.add_argument(
            '--max-logprobs',
            type=int,
            default=EngineArgs.max_logprobs,
            help=('Max number of log probs to return logprobs is specified in'
                  ' SamplingParams.'))
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='Disable logging statistics.')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=optional_str,
                            choices=[*QUANTIZATION_METHODS, None],
                            default=EngineArgs.quantization,
                            help='Method used to quantize the weights. If '
                            'None, we first check the `quantization_config` '
                            'attribute in the model config file. If that is '
                            'None, we assume the model weights are not '
                            'quantized and use `dtype` to determine the data '
                            'type of the weights.')
        parser.add_argument(
            '--rope-scaling',
            default=None,
            type=json.loads,
            help='RoPE scaling configuration in JSON format. '
            'For example, ``{"rope_type":"dynamic","factor":2.0}``')
        parser.add_argument('--rope-theta',
                            default=None,
                            type=float,
                            help='RoPE theta. Use with `rope_scaling`. In '
                            'some cases, changing the RoPE theta improves the '
                            'performance of the scaled model.')
        parser.add_argument(
            '--hf-token',
            type=str,
            nargs='?',
            const=True,
            default=None,
            help='The token to use as HTTP bearer authorization'
            ' for remote files. If `True`, will use the token '
            'generated when running `huggingface-cli login` '
            '(stored in `~/.huggingface`).')
        parser.add_argument('--hf-overrides',
                            type=json.loads,
                            default=EngineArgs.hf_overrides,
                            help='Extra arguments for the HuggingFace config. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary.')
        parser.add_argument('--enforce-eager',
                            action='store_true',
                            help='Always use eager-mode PyTorch. If False, '
                            'will use eager mode and CUDA graph in hybrid '
                            'for maximal performance and flexibility.')
        parser.add_argument('--max-seq-len-to-capture',
                            type=int,
                            default=EngineArgs.max_seq_len_to_capture,
                            help='Maximum sequence length covered by CUDA '
                            'graphs. When a sequence has context length '
                            'larger than this, we fall back to eager mode. '
                            'Additionally for encoder-decoder models, if the '
                            'sequence length of the encoder input is larger '
                            'than this, we fall back to the eager mode.')

        # Tokenizer arguments
        tokenizer_kwargs = get_kwargs(TokenizerPoolConfig)
        tokenizer_group = parser.add_argument_group(
            title="TokenizerPoolConfig",
            description=TokenizerPoolConfig.__doc__,
        )
        tokenizer_group.add_argument('--tokenizer-pool-size',
                                     **tokenizer_kwargs["pool_size"])
        tokenizer_group.add_argument('--tokenizer-pool-type',
                                     **tokenizer_kwargs["pool_type"])
        tokenizer_group.add_argument('--tokenizer-pool-extra-config',
                                     **tokenizer_kwargs["extra_config"])

        # Multimodal related configs
        multimodal_kwargs = get_kwargs(MultiModalConfig)
        multimodal_group = parser.add_argument_group(
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument('--limit-mm-per-prompt',
                                      **multimodal_kwargs["limit_per_prompt"])

        parser.add_argument(
            '--mm-processor-kwargs',
            default=None,
            type=json.loads,
            help=('Overrides for the multi-modal processor obtained from '
                  '``AutoProcessor.from_pretrained``. The available overrides '
                  'depend on the model that is being run.'
                  'For example, for Phi-3-Vision: ``{"num_crops": 4}``.'))
        parser.add_argument(
            '--disable-mm-preprocessor-cache',
            action='store_true',
            help='If True, disable caching of the processed multi-modal '
            'inputs.')

        # LoRA related configs
        lora_kwargs = get_kwargs(LoRAConfig)
        lora_group = parser.add_argument_group(
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            '--enable-lora',
            action=argparse.BooleanOptionalAction,
            help='If True, enable handling of LoRA adapters.')
        lora_group.add_argument('--enable-lora-bias',
                                **lora_kwargs["bias_enabled"])
        lora_group.add_argument('--max-loras', **lora_kwargs["max_loras"])
        lora_group.add_argument('--max-lora-rank',
                                **lora_kwargs["max_lora_rank"])
        lora_group.add_argument('--lora-extra-vocab-size',
                                **lora_kwargs["lora_extra_vocab_size"])
        lora_group.add_argument(
            '--lora-dtype',
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument('--long-lora-scaling-factors',
                                **lora_kwargs["long_lora_scaling_factors"])
        lora_group.add_argument('--max-cpu-loras',
                                **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument('--fully-sharded-loras',
                                **lora_kwargs["fully_sharded_loras"])

        # PromptAdapter related configs
        prompt_adapter_kwargs = get_kwargs(PromptAdapterConfig)
        prompt_adapter_group = parser.add_argument_group(
            title="PromptAdapterConfig",
            description=PromptAdapterConfig.__doc__,
        )
        prompt_adapter_group.add_argument(
            '--enable-prompt-adapter',
            action=argparse.BooleanOptionalAction,
            help='If True, enable handling of PromptAdapters.')
        prompt_adapter_group.add_argument(
            '--max-prompt-adapters',
            **prompt_adapter_kwargs["max_prompt_adapters"])
        prompt_adapter_group.add_argument(
            '--max-prompt-adapter-token',
            **prompt_adapter_kwargs["max_prompt_adapter_token"])

        # Device arguments
        device_kwargs = get_kwargs(DeviceConfig)
        device_group = parser.add_argument_group(
            title="DeviceConfig",
            description=DeviceConfig.__doc__,
        )
        device_group.add_argument("--device", **device_kwargs["device"])

        parser.add_argument('--num-scheduler-steps',
                            type=int,
                            default=1,
                            help=('Maximum number of forward steps per '
                                  'scheduler call.'))

        # Speculative arguments
        speculative_group = parser.add_argument_group(
            title="SpeculativeConfig",
            description=SpeculativeConfig.__doc__,
        )
        speculative_group.add_argument(
            '--speculative-config',
            type=json.loads,
            default=None,
            help='The configurations for speculative decoding.'
            ' Should be a JSON string.')

        parser.add_argument(
            '--ignore-patterns',
            action="append",
            type=str,
            default=[],
            help="The pattern(s) to ignore when loading the model."
            "Default to `original/**/*` to avoid repeated loading of llama's "
            "checkpoints.")
        parser.add_argument(
            '--preemption-mode',
            type=str,
            default=None,
            help='If \'recompute\', the engine performs preemption by '
            'recomputing; If \'swap\', the engine performs preemption by '
            'block swapping.')

        parser.add_argument(
            "--served-model-name",
            nargs="+",
            type=str,
            default=None,
            help="The model name(s) used in the API. If multiple "
            "names are provided, the server will respond to any "
            "of the provided names. The model name in the model "
            "field of a response will be the first name in this "
            "list. If not specified, the model name will be the "
            "same as the ``--model`` argument. Noted that this name(s) "
            "will also be used in `model_name` tag content of "
            "prometheus metrics, if multiple names provided, metrics "
            "tag will take the first one.")
        parser.add_argument('--qlora-adapter-name-or-path',
                            type=str,
                            default=None,
                            help='Name or path of the QLoRA adapter.')

        parser.add_argument('--show-hidden-metrics-for-version',
                            type=str,
                            default=None,
                            help='Enable deprecated Prometheus metrics that '
                            'have been hidden since the specified version. '
                            'For example, if a previously deprecated metric '
                            'has been hidden since the v0.7.0 release, you '
                            'use --show-hidden-metrics-for-version=0.7 as a '
                            'temporary escape hatch while you migrate to new '
                            'metrics. The metric is likely to be removed '
                            'completely in an upcoming release.')

        parser.add_argument(
            '--otlp-traces-endpoint',
            type=str,
            default=None,
            help='Target URL to which OpenTelemetry traces will be sent.')
        parser.add_argument(
            '--collect-detailed-traces',
            type=str,
            default=None,
            help="Valid choices are " +
            ",".join(ALLOWED_DETAILED_TRACE_MODULES) +
            ". It makes sense to set this only if ``--otlp-traces-endpoint`` is"
            " set. If set, it will collect detailed traces for the specified "
            "modules. This involves use of possibly costly and or blocking "
            "operations and hence might have a performance impact.")

        parser.add_argument(
            '--disable-async-output-proc',
            action='store_true',
            default=EngineArgs.disable_async_output_proc,
            help="Disable async output processing. This may result in "
            "lower performance.")

        # Scheduler arguments
        scheduler_kwargs = get_kwargs(SchedulerConfig)
        scheduler_group = parser.add_argument_group(
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            '--max-num-batched-tokens',
            **scheduler_kwargs["max_num_batched_tokens"])
        scheduler_group.add_argument('--max-num-seqs',
                                     **scheduler_kwargs["max_num_seqs"])
        scheduler_group.add_argument(
            "--max-num-partial-prefills",
            **scheduler_kwargs["max_num_partial_prefills"])
        scheduler_group.add_argument(
            "--max-long-partial-prefills",
            **scheduler_kwargs["max_long_partial_prefills"])
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"])
        scheduler_group.add_argument('--num-lookahead-slots',
                                     **scheduler_kwargs["num_lookahead_slots"])
        scheduler_group.add_argument('--scheduler-delay-factor',
                                     **scheduler_kwargs["delay_factor"])
        scheduler_group.add_argument(
            '--enable-chunked-prefill',
            **scheduler_kwargs["enable_chunked_prefill"])
        scheduler_group.add_argument(
            '--multi-step-stream-outputs',
            **scheduler_kwargs["multi_step_stream_outputs"])
        scheduler_group.add_argument('--scheduling-policy',
                                     **scheduler_kwargs["policy"])
        scheduler_group.add_argument(
            "--disable-chunked-mm-input",
            **scheduler_kwargs["disable_chunked_mm_input"])
        parser.add_argument('--scheduler-cls',
                            **scheduler_kwargs["scheduler_cls"])

        parser.add_argument(
            '--override-neuron-config',
            type=json.loads,
            default=None,
            help="Override or set neuron device configuration. "
            "e.g. ``{\"cast_logits_dtype\": \"bloat16\"}``.")
        parser.add_argument(
            '--override-pooler-config',
            type=PoolerConfig.from_json,
            default=None,
            help="Override or set the pooling method for pooling models. "
            "e.g. ``{\"pooling_type\": \"mean\", \"normalize\": false}``.")

        parser.add_argument('--compilation-config',
                            '-O',
                            type=CompilationConfig.from_cli,
                            default=None,
                            help='torch.compile configuration for the model.'
                            'When it is a number (0, 1, 2, 3), it will be '
                            'interpreted as the optimization level.\n'
                            'NOTE: level 0 is the default level without '
                            'any optimization. level 1 and 2 are for internal '
                            'testing only. level 3 is the recommended level '
                            'for production.\n'
                            'To specify the full compilation config, '
                            'use a JSON string, e.g. ``{"level": 3, '
                            '"cudagraph_capture_sizes": [1, 2, 4, 8]}``\n'
                            'Following the convention of traditional '
                            'compilers, using ``-O`` without space is also '
                            'supported. ``-O3`` is equivalent to ``-O 3``.')

        parser.add_argument('--kv-transfer-config',
                            type=KVTransferConfig.from_cli,
                            default=None,
                            help='The configurations for distributed KV cache '
                            'transfer. Should be a JSON string.')

        parser.add_argument(
            '--worker-cls',
            type=str,
            default="auto",
            help='The worker class to use for distributed execution.')
        parser.add_argument(
            '--worker-extension-cls',
            type=str,
            default="",
            help='The worker extension class on top of the worker cls, '
            'it is useful if you just want to add new functions to the worker '
            'class without changing the existing functions.')
        parser.add_argument(
            "--generation-config",
            type=optional_str,
            default="auto",
            help="The folder path to the generation config. "
            "Defaults to 'auto', the generation config will be loaded from "
            "model path. If set to 'vllm', no generation config is loaded, "
            "vLLM defaults will be used. If set to a folder path, the "
            "generation config will be loaded from the specified folder path. "
            "If `max_new_tokens` is specified in generation config, then "
            "it sets a server-wide limit on the number of output tokens "
            "for all requests.")

        parser.add_argument(
            "--override-generation-config",
            type=json.loads,
            default=None,
            help="Overrides or sets generation config in JSON format. "
            "e.g. ``{\"temperature\": 0.5}``. If used with "
            "--generation-config=auto, the override parameters will be merged "
            "with the default config from the model. If generation-config is "
            "None, only the override parameters are used.")

        parser.add_argument("--enable-sleep-mode",
                            action="store_true",
                            default=False,
                            help="Enable sleep mode for the engine. "
                            "(only cuda platform is supported)")

        parser.add_argument(
            "--additional-config",
            type=json.loads,
            default=None,
            help="Additional config for specified platform in JSON format. "
            "Different platforms may support different configs. Make sure the "
            "configs are valid for the platform you are using. The input format"
            " is like '{\"config_key\":\"config_value\"}'")

        parser.add_argument(
            "--enable-reasoning",
            action="store_true",
            default=False,
            help="Whether to enable reasoning_content for the model. "
            "If enabled, the model will be able to generate reasoning content."
        )

        parser.add_argument(
            "--disable-cascade-attn",
            action="store_true",
            default=False,
            help="Disable cascade attention for V1. While cascade attention "
            "does not change the mathematical correctness, disabling it "
            "could be useful for preventing potential numerical issues. "
            "Note that even if this is set to False, cascade attention will be "
            "only used when the heuristic tells that it's beneficial.")

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
            # We know this is not None because we set it in __post_init__
            tokenizer=cast(str, self.tokenizer),
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

        if(self.qlora_adapter_name_or_path is not None) and \
            self.quantization != "bitsandbytes":
            raise ValueError(
                "QLoRA adapter only support "
                f"'bitsandbytes' quantization, but got {self.quantization}")

        if self.quantization == "bitsandbytes":
            self.load_format = "bitsandbytes"
        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
            use_tqdm_on_load=self.use_tqdm_on_load,
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

        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            data_parallel_size=self.data_parallel_size,
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

        if self.qlora_adapter_name_or_path is not None and \
            self.qlora_adapter_name_or_path != "":
            self.model_loader_extra_config[
                "qlora_adapter_name_or_path"] = self.qlora_adapter_name_or_path

        # bitsandbytes pre-quantized model need a specific model loader
        if model_config.quantization == "bitsandbytes":
            self.quantization = self.load_format = "bitsandbytes"

        load_config = self.create_load_config()

        prompt_adapter_config = PromptAdapterConfig(
            max_prompt_adapters=self.max_prompt_adapters,
            max_prompt_adapter_token=self.max_prompt_adapter_token) \
                                        if self.enable_prompt_adapter else None

        decoding_config = DecodingConfig(
            guided_decoding_backend=self.guided_decoding_backend,
            reasoning_backend=self.reasoning_parser
            if self.enable_reasoning else None,
        )

        show_hidden_metrics = False
        if self.show_hidden_metrics_for_version is not None:
            show_hidden_metrics = version._prev_minor_version_was(
                self.show_hidden_metrics_for_version)

        detailed_trace_modules = []
        if self.collect_detailed_traces is not None:
            detailed_trace_modules = self.collect_detailed_traces.split(",")
        for m in detailed_trace_modules:
            if m not in ALLOWED_DETAILED_TRACE_MODULES:
                raise ValueError(
                    f"Invalid module {m} in collect_detailed_traces. "
                    f"Valid modules are {ALLOWED_DETAILED_TRACE_MODULES}")
        observability_config = ObservabilityConfig(
            show_hidden_metrics=show_hidden_metrics,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_model_forward_time="model" in detailed_trace_modules
            or "all" in detailed_trace_modules,
            collect_model_execute_time="worker" in detailed_trace_modules
            or "all" in detailed_trace_modules,
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

        # remove backend options when doing this check
        if self.guided_decoding_backend.split(':')[0] \
            not in get_args(GuidedDecodingBackendV1):
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
            if fp8_attention and will_use_fa:
                from vllm.vllm_flash_attn.fa_utils import (
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
                elif speculative_method == "eagle":
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

        # PP is supported on V1 with Ray distributed executor,
        # but off for MP distributed executor for now.
        if (self.pipeline_parallel_size > 1
                and self.distributed_executor_backend != "ray"):
            name = "Pipeline Parallelism without Ray distributed executor"
            _raise_or_fallback(feature_name=name, recommend_to_remove=False)
            return False

        # ngram is supported on V1, but off by default for now.
        if is_ngram_enabled and _warn_or_fallback("ngram"):
            return False

        # Eagle is under development, so we don't support it yet.
        if is_eagle_enabled and _warn_or_fallback("Eagle"):
            return False

        # Non-CUDA is supported on V1, but off by default for now.
        not_cuda = not current_platform.is_cuda()
        if not_cuda and _warn_or_fallback(  # noqa: SIM103
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
        try:
            from vllm.platforms import current_platform
            device_memory = current_platform.get_device_total_memory()
        except Exception:
            # This is only used to set default_max_num_batched_tokens
            device_memory = 0

        if device_memory >= 70 * GiB_bytes:
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

        use_context_value = usage_context.value if usage_context else None
        if (self.max_num_batched_tokens is None
                and usage_context in default_max_num_batched_tokens):
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
