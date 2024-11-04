import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional,
                    Tuple, Type, Union, cast, get_args)

import torch

import vllm.envs as envs
from vllm.config import (CacheConfig, ConfigFormat, DecodingConfig,
                         DeviceConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ObservabilityConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig, TaskOption, TokenizerPoolConfig,
                         VllmConfig)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.transformers_utils.utils import check_gguf_file
from vllm.utils import FlexibleArgumentParser, StoreBoolean

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

logger = init_logger(__name__)

ALLOWED_DETAILED_TRACE_MODULES = ["model", "worker", "all"]

DEVICE_OPTIONS = [
    "auto",
    "cuda",
    "neuron",
    "cpu",
    "openvino",
    "tpu",
    "xpu",
]


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


def nullable_kvs(val: str) -> Optional[Mapping[str, int]]:
    """Parses a string containing comma separate key [str] to value [int]
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


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = 'facebook/opt-125m'
    served_model_name: Optional[Union[str, List[str]]] = None
    tokenizer: Optional[str] = None
    task: TaskOption = "auto"
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    chat_template_text_format: str = 'string'
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    config_format: ConfigFormat = ConfigFormat.AUTO
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    quantization_param_path: Optional[str] = None
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[str,
                                                 Type[ExecutorBase]]] = None
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    enable_prefix_caching: bool = False
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = True
    swap_space: float = 4  # GiB
    cpu_offload_gb: float = 0  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]] = "ray"
    tokenizer_pool_extra_config: Optional[dict] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = 1
    max_prompt_adapter_token: int = 0
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    lora_dtype: Optional[Union[str, torch.dtype]] = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    num_scheduler_steps: int = 1
    multi_step_stream_outputs: bool = True
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None
    preemption_mode: Optional[str] = None

    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None

    guided_decoding_backend: str = 'outlines'
    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    speculative_model_quantization: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    speculative_disable_mqa_scorer: Optional[bool] = False
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None
    qlora_adapter_name_or_path: Optional[str] = None
    disable_logprobs_during_spec_decoding: Optional[bool] = None

    otlp_traces_endpoint: Optional[str] = None
    collect_detailed_traces: Optional[str] = None
    disable_async_output_proc: bool = False
    override_neuron_config: Optional[Dict[str, Any]] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    scheduling_policy: Literal["fcfs", "priority"] = "fcfs"

    # Pooling configuration.
    pooling_type: Optional[str] = None
    pooling_norm: Optional[bool] = None
    pooling_softmax: Optional[bool] = None
    pooling_step_tag_id: Optional[int] = None
    pooling_returned_token_ids: Optional[List[int]] = None

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = self.model

        # Setup plugins
        from vllm.plugins import load_general_plugins
        load_general_plugins()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

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
            'multiple tasks. When the model only supports one task, "auto" '
            'can be used to select it; otherwise, you must specify explicitly '
            'which task to use.')
        parser.add_argument(
            '--tokenizer',
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help='Name or path of the huggingface tokenizer to use. '
            'If unspecified, model name or path will be used.')
        parser.add_argument(
            '--skip-tokenizer-init',
            action='store_true',
            help='Skip initialization of tokenizer and detokenizer')
        parser.add_argument(
            '--revision',
            type=nullable_str,
            default=None,
            help='The specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--code-revision',
            type=nullable_str,
            default=None,
            help='The specific revision to use for the model code on '
            'Hugging Face Hub. It can be a branch name, a tag name, or a '
            'commit id. If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=nullable_str,
            default=None,
            help='Revision of the huggingface tokenizer to use. '
            'It can be a branch name, a tag name, or a commit id. '
            'If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-mode',
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=['auto', 'slow', 'mistral'],
            help='The tokenizer mode.\n\n* "auto" will use the '
            'fast tokenizer if available.\n* "slow" will '
            'always use the slow tokenizer. \n* '
            '"mistral" will always use the `mistral_common` tokenizer.')
        parser.add_argument(
            '--chat-template-text-format',
            type=str,
            default=EngineArgs.chat_template_text_format,
            choices=['string', 'openai'],
            help='The format to render text content within a chat template. '
            '"string" will keep the content field as a string whereas '
            '"openai" will parse content in the current OpenAI format.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='Trust remote code from huggingface.')
        parser.add_argument(
            '--allowed-local-media-path',
            type=str,
            help="Allowing API requests to read local images or videos"
            "from directories specified by the server file system."
            "This is a security risk."
            "Should only be enabled in trusted environments")
        parser.add_argument('--download-dir',
                            type=nullable_str,
                            default=EngineArgs.download_dir,
                            help='Directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface.')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=[f.value for f in LoadFormat],
            help='The format of the model weights to load.\n\n'
            '* "auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available.\n'
            '* "pt" will load the weights in the pytorch bin format.\n'
            '* "safetensors" will load the weights in the safetensors format.\n'
            '* "npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading.\n'
            '* "dummy" will initialize the weights with random values, '
            'which is mainly for profiling.\n'
            '* "tensorizer" will load the weights using tensorizer from '
            'CoreWeave. See the Tensorize vLLM Model script in the Examples '
            'section for more information.\n'
            '* "bitsandbytes" will load the weights using bitsandbytes '
            'quantization.\n')
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
        parser.add_argument(
            '--kv-cache-dtype',
            type=str,
            choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
            'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
        parser.add_argument(
            '--quantization-param-path',
            type=nullable_str,
            default=None,
            help='Path to the JSON file containing the KV cache '
            'scaling factors. This should generally be supplied, when '
            'KV cache dtype is FP8. Otherwise, KV cache scaling factors '
            'default to 1.0, which may cause accuracy issues. '
            'FP8_E5M2 (without scaling) is only supported on cuda version'
            'greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead '
            'supported for common inference criteria.')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=EngineArgs.max_model_len,
                            help='Model context length. If unspecified, will '
                            'be automatically derived from the model config.')
        parser.add_argument(
            '--guided-decoding-backend',
            type=str,
            default='outlines',
            choices=['outlines', 'lm-format-enforcer'],
            help='Which engine will be used for guided decoding'
            ' (JSON schema / regex etc) by default. Currently support '
            'https://github.com/outlines-dev/outlines and '
            'https://github.com/noamgat/lm-format-enforcer.'
            ' Can be overridden per request via guided_decoding_backend'
            ' parameter.')
        # Parallel arguments
        parser.add_argument(
            '--distributed-executor-backend',
            choices=['ray', 'mp'],
            default=EngineArgs.distributed_executor_backend,
            help='Backend to use for distributed serving. When more than 1 GPU '
            'is used, will be automatically set to "ray" if installed '
            'or "mp" (multiprocessing) otherwise.')
        parser.add_argument(
            '--worker-use-ray',
            action='store_true',
            help='Deprecated, use --distributed-executor-backend=ray.')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='Number of pipeline stages.')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='Number of tensor parallel replicas.')
        parser.add_argument(
            '--max-parallel-loading-workers',
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help='Load model sequentially in multiple batches, '
            'to avoid RAM OOM when using tensor '
            'parallel and large models.')
        parser.add_argument(
            '--ray-workers-use-nsight',
            action='store_true',
            help='If specified, use nsight to profile Ray workers.')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='Token block size for contiguous chunks of '
                            'tokens. This is ignored on neuron devices and '
                            'set to max-model-len')

        parser.add_argument('--enable-prefix-caching',
                            action='store_true',
                            help='Enables automatic prefix caching.')
        parser.add_argument('--disable-sliding-window',
                            action='store_true',
                            help='Disables sliding window, '
                            'capping to sliding window size')
        parser.add_argument('--use-v2-block-manager',
                            action='store_true',
                            help='[DEPRECATED] block manager v1 has been '
                            'removed and SelfAttnBlockSpaceManager (i.e. '
                            'block manager v2) is now the default. '
                            'Setting this flag to True or False'
                            ' has no effect on vLLM behavior.')
        parser.add_argument(
            '--num-lookahead-slots',
            type=int,
            default=EngineArgs.num_lookahead_slots,
            help='Experimental scheduling config necessary for '
            'speculative decoding. This will be replaced by '
            'speculative config in the future; it is present '
            'to enable correctness tests until then.')

        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='Random seed for operations.')
        parser.add_argument('--swap-space',
                            type=float,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU.')
        parser.add_argument(
            '--cpu-offload-gb',
            type=float,
            default=0,
            help='The space in GiB to offload to CPU, per GPU. '
            'Default is 0, which means no offloading. Intuitively, '
            'this argument can be seen as a virtual way to increase '
            'the GPU memory size. For example, if you have one 24 GB '
            'GPU and set this to 10, virtually you can think of it as '
            'a 34 GB GPU. Then you can load a 13B model with BF16 weight,'
            'which requires at least 26GB GPU memory. Note that this '
            'requires fast CPU-GPU interconnect, as part of the model is'
            'loaded from CPU memory to GPU memory on the fly in each '
            'model forward pass.')
        parser.add_argument(
            '--gpu-memory-utilization',
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help='The fraction of GPU memory to be used for the model '
            'executor, which can range from 0 to 1. For example, a value of '
            '0.5 would imply 50%% GPU memory utilization. If unspecified, '
            'will use the default value of 0.9. This is a global gpu memory '
            'utilization limit, for example if 50%% of the gpu memory is '
            'already used before vLLM starts and --gpu-memory-utilization is '
            'set to 0.9, then only 40%% of the gpu memory will be allocated '
            'to the model executor.')
        parser.add_argument(
            '--num-gpu-blocks-override',
            type=int,
            default=None,
            help='If specified, ignore GPU profiling result and use this number'
            'of GPU blocks. Used for testing preemption.')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='Maximum number of batched tokens per '
                            'iteration.')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='Maximum number of sequences per iteration.')
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
                            type=nullable_str,
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
            'For example, {"rope_type":"dynamic","factor":2.0}')
        parser.add_argument('--rope-theta',
                            default=None,
                            type=float,
                            help='RoPE theta. Use with `rope_scaling`. In '
                            'some cases, changing the RoPE theta improves the '
                            'performance of the scaled model.')
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
        parser.add_argument('--disable-custom-all-reduce',
                            action='store_true',
                            default=EngineArgs.disable_custom_all_reduce,
                            help='See ParallelConfig.')
        parser.add_argument('--tokenizer-pool-size',
                            type=int,
                            default=EngineArgs.tokenizer_pool_size,
                            help='Size of tokenizer pool to use for '
                            'asynchronous tokenization. If 0, will '
                            'use synchronous tokenization.')
        parser.add_argument('--tokenizer-pool-type',
                            type=str,
                            default=EngineArgs.tokenizer_pool_type,
                            help='Type of tokenizer pool to use for '
                            'asynchronous tokenization. Ignored '
                            'if tokenizer_pool_size is 0.')
        parser.add_argument('--tokenizer-pool-extra-config',
                            type=nullable_str,
                            default=EngineArgs.tokenizer_pool_extra_config,
                            help='Extra config for tokenizer pool. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary. Ignored if '
                            'tokenizer_pool_size is 0.')

        # Multimodal related configs
        parser.add_argument(
            '--limit-mm-per-prompt',
            type=nullable_kvs,
            default=EngineArgs.limit_mm_per_prompt,
            # The default value is given in
            # MultiModalRegistry.init_mm_limits_per_prompt
            help=('For each multimodal plugin, limit how many '
                  'input instances to allow for each prompt. '
                  'Expects a comma-separated list of items, '
                  'e.g.: `image=16,video=2` allows a maximum of 16 '
                  'images and 2 videos per prompt. Defaults to 1 for '
                  'each modality.'))
        parser.add_argument(
            '--mm-processor-kwargs',
            default=None,
            type=json.loads,
            help=('Overrides for the multimodal input mapping/processing,'
                  'e.g., image processor. For example: {"num_crops": 4}.'))

        # LoRA related configs
        parser.add_argument('--enable-lora',
                            action='store_true',
                            help='If True, enable handling of LoRA adapters.')
        parser.add_argument('--max-loras',
                            type=int,
                            default=EngineArgs.max_loras,
                            help='Max number of LoRAs in a single batch.')
        parser.add_argument('--max-lora-rank',
                            type=int,
                            default=EngineArgs.max_lora_rank,
                            help='Max LoRA rank.')
        parser.add_argument(
            '--lora-extra-vocab-size',
            type=int,
            default=EngineArgs.lora_extra_vocab_size,
            help=('Maximum size of extra vocabulary that can be '
                  'present in a LoRA adapter (added to the base '
                  'model vocabulary).'))
        parser.add_argument(
            '--lora-dtype',
            type=str,
            default=EngineArgs.lora_dtype,
            choices=['auto', 'float16', 'bfloat16', 'float32'],
            help=('Data type for LoRA. If auto, will default to '
                  'base model dtype.'))
        parser.add_argument(
            '--long-lora-scaling-factors',
            type=nullable_str,
            default=EngineArgs.long_lora_scaling_factors,
            help=('Specify multiple scaling factors (which can '
                  'be different from base model scaling factor '
                  '- see eg. Long LoRA) to allow for multiple '
                  'LoRA adapters trained with those scaling '
                  'factors to be used at the same time. If not '
                  'specified, only adapters trained with the '
                  'base model scaling factor are allowed.'))
        parser.add_argument(
            '--max-cpu-loras',
            type=int,
            default=EngineArgs.max_cpu_loras,
            help=('Maximum number of LoRAs to store in CPU memory. '
                  'Must be >= than max_num_seqs. '
                  'Defaults to max_num_seqs.'))
        parser.add_argument(
            '--fully-sharded-loras',
            action='store_true',
            help=('By default, only half of the LoRA computation is '
                  'sharded with tensor parallelism. '
                  'Enabling this will use the fully sharded layers. '
                  'At high sequence length, max rank or '
                  'tensor parallel size, this is likely faster.'))
        parser.add_argument('--enable-prompt-adapter',
                            action='store_true',
                            help='If True, enable handling of PromptAdapters.')
        parser.add_argument('--max-prompt-adapters',
                            type=int,
                            default=EngineArgs.max_prompt_adapters,
                            help='Max number of PromptAdapters in a batch.')
        parser.add_argument('--max-prompt-adapter-token',
                            type=int,
                            default=EngineArgs.max_prompt_adapter_token,
                            help='Max number of PromptAdapters tokens')
        parser.add_argument("--device",
                            type=str,
                            default=EngineArgs.device,
                            choices=DEVICE_OPTIONS,
                            help='Device type for vLLM execution.')
        parser.add_argument('--num-scheduler-steps',
                            type=int,
                            default=1,
                            help=('Maximum number of forward steps per '
                                  'scheduler call.'))

        parser.add_argument(
            '--multi-step-stream-outputs',
            action=StoreBoolean,
            default=EngineArgs.multi_step_stream_outputs,
            nargs="?",
            const="True",
            help='If False, then multi-step will stream outputs at the end '
            'of all steps')
        parser.add_argument(
            '--scheduler-delay-factor',
            type=float,
            default=EngineArgs.scheduler_delay_factor,
            help='Apply a delay (of delay factor multiplied by previous '
            'prompt latency) before scheduling next prompt.')
        parser.add_argument(
            '--enable-chunked-prefill',
            action=StoreBoolean,
            default=EngineArgs.enable_chunked_prefill,
            nargs="?",
            const="True",
            help='If set, the prefill requests can be chunked based on the '
            'max_num_batched_tokens.')

        parser.add_argument(
            '--speculative-model',
            type=nullable_str,
            default=EngineArgs.speculative_model,
            help=
            'The name of the draft model to be used in speculative decoding.')
        # Quantization settings for speculative model.
        parser.add_argument(
            '--speculative-model-quantization',
            type=nullable_str,
            choices=[*QUANTIZATION_METHODS, None],
            default=EngineArgs.speculative_model_quantization,
            help='Method used to quantize the weights of speculative model. '
            'If None, we first check the `quantization_config` '
            'attribute in the model config file. If that is '
            'None, we assume the model weights are not '
            'quantized and use `dtype` to determine the data '
            'type of the weights.')
        parser.add_argument(
            '--num-speculative-tokens',
            type=int,
            default=EngineArgs.num_speculative_tokens,
            help='The number of speculative tokens to sample from '
            'the draft model in speculative decoding.')
        parser.add_argument(
            '--speculative-disable-mqa-scorer',
            action='store_true',
            help=
            'If set to True, the MQA scorer will be disabled in speculative '
            ' and fall back to batch expansion')
        parser.add_argument(
            '--speculative-draft-tensor-parallel-size',
            '-spec-draft-tp',
            type=int,
            default=EngineArgs.speculative_draft_tensor_parallel_size,
            help='Number of tensor parallel replicas for '
            'the draft model in speculative decoding.')

        parser.add_argument(
            '--speculative-max-model-len',
            type=int,
            default=EngineArgs.speculative_max_model_len,
            help='The maximum sequence length supported by the '
            'draft model. Sequences over this length will skip '
            'speculation.')

        parser.add_argument(
            '--speculative-disable-by-batch-size',
            type=int,
            default=EngineArgs.speculative_disable_by_batch_size,
            help='Disable speculative decoding for new incoming requests '
            'if the number of enqueue requests is larger than this value.')

        parser.add_argument(
            '--ngram-prompt-lookup-max',
            type=int,
            default=EngineArgs.ngram_prompt_lookup_max,
            help='Max size of window for ngram prompt lookup in speculative '
            'decoding.')

        parser.add_argument(
            '--ngram-prompt-lookup-min',
            type=int,
            default=EngineArgs.ngram_prompt_lookup_min,
            help='Min size of window for ngram prompt lookup in speculative '
            'decoding.')

        parser.add_argument(
            '--spec-decoding-acceptance-method',
            type=str,
            default=EngineArgs.spec_decoding_acceptance_method,
            choices=['rejection_sampler', 'typical_acceptance_sampler'],
            help='Specify the acceptance method to use during draft token '
            'verification in speculative decoding. Two types of acceptance '
            'routines are supported: '
            '1) RejectionSampler which does not allow changing the '
            'acceptance rate of draft tokens, '
            '2) TypicalAcceptanceSampler which is configurable, allowing for '
            'a higher acceptance rate at the cost of lower quality, '
            'and vice versa.')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-threshold',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_threshold,
            help='Set the lower bound threshold for the posterior '
            'probability of a token to be accepted. This threshold is '
            'used by the TypicalAcceptanceSampler to make sampling decisions '
            'during speculative decoding. Defaults to 0.09')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-alpha',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_alpha,
            help='A scaling factor for the entropy-based threshold for token '
            'acceptance in the TypicalAcceptanceSampler. Typically defaults '
            'to sqrt of --typical-acceptance-sampler-posterior-threshold '
            'i.e. 0.3')

        parser.add_argument(
            '--disable-logprobs-during-spec-decoding',
            action=StoreBoolean,
            default=EngineArgs.disable_logprobs_during_spec_decoding,
            nargs="?",
            const="True",
            help='If set to True, token log probabilities are not returned '
            'during speculative decoding. If set to False, log probabilities '
            'are returned according to the settings in SamplingParams. If '
            'not specified, it defaults to True. Disabling log probabilities '
            'during speculative decoding reduces latency by skipping logprob '
            'calculation in proposal sampling, target sampling, and after '
            'accepted tokens are determined.')

        parser.add_argument('--model-loader-extra-config',
                            type=nullable_str,
                            default=EngineArgs.model_loader_extra_config,
                            help='Extra config for model loader. '
                            'This will be passed to the model loader '
                            'corresponding to the chosen load_format. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary.')
        parser.add_argument(
            '--ignore-patterns',
            action="append",
            type=str,
            default=[],
            help="The pattern(s) to ignore when loading the model."
            "Default to 'original/**/*' to avoid repeated loading of llama's "
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
            "same as the `--model` argument. Noted that this name(s)"
            "will also be used in `model_name` tag content of "
            "prometheus metrics, if multiple names provided, metrics"
            "tag will take the first one.")
        parser.add_argument('--qlora-adapter-name-or-path',
                            type=str,
                            default=None,
                            help='Name or path of the QLoRA adapter.')

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
            ". It makes sense to set this only if --otlp-traces-endpoint is"
            " set. If set, it will collect detailed traces for the specified "
            "modules. This involves use of possibly costly and or blocking "
            "operations and hence might have a performance impact.")

        parser.add_argument(
            '--disable-async-output-proc',
            action='store_true',
            default=EngineArgs.disable_async_output_proc,
            help="Disable async output processing. This may result in "
            "lower performance.")
        parser.add_argument(
            '--override-neuron-config',
            type=json.loads,
            default=None,
            help="Override or set neuron device configuration. "
            "e.g. {\"cast_logits_dtype\": \"bloat16\"}.'")

        parser.add_argument(
            '--scheduling-policy',
            choices=['fcfs', 'priority'],
            default="fcfs",
            help='The scheduling policy to use. "fcfs" (first come first served'
            ', i.e. requests are handled in order of arrival; default) '
            'or "priority" (requests are handled based on given '
            'priority (lower value means earlier handling) and time of '
            'arrival deciding any ties).')

        parser.add_argument(
            '--pooling-type',
            choices=['LAST', 'ALL', 'CLS', 'STEP'],
            default=None,
            help='Used to configure the pooling method in the embedding model.'
        )

        parser.add_argument('--pooling-norm',
                            default=None,
                            action='store_true',
                            help="Used to determine whether to normalize "
                            "the pooled data in the embedding model.")

        parser.add_argument('--no-pooling-norm',
                            default=None,
                            action='store_false',
                            dest='pooling_norm',
                            help="Used to determine whether to normalize "
                            "the pooled data in the embedding model.")

        parser.add_argument('--pooling-softmax',
                            default=None,
                            action='store_true',
                            help="Used to determine whether to softmax "
                            "the pooled data in the embedding model.")

        parser.add_argument('--no-pooling-softmax',
                            default=None,
                            action='store_false',
                            dest='pooling_softmax',
                            help="Used to determine whether to softmax "
                            "the pooled data in the embedding model.")

        parser.add_argument(
            '--pooling-step-tag-id',
            type=int,
            default=None,
            help="When pooling-step-tag-id is not -1, it indicates "
            "that the score corresponding to the step-tag-ids in the "
            "generated sentence should be returned. Otherwise, it "
            "returns the scores for all tokens.")

        parser.add_argument(
            '--pooling-returned-token-ids',
            nargs='+',
            type=int,
            default=None,
            help="pooling-returned-token-ids represents a list of "
            "indices for the vocabulary dimensions to be extracted, "
            "such as the token IDs of good_token and bad_token in "
            "the math-shepherd-mistral-7b-prm model.")

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_model_config(self) -> ModelConfig:
        return ModelConfig(
            model=self.model,
            task=self.task,
            # We know this is not None because we set it in __post_init__
            tokenizer=cast(str, self.tokenizer),
            tokenizer_mode=self.tokenizer_mode,
            chat_template_text_format=self.chat_template_text_format,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            enforce_eager=self.enforce_eager,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            use_async_output_proc=not self.disable_async_output_proc,
            override_neuron_config=self.override_neuron_config,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            pooling_type=self.pooling_type,
            pooling_norm=self.pooling_norm,
            pooling_softmax=self.pooling_softmax,
            pooling_step_tag_id=self.pooling_step_tag_id,
            pooling_returned_token_ids=self.pooling_returned_token_ids,
        )

    def create_load_config(self) -> LoadConfig:
        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )

    def create_engine_config(self) -> VllmConfig:
        # gguf file needs a specific model loader and doesn't use hf_repo
        if check_gguf_file(self.model):
            self.quantization = self.load_format = "gguf"

        # bitsandbytes quantization needs a specific model loader
        # so we make sure the quant method and the load format are consistent
        if (self.quantization == "bitsandbytes" or
           self.qlora_adapter_name_or_path is not None) and \
           self.load_format != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes quantization and QLoRA adapter only support "
                f"'bitsandbytes' load format, but got {self.load_format}")

        if (self.load_format == "bitsandbytes" or
            self.qlora_adapter_name_or_path is not None) and \
            self.quantization != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes load format and QLoRA adapter only support "
                f"'bitsandbytes' quantization, but got {self.quantization}")

        assert self.cpu_offload_gb >= 0, (
            "CPU offload space must be non-negative"
            f", but got {self.cpu_offload_gb}")

        device_config = DeviceConfig(device=self.device)
        model_config = self.create_model_config()

        if model_config.is_multimodal_model:
            if self.enable_prefix_caching:
                logger.warning(
                    "--enable-prefix-caching is currently not "
                    "supported for multimodal models and has been disabled.")
            self.enable_prefix_caching = False

        maybe_register_config_serialize_by_value(self.trust_remote_code)

        cache_config = CacheConfig(
            # neuron needs block_size = max_model_len
            block_size=self.block_size if self.device != "neuron" else
            (self.max_model_len if self.max_model_len is not None else 0),
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
            cpu_offload_gb=self.cpu_offload_gb,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            worker_use_ray=self.worker_use_ray,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            tokenizer_pool_config=TokenizerPoolConfig.create_config(
                self.tokenizer_pool_size,
                self.tokenizer_pool_type,
                self.tokenizer_pool_extra_config,
            ),
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            distributed_executor_backend=self.distributed_executor_backend)

        max_model_len = model_config.max_model_len
        use_long_context = max_model_len > 32768
        if self.enable_chunked_prefill is None:
            # If not explicitly set, enable chunked prefill by default for
            # long context (> 32K) models. This is to avoid OOM errors in the
            # initial memory profiling phase.

            # Chunked prefill is currently disabled for multimodal models by
            # default.
            if use_long_context and not model_config.is_multimodal_model:
                is_gpu = device_config.device_type == "cuda"
                use_sliding_window = (model_config.get_sliding_window()
                                      is not None)
                use_spec_decode = self.speculative_model is not None
                if (is_gpu and not use_sliding_window and not use_spec_decode
                        and not self.enable_lora
                        and not self.enable_prompt_adapter):
                    self.enable_chunked_prefill = True
                    logger.warning(
                        "Chunked prefill is enabled by default for models with "
                        "max_model_len > 32K. Currently, chunked prefill might "
                        "not work with some features or models. If you "
                        "encounter any issues, please disable chunked prefill "
                        "by setting --enable-chunked-prefill=False.")
            if self.enable_chunked_prefill is None:
                self.enable_chunked_prefill = False

        if not self.enable_chunked_prefill and use_long_context:
            logger.warning(
                "The model has a long context length (%s). This may cause OOM "
                "errors during the initial memory profiling phase, or result "
                "in low performance due to small KV cache space. Consider "
                "setting --max-model-len to a smaller value.", max_model_len)

        speculative_config = SpeculativeConfig.maybe_create_spec_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            target_dtype=self.dtype,
            speculative_model=self.speculative_model,
            speculative_model_quantization = \
                self.speculative_model_quantization,
            speculative_draft_tensor_parallel_size = \
                self.speculative_draft_tensor_parallel_size,
            num_speculative_tokens=self.num_speculative_tokens,
            speculative_disable_mqa_scorer=self.speculative_disable_mqa_scorer,
            speculative_disable_by_batch_size=self.
            speculative_disable_by_batch_size,
            speculative_max_model_len=self.speculative_max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_log_stats=self.disable_log_stats,
            ngram_prompt_lookup_max=self.ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=self.ngram_prompt_lookup_min,
            draft_token_acceptance_method=\
                self.spec_decoding_acceptance_method,
            typical_acceptance_sampler_posterior_threshold=self.
            typical_acceptance_sampler_posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=self.
            typical_acceptance_sampler_posterior_alpha,
            disable_logprobs=self.disable_logprobs_during_spec_decoding,
        )

        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        if self.num_scheduler_steps > 1:
            if speculative_config is not None:
                raise ValueError("Speculative decoding is not supported with "
                                 "multi-step (--num-scheduler-steps > 1)")
            if self.enable_chunked_prefill and self.pipeline_parallel_size > 1:
                raise ValueError("Multi-Step Chunked-Prefill is not supported "
                                 "for pipeline-parallel-size > 1")

        # make sure num_lookahead_slots is set the higher value depending on
        # if we are using speculative decoding or multi-step
        num_lookahead_slots = max(self.num_lookahead_slots,
                                  self.num_scheduler_steps - 1)
        num_lookahead_slots = num_lookahead_slots \
            if speculative_config is None \
            else speculative_config.num_lookahead_slots

        if not self.use_v2_block_manager:
            logger.warning(
                "[DEPRECATED] Block manager v1 has been removed, "
                "and setting --use-v2-block-manager to True or False has "
                "no effect on vLLM behavior. Please remove "
                "--use-v2-block-manager in your engine argument. "
                "If your use case is not supported by "
                "SelfAttnBlockSpaceManager (i.e. block manager v2),"
                " please file an issue with detailed information.")

        scheduler_config = SchedulerConfig(
            task=model_config.task,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            num_lookahead_slots=num_lookahead_slots,
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            is_multimodal_model=model_config.is_multimodal_model,
            preemption_mode=self.preemption_mode,
            num_scheduler_steps=self.num_scheduler_steps,
            multi_step_stream_outputs=self.multi_step_stream_outputs,
            send_delta_data=(envs.VLLM_USE_RAY_SPMD_WORKER
                             and parallel_config.use_ray),
            policy=self.scheduling_policy,
        )
        lora_config = LoRAConfig(
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
            if self.model_loader_extra_config is None:
                self.model_loader_extra_config = {}
            self.model_loader_extra_config[
                "qlora_adapter_name_or_path"] = self.qlora_adapter_name_or_path

        load_config = self.create_load_config()

        prompt_adapter_config = PromptAdapterConfig(
            max_prompt_adapters=self.max_prompt_adapters,
            max_prompt_adapter_token=self.max_prompt_adapter_token) \
                                        if self.enable_prompt_adapter else None

        decoding_config = DecodingConfig(
            guided_decoding_backend=self.guided_decoding_backend)

        detailed_trace_modules = []
        if self.collect_detailed_traces is not None:
            detailed_trace_modules = self.collect_detailed_traces.split(",")
        for m in detailed_trace_modules:
            if m not in ALLOWED_DETAILED_TRACE_MODULES:
                raise ValueError(
                    f"Invalid module {m} in collect_detailed_traces. "
                    f"Valid modules are {ALLOWED_DETAILED_TRACE_MODULES}")
        observability_config = ObservabilityConfig(
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_model_forward_time="model" in detailed_trace_modules
            or "all" in detailed_trace_modules,
            collect_model_execute_time="worker" in detailed_trace_modules
            or "all" in detailed_trace_modules,
        )

        return VllmConfig(
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
        )


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser,
                     async_args_only: bool = False) -> FlexibleArgumentParser:
        if not async_args_only:
            parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='Disable logging requests.')
        return parser


# These functions are used by sphinx to build the documentation
def _engine_args_parser():
    return EngineArgs.add_cli_args(FlexibleArgumentParser())


def _async_engine_args_parser():
    return AsyncEngineArgs.add_cli_args(FlexibleArgumentParser(),
                                        async_args_only=True)
