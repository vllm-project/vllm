import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, TokenizerPoolConfig,
                         VisionLanguageConfig)
from vllm.utils import str_to_int_tuple


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    enable_prefix_caching: bool = False
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_logprobs: int = 5  # OpenAI default value
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    tokenizer_pool_type: str = "ray"
    tokenizer_pool_extra_config: Optional[dict] = None
    enable_lora: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    lora_extra_vocab_size: int = 256
    lora_dtype = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    ray_workers_use_nsight: bool = False
    # Related to Vision-language models such as llava
    image_input_type: Optional[str] = None
    image_token_id: Optional[int] = None
    image_input_shape: Optional[str] = None
    image_feature_size: Optional[int] = None
    scheduler_delay_factor: float = 0.0

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # NOTE: If you update any of the arguments below, please also
        # make sure to update docs/source/models/engine_args.rst

        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            default='facebook/opt-125m',
            help='name or path of the huggingface model to use')
        parser.add_argument(
            '--tokenizer',
            type=str,
            default=EngineArgs.tokenizer,
            help='name or path of the huggingface tokenizer to use')
        parser.add_argument(
            '--revision',
            type=str,
            default=None,
            help='the specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--code-revision',
            type=str,
            default=None,
            help='the specific revision to use for the model code on '
            'Hugging Face Hub. It can be a branch name, a tag name, or a '
            'commit id. If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=str,
            default=None,
            help='the specific tokenizer version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument('--tokenizer-mode',
                            type=str,
                            default=EngineArgs.tokenizer_mode,
                            choices=['auto', 'slow'],
                            help='tokenizer mode. "auto" will use the fast '
                            'tokenizer if available, and "slow" will '
                            'always use the slow tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='trust remote code from huggingface')
        parser.add_argument('--download-dir',
                            type=str,
                            default=EngineArgs.download_dir,
                            help='directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=['auto', 'pt', 'safetensors', 'npcache', 'dummy'],
            help='The format of the model weights to load. '
            '"auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available. '
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading. '
            '"dummy" will initialize the weights with random values, '
            'which is mainly for profiling.')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
        parser.add_argument(
            '--kv-cache-dtype',
            type=str,
            choices=['auto', 'fp8_e5m2'],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            'data type. Note FP8 is not supported when cuda version is '
            'lower than 11.8.')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=EngineArgs.max_model_len,
                            help='model context length. If unspecified, '
                            'will be automatically derived from the model.')
        # Parallel arguments
        parser.add_argument('--worker-use-ray',
                            action='store_true',
                            help='use Ray for distributed serving, will be '
                            'automatically set when using more than 1 GPU')
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        parser.add_argument(
            '--max-parallel-loading-workers',
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help='load model sequentially in multiple batches, '
            'to avoid RAM OOM when using tensor '
            'parallel and large models')
        parser.add_argument(
            '--ray-workers-use-nsight',
            action='store_true',
            help='If specified, use nsight to profile ray workers')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32, 128],
                            help='token block size')

        parser.add_argument('--enable-prefix-caching',
                            action='store_true',
                            help='Enables automatic prefix caching')

        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='random seed')
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument(
            '--gpu-memory-utilization',
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help='the fraction of GPU memory to be used for '
            'the model executor, which can range from 0 to 1.'
            'If unspecified, will use the default value of 0.9.')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument(
            '--max-logprobs',
            type=int,
            default=EngineArgs.max_logprobs,
            help=('max number of log probs to return logprobs is specified in'
                  ' SamplingParams'))
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='disable logging statistics')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=str,
                            choices=['awq', 'gptq', 'squeezellm', None],
                            default=EngineArgs.quantization,
                            help='Method used to quantize the weights. If '
                            'None, we first check the `quantization_config` '
                            'attribute in the model config file. If that is '
                            'None, we assume the model weights are not '
                            'quantized and use `dtype` to determine the data '
                            'type of the weights.')
        parser.add_argument('--enforce-eager',
                            action='store_true',
                            help='Always use eager-mode PyTorch. If False, '
                            'will use eager mode and CUDA graph in hybrid '
                            'for maximal performance and flexibility.')
        parser.add_argument('--max-context-len-to-capture',
                            type=int,
                            default=EngineArgs.max_context_len_to_capture,
                            help='maximum context length covered by CUDA '
                            'graphs. When a sequence has context length '
                            'larger than this, we fall back to eager mode.')
        parser.add_argument('--disable-custom-all-reduce',
                            action='store_true',
                            default=EngineArgs.disable_custom_all_reduce,
                            help='See ParallelConfig')
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
                            type=str,
                            default=EngineArgs.tokenizer_pool_extra_config,
                            help='Extra config for tokenizer pool. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary. Ignored if '
                            'tokenizer_pool_size is 0.')
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
            '--max-cpu-loras',
            type=int,
            default=EngineArgs.max_cpu_loras,
            help=('Maximum number of LoRAs to store in CPU memory. '
                  'Must be >= than max_num_seqs. '
                  'Defaults to max_num_seqs.'))
        parser.add_argument("--device",
                            type=str,
                            default=EngineArgs.device,
                            choices=["auto", "cuda", "neuron"],
                            help='Device type for vLLM execution.')
        # Related to Vision-language models such as llava
        parser.add_argument(
            '--image-input-type',
            type=str,
            default=None,
            choices=[
                t.name.lower() for t in VisionLanguageConfig.ImageInputType
            ],
            help=('The image input type passed into vLLM. '
                  'Should be one of "pixel_values" or "image_features".'))
        parser.add_argument('--image-token-id',
                            type=int,
                            default=None,
                            help=('Input id for image token.'))
        parser.add_argument(
            '--image-input-shape',
            type=str,
            default=None,
            help=('The biggest image input shape (worst for memory footprint) '
                  'given an input type. Only used for vLLM\'s profile_run.'))
        parser.add_argument(
            '--image-feature-size',
            type=int,
            default=None,
            help=('The image feature size along the context dimension.'))
        parser.add_argument(
            '--scheduler-delay-factor',
            type=float,
            default=EngineArgs.scheduler_delay_factor,
            help='Apply a delay (of delay factor multiplied by previous'
            'prompt latency) before scheduling next prompt.')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig,
               DeviceConfig, Optional[LoRAConfig],
               Optional[VisionLanguageConfig]]:
        device_config = DeviceConfig(self.device)
        model_config = ModelConfig(
            self.model, self.tokenizer, self.tokenizer_mode,
            self.trust_remote_code, self.download_dir, self.load_format,
            self.dtype, self.seed, self.revision, self.code_revision,
            self.tokenizer_revision, self.max_model_len, self.quantization,
            self.enforce_eager, self.max_context_len_to_capture,
            self.max_logprobs)
        cache_config = CacheConfig(self.block_size,
                                   self.gpu_memory_utilization,
                                   self.swap_space, self.kv_cache_dtype,
                                   model_config.get_sliding_window(),
                                   self.enable_prefix_caching)
        parallel_config = ParallelConfig(
            self.pipeline_parallel_size, self.tensor_parallel_size,
            self.worker_use_ray, self.max_parallel_loading_workers,
            self.disable_custom_all_reduce,
            TokenizerPoolConfig.create_config(
                self.tokenizer_pool_size,
                self.tokenizer_pool_type,
                self.tokenizer_pool_extra_config,
            ), self.ray_workers_use_nsight)
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens,
                                           self.max_num_seqs,
                                           model_config.max_model_len,
                                           self.scheduler_delay_factor)
        lora_config = LoRAConfig(
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            lora_extra_vocab_size=self.lora_extra_vocab_size,
            lora_dtype=self.lora_dtype,
            max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras
            and self.max_cpu_loras > 0 else None) if self.enable_lora else None

        if self.image_input_type:
            if (not self.image_token_id or not self.image_input_shape
                    or not self.image_feature_size):
                raise ValueError(
                    'Specify `image_token_id`, `image_input_shape` and '
                    '`image_feature_size` together with `image_input_type`.')
            vision_language_config = VisionLanguageConfig(
                image_input_type=VisionLanguageConfig.
                get_image_input_enum_type(self.image_input_type),
                image_token_id=self.image_token_id,
                image_input_shape=str_to_int_tuple(self.image_input_shape),
                image_feature_size=self.image_feature_size,
            )
        else:
            vision_language_config = None

        return (model_config, cache_config, parallel_config, scheduler_config,
                device_config, lora_config, vision_language_config)


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    engine_use_ray: bool = False
    disable_log_requests: bool = False
    max_log_len: Optional[int] = None

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--engine-use-ray',
                            action='store_true',
                            help='use Ray to start the LLM engine in a '
                            'separate process as the server process.')
        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='disable logging requests')
        parser.add_argument('--max-log-len',
                            type=int,
                            default=None,
                            help='max number of prompt characters or prompt '
                            'ID numbers being printed in log. '
                            'Default: unlimited.')
        return parser
