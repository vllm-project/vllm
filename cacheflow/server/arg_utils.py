import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from cacheflow.config import (CacheConfig, ModelConfig, ParallelConfig,
                              SchedulerConfig)


@dataclass
class ServerArgs:
    model: str
    download_dir: Optional[str] = None
    use_np_weights: bool = False
    use_dummy_weights: bool = False
    dtype: str = "default"
    seed: int = 0
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.95
    max_num_batched_tokens: int = 2560
    max_num_seqs: int = 256
    disable_log_stats: bool = False

    def __post_init__(self):
        self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Shared CLI arguments for CacheFlow servers."""
        # Model arguments
        parser.add_argument('--model', type=str, default='facebook/opt-125m',
                            help='name or path of the huggingface model to use')
        parser.add_argument('--download-dir', type=str,
                            default=ServerArgs.download_dir,
                            help='directory to download and load the weights, '
                                 'default to the default cache dir of '
                                 'huggingface')
        parser.add_argument('--use-np-weights', action='store_true',
                            help='save a numpy copy of model weights for '
                                 'faster loading. This can increase the disk '
                                 'usage by up to 2x.')
        parser.add_argument('--use-dummy-weights', action='store_true',
                            help='use dummy values for model weights')
        # TODO(woosuk): Support FP32.
        parser.add_argument('--dtype', type=str, default=ServerArgs.dtype,
                            choices=['default', 'half', 'bfloat16'],
                            help='data type for model weights and activations. '
                                 'The "default" option will use FP16 precision '
                                 'for FP32 and FP16 models, and BF16 precision '
                                 'for BF16 models.')
        # Parallel arguments
        parser.add_argument('--worker-use-ray', action='store_true',
                            help='use Ray for distributed serving, will be '
                                 'automatically set when using more than 1 GPU')
        parser.add_argument('--pipeline-parallel-size', '-pp', type=int,
                            default=ServerArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size', '-tp', type=int,
                            default=ServerArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        # KV cache arguments
        parser.add_argument('--block-size', type=int,
                            default=ServerArgs.block_size,
                            choices=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                            help='token block size')
        # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
        parser.add_argument('--seed', type=int, default=ServerArgs.seed,
                            help='random seed')
        parser.add_argument('--swap-space', type=int,
                            default=ServerArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument('--gpu-memory-utilization', type=float,
                            default=ServerArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                                 'the model executor')
        parser.add_argument('--max-num-batched-tokens', type=int,
                            default=ServerArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                                 'iteration')
        parser.add_argument('--max-num-seqs', type=int,
                            default=ServerArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--disable-log-stats', action='store_true',
                            help='disable logging statistics')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ServerArgs":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        server_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return server_args

    def create_server_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        # Initialize the configs.
        model_config = ModelConfig(
            self.model, self.download_dir, self.use_np_weights,
            self.use_dummy_weights, self.dtype, self.seed)
        cache_config = CacheConfig(self.block_size, self.gpu_memory_utilization,
                                   self.swap_space)
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size,
                                         self.worker_use_ray)
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens,
                                           self.max_num_seqs)
        return model_config, cache_config, parallel_config, scheduler_config


@dataclass
class AsyncServerArgs(ServerArgs):
    server_use_ray: bool = False

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = ServerArgs.add_cli_args(parser)
        parser.add_argument('--server-use-ray', action='store_true',
                            help='use Ray to start the LLM server in a '
                                 'separate process as the web server process.')
        return parser
