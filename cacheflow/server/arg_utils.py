import argparse
from typing import Tuple

from cacheflow.config import (CacheConfig, ModelConfig, ParallelConfig,
                              SchedulerConfig)
from cacheflow.server.llm_server import LLMServer
from cacheflow.server.ray_utils import initialize_cluster

_GiB = 1 << 30


def add_server_arguments(parser: argparse.ArgumentParser):
    """Shared arguments for CacheFlow servers."""
    # Model arguments
    parser.add_argument('--model', type=str, default='facebook/opt-125m', help='model name')
    parser.add_argument('--download-dir', type=str, default=None,
                        help='directory to download and load the weights, '
                             'default to the default cache dir of huggingface')
    parser.add_argument('--use-np-weights', action='store_true',
                        help='save a numpy copy of model weights for faster loading')
    parser.add_argument('--use-dummy-weights', action='store_true', help='use dummy values for model weights')
    # TODO(woosuk): Support FP32.
    parser.add_argument('--dtype', type=str, default='default', choices=['default', 'half', 'bfloat16'],
                        help=('data type for model weights and activations. '
                              'The "default" option will use FP16 precision '
                              'for FP32 and FP16 models, and BF16 precision '
                              'for BF16 models.'))
    # Parallel arguments
    parser.add_argument('--use-ray', action='store_true', help='use Ray for distributed serving, will be automatically set when using more than 1 GPU')
    parser.add_argument('--pipeline-parallel-size', '-pp', type=int, default=1, help='number of pipeline stages')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1, help='number of tensor parallel replicas')
    # KV cache arguments
    parser.add_argument('--block-size', type=int, default=16, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256], help='token block size')
    # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--swap-space', type=int, default=4, help='CPU swap space size (GiB) per GPU')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95, help='the percentage of GPU memory to be used for the model executor')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2560, help='maximum number of batched tokens per iteration')
    parser.add_argument('--max-num-seqs', type=int, default=256, help='maximum number of sequences per iteration')
    parser.add_argument('--disable-log-stats', action='store_true', help='disable logging statistics')
    return parser


def create_server_configs_from_args(
    args: argparse.Namespace,
) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
    # Post-process the parsed arguments.
    args.swap_space = args.swap_space * _GiB
    args.max_num_seqs = min(args.max_num_seqs, args.max_num_batched_tokens)

    # Initialize the configs.
    model_config = ModelConfig(
        args.model, args.download_dir, args.use_np_weights,
        args.use_dummy_weights, args.dtype, args.seed)
    cache_config = CacheConfig(args.block_size, args.gpu_memory_utilization,
                               args.swap_space)
    parallel_config = ParallelConfig(args.pipeline_parallel_size,
                                     args.tensor_parallel_size, args.use_ray)
    scheduler_config = SchedulerConfig(args.max_num_batched_tokens,
                                       args.max_num_seqs)
    return model_config, cache_config, parallel_config, scheduler_config


def initialize_server_from_args(args: argparse.Namespace) -> LLMServer:
    server_configs = create_server_configs_from_args(args)
    parallel_config = server_configs[2]

    # Initialize the cluster.
    distributed_init_method, devices = initialize_cluster(parallel_config)

    # Create the LLM server.
    server = LLMServer(*server_configs, distributed_init_method, devices,
                       log_stats=not args.disable_log_stats)
    return server
