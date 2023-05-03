import argparse
from typing import List, Tuple, Optional
import random

import torch
try:
    import ray
except ImportError:
    ray = None

from cacheflow.master.scheduler import Scheduler
from cacheflow.master.simple_frontend import SimpleFrontend
from cacheflow.models import get_memory_analyzer
from cacheflow.worker.controller import Controller, DeviceID
from cacheflow.sequence import SequenceGroup
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import get_gpu_memory, get_cpu_memory


class Server:
    def __init__(
        self,
        model: str,
        cache_dir: Optional[str],
        use_dummy_weights: bool,
        use_np_cache: bool,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        block_size: int,
        dtype: str,
        seed: int,
        swap_space: int,
        max_num_batched_tokens: int,
        max_num_sequences: int,
        num_nodes: int,
        num_devices_per_node: int,
        distributed_init_method: str,
        all_stage_devices: List[List[DeviceID]],
        gpu_memory: int,
        cpu_memory: int,
        use_ray: bool,
        collect_stats: bool = False,
        do_memory_analysis: bool = False,
    ):
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        self.world_size = pipeline_parallel_size * tensor_parallel_size

        if not use_ray:
            assert self.world_size == 1, (
                "Only support single GPU without Ray.")

        self.memory_analyzer = get_memory_analyzer(
            model_name=model,
            block_size=block_size,
            dtype=dtype,
            gpu_memory=gpu_memory,
            cpu_memory=cpu_memory,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.num_gpu_blocks = self.memory_analyzer.get_max_num_gpu_blocks(
            max_num_batched_tokens=max_num_batched_tokens)
        self.num_cpu_blocks = self.memory_analyzer.get_max_num_cpu_blocks(
            swap_space=swap_space)
        print(f'# GPU blocks: {self.num_gpu_blocks}, '
              f'# CPU blocks: {self.num_cpu_blocks}')

        # Create a controller for each pipeline stage.
        self.controllers: List[Controller] = []
        for i in range(pipeline_parallel_size):
            controller = Controller(
                stage_id=i,
                stage_devices=all_stage_devices[i],
                world_size=self.world_size,
                pipeline_parallel_size=pipeline_parallel_size,
                tensor_parallel_size=tensor_parallel_size,
                distributed_init_method=distributed_init_method,
                model_name=model,
                block_size=block_size,
                num_gpu_blocks=self.num_gpu_blocks,
                num_cpu_blocks=self.num_cpu_blocks,
                dtype=dtype,
                seed=seed,
                cache_dir=cache_dir,
                use_dummy_weights=use_dummy_weights,
                use_np_cache=use_np_cache,
                max_num_batched_tokens=max_num_batched_tokens,
                use_ray=use_ray,
            )
            self.controllers.append(controller)

        # Create a scheduler.
        self.scheduler = Scheduler(
            controllers=self.controllers,
            block_size=block_size,
            num_gpu_blocks=self.num_gpu_blocks,
            num_cpu_blocks=self.num_cpu_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_sequences=max_num_sequences,
            collect_stats=collect_stats,
            do_memory_analysis=do_memory_analysis,
        )
        # Connect the controllers.
        for i in range(len(self.controllers) - 1):
            self.controllers[i].set_next(self.controllers[i + 1])
        self.controllers[-1].set_next(self.scheduler)

    def add_sequence_groups(
        self,
        sequence_groups: List[Tuple[SequenceGroup, SamplingParams]]
    ):
        self.scheduler.add_sequence_groups(sequence_groups)

    def step(self):
        return self.scheduler.step()

    def has_unfinished_requests(self):
        return (self.scheduler.waiting or self.scheduler.running or
                self.scheduler.swapped)


def initialize_cluster(
    use_ray: bool = False,
    address: Optional[str] = None,
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
) -> Tuple[int, int, str, List[List[DeviceID]]]:
    # Initialize cluster locally.
    if not use_ray:
        assert pipeline_parallel_size * tensor_parallel_size == 1, (
            "Only support single GPU without Ray.")
        num_nodes = 1
        num_devices_per_node = torch.cuda.device_count()
        port = random.randint(10000, 20000)
        # We need to setup the distributed init method to make sure
        # the distributed megatron code (e.g., get world size) works correctly.
        distributed_init_method = f"tcp://localhost:{port}"
        all_stage_devices = [[(0, None, 0)]]
        return (num_nodes, num_devices_per_node, distributed_init_method,
                all_stage_devices)

    assert ray is not None, (
        "Ray is not installed. Please install Ray to use distributed "
        "serving.")

    # Connect to a ray cluster.
    ray.init(address=address)

    # Assume we have a uniform cluster that each node has the same number of
    # GPUs for now.
    valid_node_resources = []
    num_devices_per_node = None
    for node in ray.nodes():
        if (not node['Alive']) or node['Resources']['GPU'] <= 0:
            continue
        if num_devices_per_node is None:
            num_devices_per_node = node['Resources']['GPU']
        else:
            assert num_devices_per_node == node['Resources']['GPU'], (
                "The number of GPUs per node is not uniform.")
        for key in node['Resources']:
            if key.startswith('node:'):
                valid_node_resources.append(key)

    num_nodes = len(valid_node_resources)

    assert (pipeline_parallel_size * tensor_parallel_size
            <= num_nodes * num_devices_per_node), (
                "The number of required GPUs exceeds the total number of "
                "available GPUs.")
    if tensor_parallel_size >= num_devices_per_node:
        assert tensor_parallel_size % num_devices_per_node == 0, (
            "The number of tensor parallelism is not divisible by the "
            "number of GPUs per node.")
    else:
        assert num_devices_per_node % tensor_parallel_size == 0, (
            "The number of GPUs per node is not divisible by the number "
            "of tensor parallelism.")

    # Assign GPUs to pipeline stages.
    rank = 0
    current_node_id = 0
    current_device_id = 0
    distributed_init_method = None
    all_stage_devices = []

    for i in range(pipeline_parallel_size):
        stage_devices = []
        for j in range(tensor_parallel_size):
            node_resource = valid_node_resources[current_node_id]
            stage_devices.append((rank, node_resource, current_device_id))
            if distributed_init_method is None:
                ip = node_resource.split("node:")[-1]
                port = random.randint(10000, 20000)
                distributed_init_method = f"tcp://{ip}:{port}"
            rank += 1
            current_device_id += 1
            if current_device_id >= num_devices_per_node:
                current_node_id += 1
                current_device_id = 0
        all_stage_devices.append(stage_devices)

    return (num_nodes, num_devices_per_node, distributed_init_method,
            all_stage_devices)


def add_server_arguments(parser: argparse.ArgumentParser):
    # Model arguments
    parser.add_argument('--model', type=str, default='facebook/opt-125m', help='model name')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='cache dir to download and load the weights, '
                             'default to the default cache dir of huggingface')
    parser.add_argument('--use-np-cache', action='store_true',
                        help='save a numpy copy of model weights for faster loading')
    parser.add_argument('--use-dummy-weights', action='store_true', help='use dummy values for model weights')
    # NOTE(woosuk): FlashAttention does not support float32.
    parser.add_argument('--dtype', type=str, default='half', choices=['half', 'bfloat16'], help='data type')
    # Parallel arguments
    parser.add_argument('--use-ray', action='store_true', help='use Ray for distributed serving, will be automatically set when using more than 1 GPU')
    parser.add_argument('--pipeline-parallel-size', '-pp', type=int, default=1, help='number of pipeline stages')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1, help='number of tensor parallel replicas')
    # KV cache arguments
    parser.add_argument('--block-size', type=int, default=16, choices=[1, 2, 4, 8, 16, 32, 64, 128, 256], help='token block size')
    # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--swap-space', type=int, default=20, help='CPU swap space size (GiB) per GPU')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2560, help='maximum number of batched tokens per iteration')
    parser.add_argument('--max-num-sequences', type=int, default=256, help='maximum number of sequences per iteration')
    return parser


def process_server_arguments(args: argparse.Namespace):
    if args.pipeline_parallel_size * args.tensor_parallel_size > 1:
        args.use_ray = True
    return args


def init_local_server_and_frontend_with_arguments(args: argparse.Namespace):
    # TODO(zhuohan): Support pipeline parallelism.
    assert args.pipeline_parallel_size == 1, (
        'Pipeline parallelism is not supported yet.')

    (num_nodes, num_devices_per_node, distributed_init_method,
    all_stage_devices) = (
        initialize_cluster(
            use_ray=args.use_ray,
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size))

    # Create a server.
    server = Server(
        model=args.model,
        cache_dir=args.cache_dir,
        use_dummy_weights=args.use_dummy_weights,
        use_np_cache=args.use_np_cache,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        dtype=args.dtype,
        seed=args.seed,
        swap_space=args.swap_space,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_sequences=args.max_num_sequences,
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        distributed_init_method=distributed_init_method,
        all_stage_devices=all_stage_devices,
        gpu_memory=get_gpu_memory(),
        cpu_memory=get_cpu_memory(),
        use_ray=args.use_ray,
    )

    # Create a frontend.
    frontend = SimpleFrontend(
        model_name=args.model,
        block_size=args.block_size,
    )
    return server, frontend
