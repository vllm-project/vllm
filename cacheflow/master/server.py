import argparse
from typing import List, Tuple
import random

import ray

from cacheflow.master.scheduler import Scheduler
from cacheflow.models import get_memory_analyzer
from cacheflow.worker.controller import Controller, DeviceID
from cacheflow.sequence import SequenceGroup
from cacheflow.sampling_params import SamplingParams


class Server:
    def __init__(
        self,
        model: str,
        model_path: str,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        block_size: int,
        dtype: str,
        seed: int,
        swap_space: int,
        max_num_batched_tokens: int,
        num_nodes: int,
        num_devices_per_node: int,
        distributed_init_method: str,
        all_stage_devices: List[List[DeviceID]],
        gpu_memory: int,
        cpu_memory: int,
    ):
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        self.world_size = pipeline_parallel_size * tensor_parallel_size

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
                model_path=model_path,
                max_num_batched_tokens=max_num_batched_tokens,
            )
            self.controllers.append(controller)

        # Create a scheduler.
        self.scheduler = Scheduler(
            controllers=self.controllers,
            block_size=block_size,
            num_gpu_blocks=self.num_gpu_blocks,
            num_cpu_blocks=self.num_cpu_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
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


def initialize_ray_cluster(
    address: str = 'auto',
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
) -> Tuple[int, int, str, List[List[DeviceID]]]:
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
    parser.add_argument('--model-path', type=str, default='~/.cacheflow/model_weights',
                        help='model path to download and load the weights')
    # Parallel arguments
    parser.add_argument('--pipeline-parallel-size', '-pp', type=int, default=1, help='number of pipeline stages')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1, help='number of tensor parallel replicas')
    # KV cache arguments
    parser.add_argument('--block-size', type=int, default=8, choices=[8, 16], help='token block size')
    # NOTE(woosuk): If FlashAttention is used, the float data type is not supported.
    parser.add_argument('--dtype', type=str, default='half', choices=['half', 'float'], help='data type')
    # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--swap-space', type=int, default=20, help='CPU swap space size (GiB) per GPU')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2560, help='maximum number of batched tokens')
    return parser
