import argparse
import random
from typing import List, Tuple, Dict

import ray

from cacheflow.master.frontend import Frontend
from cacheflow.master.scheduler import Scheduler
from cacheflow.models import get_memory_analyzer
from cacheflow.worker.controller import Controller, DeviceID


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


def main(args: argparse.Namespace):
    # TODO(zhuohan): Support pipeline parallelism.
    assert args.pipeline_parallel_size == 1, (
        'Pipeline parallelism is not supported yet.')

    (num_nodes, num_devices_per_node, distributed_init_method,
     all_stage_devices) = (
        initialize_ray_cluster(
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size))

    world_size = args.pipeline_parallel_size * args.tensor_parallel_size

    memory_analyzer = get_memory_analyzer(
        model_name=args.model,
        block_size=args.block_size,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    num_gpu_blocks = memory_analyzer.get_max_num_gpu_blocks(
        max_num_batched_tokens=args.max_batch_size)
    num_cpu_blocks = memory_analyzer.get_max_num_cpu_blocks(
        swap_space=args.swap_space)
    print(f'# GPU blocks: {num_gpu_blocks}, # CPU blocks: {num_cpu_blocks}')

    # Create a controller for each pipeline stage.
    controllers: List[Controller] = []
    for i in range(args.pipeline_parallel_size):
        controller = Controller(
            stage_id=i,
            stage_devices=all_stage_devices[i],
            world_size=world_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_init_method=distributed_init_method,
            model_name=args.model,
            block_size=args.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            dtype=args.dtype,
            seed=args.seed,
            model_path=args.model_path,
        )
        controllers.append(controller)

    # Create a frontend.
    frontend = Frontend(
        model_name=args.model,
        block_size=args.block_size,
    )

    # Create a scheduler.
    scheduler = Scheduler(
        frontend=frontend,
        controllers=controllers,
        block_size=args.block_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        max_num_batched_tokens=args.max_batch_size,
    )
    # Connect the controllers.
    for i in range(len(controllers) - 1):
        controllers[i].set_next(controllers[i + 1])
    controllers[-1].set_next(scheduler)

    # Test the following inputs.
    test_inputs = [
        ('Ion Stoica is a', {'n': 4, 'use_beam_search': True, 'temperature': 0.0}),
        ('UC Berkeley is', {'n': 3, 'temperature': 0.8, 'top_p': 0.99}),
        ('The future of cloud computing is', {}),   # Use default parameters.
    ]
    while True:
        if test_inputs:
            text, sampling_params = test_inputs.pop(0)
            frontend.query(text, **sampling_params)
        scheduler.step()
        if not (scheduler.pending or scheduler.running or test_inputs):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow server')
    # Model arguments
    parser.add_argument('--model', type=str, default='facebook/opt-125m', help='model name')
    parser.add_argument('--model-path', type=str, default='~/.cacheflow/model_weights',
                        help='model path to download and load the weights')
    # Parallel arguments
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='number of pipeline stages')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='number of tensor parallel replicas')
    # KV cache arguments
    parser.add_argument('--block-size', type=int, default=8, choices=[8, 16], help='token block size')
    # NOTE(woosuk): If FlashAttention is used, the float data type is not supported.
    parser.add_argument('--dtype', type=str, default='half', choices=['half', 'float'], help='data type')
    # TODO(woosuk): Support fine-grained seeds (e.g., seed per request).
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--swap-space', type=int, default=20, help='CPU swap space size (GiB) per GPU')
    parser.add_argument('--max-batch-size', type=int, default=2560, help='maximum number of batched tokens')
    args = parser.parse_args()

    main(args)
