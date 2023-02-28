import argparse
import random
from typing import List, Tuple, Dict

import ray

from cacheflow.master.frontend import Frontend
from cacheflow.master.scheduler import Scheduler
from cacheflow.worker.controller import Controller


def initialize_ray_cluster(
        address: str = 'auto',
        pipeline_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
    ) -> Tuple[int, int, str, List[List[Tuple[int, str, int]]]]:
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
                ip = "node:".split(node_resource)[-1]
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
            num_gpu_blocks=args.num_gpu_blocks,
            num_cpu_blocks=args.num_cpu_blocks,
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
        num_gpu_blocks=args.num_gpu_blocks,
        num_cpu_blocks=args.num_cpu_blocks,
    )
    # Connect the controllers.
    for i in range(len(controllers) - 1):
        controllers[i].set_next(controllers[i + 1])
    controllers[-1].set_next(scheduler)

    test_inputs = [
        'Ion Stoica is a',
        'UC Berkeley is',
        'The future of cloud computing is',
    ]
    for prompt in test_inputs:
        frontend.query(prompt)

    # FIXME
    while True:
        scheduler.step()
        if not scheduler.pending and not scheduler.running:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow server')
    # Model arguments
    parser.add_argument('--model', type=str, default='facebook/opt-125m', help='model name')
    # Parallel arguments
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='number of pipeline stages')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='number of tensor parallel replicas')
    # KV cache arguments
    parser.add_argument('--block-size', type=int, default=8, help='token block size')
    # TODO(woosuk): Add an analytical model to determine the maximum number of GPU/CPU blocks.
    parser.add_argument('--num-gpu-blocks', type=int, default=1024, help='number of GPU blocks (per GPU)')
    parser.add_argument('--num-cpu-blocks', type=int, default=256, help='number of CPU blocks (per GPU)')
    args = parser.parse_args()

    main(args)
