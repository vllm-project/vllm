import argparse
from typing import List

from cacheflow.master.frontend import Frontend
from cacheflow.master.scheduler import Scheduler
from cacheflow.master.server_utils import (initialize_ray_cluster,
                                           add_server_arguments)
from cacheflow.models import get_memory_analyzer
from cacheflow.worker.controller import Controller
from cacheflow.sampling_params import SamplingParams

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

    # Create a scheduler.
    scheduler = Scheduler(
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

    # Create a frontend.
    frontend = Frontend(
        model_name=args.model,
        block_size=args.block_size,
    )

    # Test the following inputs.
    test_inputs = [
        ('Ion Stoica is a', {'n': 4, 'use_beam_search': True, 'temperature': 0.0}),
        ('UC Berkeley is', {'n': 3, 'temperature': 0.8, 'top_p': 0.99}),
        ('The future of cloud computing is', {}),   # Use default parameters.
    ]
    while True:
        if test_inputs:
            text, sampling_params_dict = test_inputs.pop(0)
            sampling_params = SamplingParams.from_dict(sampling_params_dict)
            sampling_params = frontend.add_eos_token(sampling_params)
            frontend.query(text, **sampling_params)
        scheduler.add_sequence_groups(frontend.get_inputs())
        scheduler.step()
        for seq_group in scheduler.get_finished():
            frontend.print_response(seq_group)
        if not (scheduler.pending or scheduler.running or test_inputs):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow server')
    parser = add_server_arguments(parser)
    args = parser.parse_args()
    main(args)
