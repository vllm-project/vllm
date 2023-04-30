import argparse
import time
from typing import List

from tqdm import tqdm
import numpy as np
import torch

from cacheflow.master.simple_frontend import SimpleFrontend
from cacheflow.master.server import (Server, add_server_arguments,
                                     process_server_arguments,
                                     initialize_cluster)
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import get_gpu_memory, get_cpu_memory


def main(args: argparse.Namespace):
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
        model_path=args.model_path,
        use_dummy_weights=args.use_dummy_weights,
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
    sampling_params_dict = {
        'n': args.n,
        'temperature': 0.0 if args.use_beam_search else 1.0,
        'top_p': 1.0,
        'use_beam_search': args.use_beam_search,
        'stop_token_ids': set(),
        'max_num_steps': args.output_len,
    }
    sampling_params = SamplingParams.from_dict(sampling_params_dict)
    print(sampling_params)
    input_token_ids = [0] * args.input_len

    def profile_step(profile=False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.batch_size):
            frontend._add_query(input_token_ids, sampling_params)
        server.add_sequence_groups(frontend.get_inputs())
        start_time = time.time()
        while True:
            server.step()
            if not server.has_unfinished_requests():
                break
        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warm up step")
    profile_step()

    # Benchmark.
    latencies = []
    for _ in tqdm(range(3), desc="Profile step"):
        latencies.append(profile_step())
    print(f'Avg latency: {np.mean(latencies)} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of decoding a single sentence.')
    parser = add_server_arguments(parser)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--use-beam-search', action='store_true')
    args = parser.parse_args()
    args = process_server_arguments(args)
    args.max_num_batched_tokens = max(
        args.max_num_batched_tokens, args.batch_size * args.input_len)
    print(args)
    main(args)
