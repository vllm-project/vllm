import json
import argparse
import time
from typing import List

from tqdm import tqdm
import numpy as np
import torch

from cacheflow.master.simple_frontend import SimpleFrontend
from cacheflow.master.server import (Server, add_server_arguments,
                                     initialize_ray_cluster)
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import get_gpu_memory, get_cpu_memory
from cacheflow.profile import set_sync_for_profiling


def main(args: argparse.Namespace):
    print(json.dumps(args.__dict__))
    set_sync_for_profiling()

    # TODO(zhuohan): Support pipeline parallelism.
    assert args.pipeline_parallel_size == 1, (
        'Pipeline parallelism is not supported yet.')

    cuda_profiler = False
    ray_cluster_address = "local" if cuda_profiler else "auto"

    (num_nodes, num_devices_per_node, distributed_init_method,
    all_stage_devices) = (
        initialize_ray_cluster(
            address=ray_cluster_address,
            pipeline_parallel_size=args.pipeline_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size))

    # Create a server.
    server = Server(
        model=args.model,
        model_path=args.model_path,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        dtype=args.dtype,
        seed=args.seed,
        swap_space=args.swap_space,
        max_batch_size=args.max_batch_size,
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        distributed_init_method=distributed_init_method,
        all_stage_devices=all_stage_devices,
        gpu_memory=get_gpu_memory(),
        cpu_memory=get_cpu_memory(),
    )

    # Create a frontend.
    frontend = SimpleFrontend(
        model_name=args.model,
        block_size=args.block_size,
    )
    sampling_params_dict = {
        'n': 1,
        'temperature': 0.0,
        'top_p': 1.0,
        'use_beam_search': False,
        'stop_token_ids': set(),
        'max_num_steps': args.output_len,
    }
    sampling_params = SamplingParams.from_dict(sampling_params_dict)
    input_token_ids = [0] * args.input_len

    def profile_step():
        if cuda_profiler:
            torch.cuda.cudart().cudaProfilerStart()
        server.reset_timer()
        for _ in range(args.batch_size):
            frontend._add_query(input_token_ids, sampling_params)
        server.add_sequence_groups(frontend.get_inputs())
        # Prompt step
        start_time = time.time()
        server.step()
        end_time = time.time()
        prompt_latency = end_time - start_time
        # Decoding steps
        num_decoding_steps = 0
        start_time = time.time()
        while server.has_unfinished_requests():
            server.step()
            num_decoding_steps += 1
        end_time = time.time()
        decoding_latency = end_time - start_time
        if cuda_profiler:
            torch.cuda.cudart().cudaProfilerStop()
        server_profile_results = server.get_profile_results()
        # First controller's first worker
        worker_execution_latency = server_profile_results[0][0]["execution_latency"]
        worker_communication_latency = server_profile_results[0][0]["communication_latency"]
        return (prompt_latency, decoding_latency, num_decoding_steps,
                worker_execution_latency, worker_communication_latency)

    print("== Warm up step ==")
    profile_step()

    # Benchmark.
    print("== Profile steps ==")
    num_profile_steps = 5
    for step in range(num_profile_steps):
        (prompt_latency, decoding_latency, num_decoding_steps,
         worker_execution_latency, worker_communication_latency) = profile_step()
        decoding_latency_per_step = decoding_latency / num_decoding_steps if num_decoding_steps > 0 else 0.0
        result = {
            "step": step,
            "prompt_latency_seconds": prompt_latency,
            "decoding_latency_seconds": decoding_latency,
            "decoding_latency_per_step_seconds": decoding_latency_per_step,
            "num_decoding_steps": num_decoding_steps,
            "worker_execution_latency_seconds": worker_execution_latency,
            "worker_communication_latency_seconds": worker_communication_latency,
        }
        print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow simple server.')
    parser = add_server_arguments(parser)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    args.max_batch_size = max(args.max_batch_size, args.batch_size * args.input_len)
    main(args)
