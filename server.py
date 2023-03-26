import argparse
from typing import List

from cacheflow.master.frontend import Frontend
from cacheflow.master.server import Server, add_server_arguments
from cacheflow.sampling_params import SamplingParams

def main(args: argparse.Namespace):
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
    )

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
            frontend.query(text, sampling_params)
        server.scheduler.add_sequence_groups(frontend.get_inputs())
        server.scheduler.step()
        for seq_group in server.scheduler.get_finished():
            frontend.print_response(seq_group)
        if not (server.scheduler.pending or server.scheduler.running or
                test_inputs):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow server')
    parser = add_server_arguments(parser)
    args = parser.parse_args()
    main(args)
