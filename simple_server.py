import argparse

from cacheflow.core.server import (
    add_server_arguments, process_server_arguments,
    init_local_server_and_frontend_with_arguments)
from cacheflow.sampling_params import SamplingParams


def main(args: argparse.Namespace):
    server, frontend = init_local_server_and_frontend_with_arguments(args)
    # Test the following inputs.
    test_inputs = [
        ("A robot may not injure a human being", {}),   # Use default parameters.
        ("To be or not to be,", {"temperature": 0.8, "top_k": 5}),
        ("What is the meaning of life?", {"n": 2, "temperature": 0.8, "top_p": 0.95}),
        ("It is only with the heart that one can see rightly", {"n": 3, "use_beam_search": True, "temperature": 0.0}),
    ]
    while True:
        if test_inputs:
            text, sampling_params_dict = test_inputs.pop(0)
            sampling_params = SamplingParams.from_dict(sampling_params_dict)
            sampling_params = frontend.add_eos_token(sampling_params)
            frontend.query(text, sampling_params)
        server.add_sequence_groups(frontend.get_inputs())
        updated_seq_groups = server.step()
        for seq_group in updated_seq_groups:
            if seq_group.is_finished():
                frontend.print_response(seq_group)
        if not (server.has_unfinished_requests() or test_inputs):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CacheFlow simple server.')
    parser = add_server_arguments(parser)
    args = parser.parse_args()
    args = process_server_arguments(args)
    main(args)
