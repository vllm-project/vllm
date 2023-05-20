import argparse

from cacheflow.sampling_params import SamplingParams
from cacheflow.server.arg_utils import (
    add_server_arguments, initialize_server_from_args)


def main(args: argparse.Namespace):
    # Initialize the server.
    server = initialize_server_from_args(args)

    # Test the following prompts.
    test_prompts = [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2, temperature=0.8, top_p=0.95, frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=3, use_beam_search=True, temperature=0.0)),
    ]

    # Run the server. Here we assume that one request arrives at a time.
    while True:
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            server.add_request(prompt, sampling_params)

        updated_seq_groups = server.step()
        for seq_group in updated_seq_groups:
            if seq_group.is_finished():
                for seq in seq_group.seqs:
                    token_ids = seq.get_token_ids()
                    print(f"Seq {seq.seq_id}: {token_ids}")

        if not (server.has_unfinished_requests() or test_prompts):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple CacheFlow server.')
    parser = add_server_arguments(parser)
    args = parser.parse_args()
    main(args)
