import argparse
import uuid

from cacheflow import (add_server_arguments, initialize_server_from_args,
                       SamplingParams)


def main(args: argparse.Namespace):
    # Initialize the server.
    server = initialize_server_from_args(args)

    # Test the following prompts.
    test_prompts = [
        ("A robot may not injure a human being", SamplingParams()),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2, best_of=5, temperature=0.8, top_p=0.95, frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=3, best_of=3, use_beam_search=True, temperature=0.0)),
    ]

    # Run the server.
    while True:
        # To test iteration-level scheduling, we add one request at each step.
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            request_id = str(uuid.uuid4().hex[:8])
            server.add_request(request_id, prompt, sampling_params)

        request_outputs = server.step()
        for request_output in request_outputs:
            if request_output.done:
                print(request_output)

        if not (server.has_unfinished_requests() or test_prompts):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple CacheFlow server.')
    parser = add_server_arguments(parser)
    args = parser.parse_args()
    main(args)
