import argparse

from cacheflow import ServerArgs, LLMServer, SamplingParams


def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the server.
    server_args = ServerArgs.from_cli_args(args)
    server = LLMServer.from_server_args(server_args)

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

    request_id = 0

    # Run the server.
    while True:
        # To test iteration-level scheduling, we add one request at each step.
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            server.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs = server.step()
        for request_output in request_outputs:
            if request_output.finished():
                print(request_output)

        if not (server.has_unfinished_requests() or test_prompts):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple CacheFlow server.')
    parser = ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
