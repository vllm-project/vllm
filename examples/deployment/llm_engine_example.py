# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates using the `LLMEngine`
for processing prompts with various sampling parameters.
"""

import argparse

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_test_prompts() -> list[tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        (
            "A robot may not injure a human being",
            SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1),
        ),
        (
            "To be or not to be,",
            SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2),
        ),
        (
            "What is the meaning of life?",
            SamplingParams(n=2, temperature=0.8, top_p=0.95, frequency_penalty=0.1),
        ),
    ]


def process_requests(engine: LLMEngine, test_prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    print("-" * 50)
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                print("-" * 50)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    args = parse_args()
    main(args)
