import pytest
import argparse
from typing import List, Tuple
from vllm.logger import init_logger

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

# init variables
max_wait_q_len = 2

logger = init_logger(__name__)


class QueueOverflowError(Exception):
    pass


def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        ignore_eos=True,
                        max_tokens=1000)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        ignore_eos=True,
                        max_tokens=1000)),
        ("What is the meaning of life?",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        ignore_eos=True,
                        max_tokens=1000)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(temperature=0.8,
                        top_k=5,
                        presence_penalty=0.2,
                        ignore_eos=True,
                        max_tokens=1000)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    # make sure to set something like max_num_seq to ONE
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            try:
                engine.add_request(str(request_id), prompt, sampling_params)
            except ValueError as e:
                # Log error, cleanup, end test
                logger.info(f"{e}")
                for i in range(request_id):
                    engine.abort_request(str(i))
                raise QueueOverflowError(
                    f"Queue exceeded max length: {e}") from e
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    with pytest.raises(QueueOverflowError,
                       match="Queue exceeded max length: .*"):
        process_requests(engine, test_prompts)


# def test_max_queue_length():
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args_to_test = [
        '--max-num-seqs',
        str(1), '--max-queue-length',
        str(max_wait_q_len)
    ]
    args = parser.parse_args(args_to_test)
    main(args)
