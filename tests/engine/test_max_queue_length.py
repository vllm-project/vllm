import pytest
import argparse
from typing import List, Tuple
from vllm.logger import init_logger

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput

# initialize constants
logger = init_logger(__name__)


class QueueOverflowError(Exception):
    pass


@pytest.fixture
def test_prompts() -> List[Tuple[str, SamplingParams]]:
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


@pytest.mark.parametrize(
    "max_wait_q_len, expect_error",
    [
        (1, True),  # error expected 
        (2, True),
        (3, False),  # No error expected 
        (4, False),
    ])
def test_max_queue_length(max_wait_q_len, expect_error, test_prompts):

    # Setup engine with appropriate max_queue_length value
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args_to_test = [
        '--max-num-seqs',
        str(1), '--max-queue-length',
        str(max_wait_q_len),
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "1",
        "--max-model-len",
        "1024",
    ]
    args = parser.parse_args(args_to_test)
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    # Test engine against request
    try:
        process_requests(engine, test_prompts)
        assert not expect_error, "QueueOverflowError did not occur as expected."
    except QueueOverflowError as e:
        assert expect_error, f" QueueOverflowError occurred as expected: {e}"
