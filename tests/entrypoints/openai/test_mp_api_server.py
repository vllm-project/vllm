import time

import pytest

from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


@pytest.mark.asyncio
async def test_mp_crash_detection():

    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args([])
    # use an invalid tensor_parallel_size to trigger the
    # error in the server
    args.tensor_parallel_size = 65536

    start = time.perf_counter()
    async with build_async_engine_client(args):
        pass
    end = time.perf_counter()

    assert end - start < 60, ("Expected vLLM to gracefully shutdown in <60s "
                              "if there is an error in the startup.")


@pytest.mark.asyncio
async def test_mp_cuda_init():
    # it should not crash, when cuda is initialized
    # in the API server process
    import torch
    torch.cuda.init()
    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args([])

    async with build_async_engine_client(args):
        pass
