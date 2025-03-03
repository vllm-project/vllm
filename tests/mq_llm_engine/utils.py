# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
from typing import Callable, Union

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext


async def generate(
        client: MQLLMEngineClient,
        request_id: str,
        num_tokens: int,
        return_output: bool = False) -> Union[RequestOutput, tuple[int, str]]:

    final_output = None
    count = 0
    async for out in client.generate(
            request_id=request_id,
            prompt="Hello my name is Robert and",
            sampling_params=SamplingParams(max_tokens=num_tokens,
                                           temperature=0)):

        count += 1
        final_output = out
        await asyncio.sleep(0.)

    if return_output:
        return final_output

    # Confirm we generated all the tokens we expected.
    return count, request_id


def run_normal(engine_args: AsyncEngineArgs, ipc_path: str):
    # Make engine.
    engine = MQLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.UNKNOWN_CONTEXT,
        ipc_path=ipc_path)

    # Run engine.
    engine.start()


class RemoteMQLLMEngine:

    def __init__(self,
                 engine_args: AsyncEngineArgs,
                 ipc_path: str,
                 run_fn: Callable = run_normal) -> None:

        self.engine_args = engine_args
        self.ipc_path = ipc_path
        context = multiprocessing.get_context("spawn")
        self.proc = context.Process(target=run_fn,
                                    args=(engine_args, ipc_path))
        self.proc.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.kill()

    async def make_client(self) -> MQLLMEngineClient:
        engine_config = self.engine_args.create_engine_config()
        client = MQLLMEngineClient(self.ipc_path, engine_config, self.proc.pid)
        while True:
            try:
                await client.setup()
                break
            except TimeoutError:
                assert self.proc.is_alive()
        return client
