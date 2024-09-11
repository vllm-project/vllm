import multiprocessing
from typing import Callable

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.usage.usage_lib import UsageContext


def run_normal(engine_args: AsyncEngineArgs, ipc_path: str):
    # Make engine.
    engine = MQLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.UNKNOWN_CONTEXT,
        ipc_path=ipc_path)

    # Run engine.
    engine.start()

class RemoteMQLLMEngine:

    def __init__(self, engine_args: AsyncEngineArgs,
                 ipc_path: str, run_fn: Callable = run_normal) -> None:

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
        client = MQLLMEngineClient(self.ipc_path, engine_config)
        while True:
            try:
                await client.setup()
                break
            except TimeoutError:
                assert self.proc.is_alive()
        return client
