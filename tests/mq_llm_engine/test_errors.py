import asyncio
import pytest
import tempfile
import uuid

from unittest.mock import Mock

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing import ENGINE_DEAD_ERROR

from vllm.engine.multiprocessing.engine import MQLLMEngine

from vllm.usage.usage_lib import UsageContext

from tests.mq_llm_engine.utils import RemoteMQLLMEngine


MODEL = "Qwen/Qwen2-0.5B-Instruct"
ENGINE_ARGS = AsyncEngineArgs(model=MODEL)
RAISED_ERROR = KeyError("foo")

@pytest.fixture(scope="function")
def tmp_socket():
    with tempfile.TemporaryDirectory() as td:
        yield f"ipc://{td}/{uuid.uuid4()}"

def run_with_evil_forward(engine_args: AsyncEngineArgs, 
                              ipc_path: str):
    engine = MQLLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context= UsageContext.UNKNOWN_CONTEXT,
        ipc_path=ipc_path)
    # Raise error during first forward pass.
    engine.engine.model_executor.execute_model = Mock(
        side_effect=RAISED_ERROR)
    engine.start()


@pytest.mark.asyncio
async def test_health_loop(tmp_socket):
    with RemoteMQLLMEngine(engine_args=ENGINE_ARGS,
                           ipc_path=tmp_socket,
                           run_fn=run_with_evil_forward) as engine:

        # Make client.
        client = await engine.make_client()

        POLL_TIME = 1.0
        health_task = asyncio.create_task(
            client.run_check_health_loop(timeout=POLL_TIME))

        # Server should be healthy.
        await asyncio.sleep(POLL_TIME * 3)
        await client.check_health()

        # Throws an error in engine.step().
        try:
            async for _ in client.generate(inputs="Hello my name is",
                                           sampling_params=SamplingParams(),
                                           request_id=uuid.uuid4()):
                pass
        except Exception as e:
            # First exception should be a RAISED_ERROR
            assert repr(e) == repr(RAISED_ERROR)

        # Engine is errored, should get ENGINE_DEAD_ERROR.
        try:
            async for _ in client.generate(inputs="Hello my name is",
                                           sampling_params=SamplingParams(),
                                           request_id=uuid.uuid4()):
                pass
        except Exception as e:
            # First exception should be a RAISED_ERROR
            assert e == ENGINE_DEAD_ERROR, (
                "Engine should be dead and raise ENGINE_DEAD_ERROR")
        

        asyncio.sleep(POLL_TIME * 3)
        try:
            await client.check_health()
        except Exception as e:
            assert e == ENGINE_DEAD_ERROR, (
                "Engine should be dead and raise ENGINE_DEAD_ERROR")

        await health_task
        client.close()


