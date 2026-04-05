import asyncio
import os

import pytest

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import FlexibleArgumentParser


@pytest.fixture
def parser():
    return FlexibleArgumentParser()


@pytest.mark.asyncio
async def test_abort_with_async_spec_decode():
    """Test aborting a request when async scheduling and spec decode are enabled.

    This targets the TOCTOU gap in the worker where an aborted request could 
    leave a stale callback modifying a recycled batch slot, resulting in negative 
    or incorrect num_computed_tokens metrics.
    """
    os.environ["VLLM_TEST_ENABLE_ASYNC_SPEC_DECODE"] = "1"
    
    engine_args = AsyncEngineArgs(
        model="Jackalope/llama-160m",
        speculative_model="Jackalope/llama-160m",
        use_v2_block_manager=True,
        enforce_eager=True,
        num_speculative_tokens=3,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    req_id = "test-abort-async-spec"
    prompt = "Write a long essay on the history of asynchronous scheduling in Python."
    
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

    # Add the request
    results_generator = engine.generate(
        inputs=prompt,
        sampling_params=sampling_params,
        request_id=req_id,
    )

    # Wait for the first few tokens
    got_first_token = False
    async for _ in results_generator:
        got_first_token = True
        break

    assert got_first_token

    # Abort the request while it's generating
    await engine.abort(req_id)
    
    # Wait to ensure the abort cycles through the engine steps
    await asyncio.sleep(0.5)

    # Re-submit the request with the same ID. This causes the slot recycling
    # edge-case that exposed the TOCTOU reference bug.
    results_generator = engine.generate(
        inputs="Write another long essay on something else.",
        sampling_params=sampling_params,
        request_id=req_id,
    )
    
    # Fully consume the new request to ensure it completes successfully
    # without underflowing KV cache arrays from rogue deferred callbacks.
    final_output = None
    async for output in results_generator:
        final_output = output
        
    assert final_output is not None
    assert final_output.finished, "The resubmitted request failed to finish properly"

    # Stop the engine
    engine.shutdown()
