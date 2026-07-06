# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from unittest import mock

import pytest

from vllm import LLM, PoolingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "intfloat/multilingual-e5-small"

@pytest.fixture(scope="module")
def llm():
    llm = LLM(
        model=MODEL_NAME,
        max_num_seqs=2,  # small to trigger tiling
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()



@pytest.mark.skip_global_cleanup
def test_tiling_engine_basic(llm):
    """
    Basic test with a small number of prompts (less than max_num_seqs).
    No tiling should be triggered, but the engine still processes correctly.
    """
    prompts = ["Hello", "World"]
    outputs = llm.encode(prompts, pooling_task="embed")
    assert len(outputs) == len(prompts)

@pytest.mark.skip_global_cleanup
def test_tiling_engine_many_requests(llm):
    """
    Test with a large number of prompts that exceeds max_num_seqs.
    This verifies that _run_tiling_engine correctly chunks requests,
    processes all of them, and returns outputs in the correct order.
    """
    num_prompts = 10
    prompts = [f"Prompt {i}" for i in range(num_prompts)]
    outputs = llm.encode(prompts, pooling_task="embed")
    assert len(outputs) == num_prompts


@pytest.mark.skip_global_cleanup
def test_tiling_engine_with_pooling_params(llm):
    """
    Test the tiling engine when different PoolingParams are provided.
    The engine must handle a list of params that matches the number of prompts.
    """
    num_prompts = 10
    prompts = [f"Prompt {i}" for i in range(num_prompts)]
    pooling_params = [PoolingParams() for _ in range(num_prompts)]

    outputs = llm.encode(
        prompts,
        pooling_params=pooling_params,
        pooling_task="embed"
    )
    assert len(outputs) == num_prompts

    # Single PoolingParams shared across all prompts
    single_param = PoolingParams()
    outputs = llm.encode(
        prompts,
        pooling_params=single_param,
        pooling_task="embed"
    )
    assert len(outputs) == num_prompts

    # None PoolingParams should fall back to default
    outputs = llm.encode(
        prompts,
        pooling_params=None,
        pooling_task="embed"
    )
    assert len(outputs) == num_prompts


@pytest.mark.skip_global_cleanup
def test_tiling_engine_abort_on_exception(llm):
    """
    Test that abort_request IS called with the correct arguments when an
    exception occurs inside the engine's step() loop.
    """
    prompts = ["Prompt 0", "Prompt 1", "Prompt 2"]

    # Mock the step method to throw an exception on the second call
    original_step = llm.llm_engine.step
    call_count = 0

    def mocked_step():
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated engine error")
        return original_step()

    with mock.patch.object(llm.llm_engine, 'step', side_effect=mocked_step):
        # We expect an exception to be raised from encode
        with mock.patch.object(llm.llm_engine, "abort_request") as mock_abort:
            with pytest.raises(RuntimeError, match="Simulated engine error"):
                llm.encode(prompts, pooling_task="embed")

        args, kwargs = mock_abort.call_args
        request_ids = args[0]
        assert isinstance(request_ids, list)
        assert len(request_ids) > 0
        assert kwargs.get("internal") is True