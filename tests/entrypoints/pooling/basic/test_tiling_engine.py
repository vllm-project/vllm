# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm import PoolingParams
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.outputs import (
    LateChunk,
    LateChunkingMetadata,
    PoolingOutput,
    PoolingRequestOutput,
    RequestError,
)

MODEL_NAME = "intfloat/multilingual-e5-small"


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_model_len=None,
        max_num_seqs=2,  # small to trigger tiling
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        enable_chunked_prefill=None,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.skip_global_cleanup
def test_tiling_engine_basic(llm):
    """Basic test with a small number of prompts (less than max_num_seqs).
    No tiling should be triggered, but the engine still processes correctly.
    """
    prompts = ["Hello", "World"]
    outputs = llm.encode(prompts, pooling_task="embed")
    assert len(outputs) == len(prompts)


@pytest.mark.skip_global_cleanup
def test_tiling_engine_many_requests(llm):
    """Test with a large number of prompts that exceeds max_num_seqs.
    This verifies that _run_tiling_engine correctly chunks requests,
    processes all of them, and returns outputs in the correct order.
    """
    num_prompts = 10
    prompts = [f"Prompt {i}" for i in range(num_prompts)]
    outputs = llm.encode(prompts, pooling_task="embed")
    assert len(outputs) == num_prompts


@pytest.mark.skip_global_cleanup
def test_tiling_engine_with_pooling_params(llm):
    """Test the tiling engine when different PoolingParams are provided.
    The engine must handle a list of params that matches the number of prompts.
    """
    num_prompts = 10
    prompts = [f"Prompt {i}" for i in range(num_prompts)]
    pooling_params = [PoolingParams() for _ in range(num_prompts)]

    outputs = llm.encode(prompts, pooling_params=pooling_params, pooling_task="embed")
    assert len(outputs) == num_prompts

    # Single PoolingParams shared across all prompts
    single_param = PoolingParams()
    outputs = llm.encode(prompts, pooling_params=single_param, pooling_task="embed")
    assert len(outputs) == num_prompts

    # None PoolingParams should fall back to default
    outputs = llm.encode(prompts, pooling_params=None, pooling_task="embed")
    assert len(outputs) == num_prompts


@pytest.mark.skip_global_cleanup
def test_tiling_engine_abort_on_exception(llm):
    """Test that abort_request IS called with the correct arguments when an
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

    with mock.patch.object(llm.llm_engine, "step", side_effect=mocked_step):
        # We expect an exception to be raised from encode
        with mock.patch.object(llm.llm_engine, "abort_request") as mock_abort:  # noqa: SIM117
            with pytest.raises(RuntimeError, match="Simulated engine error"):
                llm.encode(prompts, pooling_task="embed")

        args, kwargs = mock_abort.call_args
        request_ids = args[0]
        assert isinstance(request_ids, list)
        assert len(request_ids) > 0


def _mock_chunk_tiling_engine(outputs):
    llm = mock.Mock(spec=PoolingOfflineMixin)
    llm._run_tiling_engine = PoolingOfflineMixin._run_tiling_engine.__get__(llm)
    llm._executor = SimpleNamespace(map=map)
    llm.llm_engine = mock.Mock()
    llm.llm_engine.vllm_config.scheduler_config.max_num_seqs = 2
    llm.llm_engine.has_unfinished_requests.return_value = False
    llm.llm_engine.step.side_effect = outputs
    llm._render_and_add_requests = mock.Mock(return_value=["0-internal", "1-internal"])
    requests = [
        {
            "prompts": {"type": "token", "prompt_token_ids": [1, 2]},
            "params": PoolingParams(task="token_embed", late_chunk_size=2),
            "lora_requests": None,
            "priorities": 0,
            "late_chunking": LateChunkingMetadata(
                chunk_size=2,
                input_tokens=2,
                chunks=[LateChunk((0, 2), (i, i + 2))],
            ),
        }
        for i in range(2)
    ]
    return llm, requests


def _chunk_output(request_id, **kwargs):
    return PoolingRequestOutput(
        str(request_id), PoolingOutput(torch.ones(1, 4)), [1, 2], 0, True, **kwargs
    )


def test_late_chunk_mapping_follows_request_ids_and_preserves_request_errors():
    error = RequestError("test_error", "original error")
    failed = _chunk_output(1, error=error)
    # A failed request need not have a valid chunk tensor.
    failed.outputs.data = failed.outputs.data[:0]
    llm, requests = _mock_chunk_tiling_engine([[failed, _chunk_output(0)]])
    outputs = llm._run_tiling_engine(
        SimpleNamespace(render=lambda x: x), lambda: iter(requests), 2, use_tqdm=False
    )
    assert outputs[0].late_chunking is requests[0]["late_chunking"]
    assert outputs[1].late_chunking is None
    assert outputs[1].error is error
    llm.llm_engine.abort_request.assert_not_called()


@pytest.mark.parametrize(
    "error, abort_expected",
    [
        pytest.param(RuntimeError("step failed"), True, id="runtime-error"),
        pytest.param(KeyboardInterrupt(), False, id="keyboard-interrupt"),
    ],
)
def test_late_chunk_mapping_propagates_errors_and_is_not_reused(error, abort_expected):
    llm, requests = _mock_chunk_tiling_engine(error)
    processor = SimpleNamespace(render=lambda x: x)
    with pytest.raises(type(error)):
        llm._run_tiling_engine(processor, lambda: iter(requests), 2, use_tqdm=False)
    if abort_expected:
        llm.llm_engine.abort_request.assert_called_once()
        assert set(llm.llm_engine.abort_request.call_args.args[0]) == {"0", "1"}
    else:
        llm.llm_engine.abort_request.assert_not_called()
    for request in requests:
        del request["late_chunking"]
        request["params"] = PoolingParams(task="token_embed")
    llm.llm_engine.step.side_effect = [[_chunk_output(1), _chunk_output(0)]]
    outputs = llm._run_tiling_engine(
        processor, lambda: iter(requests), 2, use_tqdm=False
    )
    assert all(output.late_chunking is None for output in outputs)


def test_late_chunk_mapping_rejects_successful_output_with_wrong_row_count():
    first = _chunk_output(0)
    first.outputs.data = first.outputs.data[:0]
    llm, requests = _mock_chunk_tiling_engine([[first, _chunk_output(1)]])
    with pytest.raises(ValueError, match="does not match"):
        llm._run_tiling_engine(
            SimpleNamespace(render=lambda x: x),
            lambda: iter(requests),
            2,
            use_tqdm=False,
        )
    assert set(llm.llm_engine.abort_request.call_args.args[0]) == {"0", "1"}
