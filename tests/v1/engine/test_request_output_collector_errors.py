# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import traceback

import pytest

from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.output_processor import RequestOutputCollector


@pytest.mark.asyncio
async def test_collector_raises_fresh_engine_dead_error():
    """propagate_error() shares one EngineDeadError across all collectors.

    Each consumer must raise its own fresh instance, not the shared one,
    so the traceback logged per request stays constant-size instead of
    accumulating one round of frames per failed request.
    """
    shared_error = EngineDeadError()
    # Simulate earlier consumers having re-raised the shared error: its
    # traceback already carries several rounds of frames.
    for _ in range(10):
        try:
            raise shared_error
        except EngineDeadError:
            pass
    shared_tb_len = len(traceback.extract_tb(shared_error.__traceback__))
    assert shared_tb_len > 1

    for use_await in (False, True):
        collector = RequestOutputCollector(
            RequestOutputKind.CUMULATIVE, request_id="my-request-id-int"
        )
        collector.put(shared_error)

        with pytest.raises(EngineDeadError) as exc_info:
            if use_await:
                await collector.get()
            else:
                collector.get_nowait()

        fresh_error = exc_info.value
        assert fresh_error is not shared_error
        # The fresh error carries only this consumer's own frames, not the
        # accumulated traceback of the shared instance.
        assert len(traceback.extract_tb(fresh_error.__traceback__)) < shared_tb_len
        # The shared error must stay out of the logged chain.
        assert fresh_error.__suppress_context__


@pytest.mark.asyncio
async def test_collector_reraises_request_error_instance_unchanged():
    """Non-EngineDeadError exceptions are re-raised as the same instance."""
    error = ValueError("request-level error")
    collector = RequestOutputCollector(
        RequestOutputKind.CUMULATIVE, request_id="my-request-id-int"
    )
    collector.put(error)

    with pytest.raises(ValueError) as exc_info:
        await collector.get()

    assert exc_info.value is error
