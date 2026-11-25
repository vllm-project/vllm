# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM


@pytest.mark.asyncio
async def test_kv_transfer_rejection_uses_internal_request():
    engine = object.__new__(AsyncLLM)
    add_request = AsyncMock()
    engine.engine_core = SimpleNamespace(
        add_request_async=add_request,
        shutdown=lambda **kwargs: None,
    )

    transfer_params = {"do_remote_prefill": True, "remote_request_id": "prefill"}
    for _ in range(2):
        await engine.notify_kv_transfer_request_rejected(
            "request", transfer_params, data_parallel_rank=1
        )

    first, second = [call.args[0] for call in add_request.await_args_list]
    assert first.request_id != second.request_id
    for request in (first, second):
        assert request.request_id != "request"
        assert request.external_req_id == "request"
        assert request.abort_immediately
        assert request.data_parallel_rank == 1
        assert request.sampling_params.watermarking is False
        assert request.sampling_params.extra_args == {
            "kv_transfer_params": transfer_params
        }
