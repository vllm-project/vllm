# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.mrope import MROPE_POSITION_STATE_KEY, MRoPEPositionState


@pytest.mark.asyncio
async def test_kv_transfer_rejection_stub_disables_watermarking():
    engine = object.__new__(AsyncLLM)
    add_request = AsyncMock()
    engine.engine_core = SimpleNamespace(
        add_request_async=add_request,
        shutdown=lambda **kwargs: None,
    )

    await engine.notify_kv_transfer_request_rejected(
        "request",
        {
            "do_remote_prefill": True,
            MROPE_POSITION_STATE_KEY: MRoPEPositionState(
                positions=[[0, 1], [0, 2], [0, 3]], delta=-1
            ).to_payload(),
        },
    )

    request = add_request.call_args.args[0]
    assert request.sampling_params.watermarking is False
    params = request.sampling_params.extra_args["kv_transfer_params"]
    assert MROPE_POSITION_STATE_KEY not in params
