# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the elastic EP client; no model or tokenizer, so they
collect offline."""

import asyncio

import pytest

from vllm.v1.engine.core_client import DPLBAsyncMPClient


@pytest.mark.asyncio
async def test_elastic_ep_scale_waits_for_removed_engines():
    """A scale right after a scale-down waits for the removed engines'
    shutdown, and gives up with an error if they never finish."""
    client = object.__new__(DPLBAsyncMPClient)
    client.eep_scaling_cache = object()

    async def finish_shutdown():
        await asyncio.sleep(0.05)
        client.eep_scaling_cache = None

    shutdown = asyncio.create_task(finish_shutdown())
    await client._wait_for_scale_down_cleanup(timeout_s=5)
    assert shutdown.done()

    client.eep_scaling_cache = object()
    with pytest.raises(RuntimeError, match="did not shut down"):
        await client._wait_for_scale_down_cleanup(timeout_s=0.2)
