# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Body handling across the EPD proxy's decode retries.

Exercises the REAL helpers loaded from the ``examples/`` proxy, so a future
change to them is what these tests catch.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

PROXY_REL = "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"


@pytest.fixture(scope="module")
def proxy():
    path = Path(__file__).parents[4] / PROXY_REL
    spec = importlib.util.spec_from_file_location("disagg_epd_proxy_retry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, params):
        self._params = params

    async def json(self):
        return {"kv_transfer_params": self._params}


def test_maybe_prefill_leaves_the_caller_body_untouched(proxy, monkeypatch):
    """A decode retry re-enters this function with the body it was given.

    Mutating that body in place let one attempt's `remote_block_ids` survive
    into the next, so a retry whose prefill returns nothing sent decode blocks
    the prefiller may already have freed.
    """
    served = [{"remote_block_ids": [1, 2]}, {}]

    async def _stage(req_data, p_url, req_id):
        assert "kv_transfer_params" not in req_data
        return _Response(served.pop(0))

    monkeypatch.setattr(proxy, "process_prefill_stage", _stage)

    body = {"messages": [], "stream": False}
    first = asyncio.run(proxy.maybe_prefill(body, "http://prefill", "r1"))
    assert first["kv_transfer_params"] == {"remote_block_ids": [1, 2]}
    assert "kv_transfer_params" not in body

    second = asyncio.run(proxy.maybe_prefill(body, "http://prefill", "r1"))
    assert "kv_transfer_params" not in second
