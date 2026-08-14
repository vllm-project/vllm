# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import runpy
from pathlib import Path

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "MoRIIO toy proxy tests run only in the ROCm image.",
        allow_module_level=True,
    )

pytest.importorskip("quart")

_PROXY_SERVER = (
    Path(__file__).parents[4]
    / "examples"
    / "disaggregated"
    / "disaggregated_serving"
    / "moriio_toy_proxy_server.py"
)


def _load_proxy():
    namespace = runpy.run_path(str(_PROXY_SERVER))
    return namespace["handle_request"].__globals__


@pytest.mark.asyncio
async def test_health_route_returns_ok():
    proxy = _load_proxy()

    response = await proxy["app"].test_client().get("/health")

    assert response.status_code == 200
    assert await response.get_data(as_text=True) == "ok"
