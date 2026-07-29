# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from vllm.entrypoints.serve.dev.rlhf.api_router import (
    get_weight_update_baseline,
)


def test_weight_update_baseline_endpoint_returns_engine_manifest() -> None:
    baseline = {
        "ready": True,
        "scope_template": {
            "kind": "base_checkpoint",
            "mode": "partial",
            "source_names": [],
        },
        "source_names": ["layer.weight"],
        "atomic_source_groups": [["layer.weight"]],
        "workers": [],
        "reason": None,
    }
    engine = SimpleNamespace(
        get_weight_update_baseline=AsyncMock(return_value=baseline)
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine))
    )

    response = asyncio.run(get_weight_update_baseline(request))

    assert response.status_code == 200
    assert json.loads(response.body) == baseline
    engine.get_weight_update_baseline.assert_awaited_once_with()
