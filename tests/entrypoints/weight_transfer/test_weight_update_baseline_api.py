# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from vllm.entrypoints.serve.dev.rlhf.api_router import (
    get_weight_update_manifest,
)


def test_weight_update_manifest_endpoint_returns_engine_manifest() -> None:
    manifest = {
        "model_weights": {
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
        },
        "lora_adapters": [],
    }
    engine = SimpleNamespace(
        get_weight_update_manifest=AsyncMock(return_value=manifest)
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine))
    )

    response = asyncio.run(get_weight_update_manifest(request))

    assert response.status_code == 200
    assert json.loads(response.body) == manifest
    engine.get_weight_update_manifest.assert_awaited_once_with()
