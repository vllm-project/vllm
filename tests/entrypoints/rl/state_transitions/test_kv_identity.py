# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routers pair engines by `GET /kv_identity`, so it must follow weight
changes: unknown until the new weights are identified, then an identity other
engines with the same weights share."""

import json

import requests

from tests.entrypoints.rl.conftest import gen, ok, server

# With enforcement, the engine also resets the caches computed with the
# previous weights before publishing a new identity.
ENFORCED_OFFLOADING = [
    "--kv-transfer-config",
    json.dumps(
        {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": 1 << 30,
                "enforce_kv_identity": True,
            },
        }
    ),
]


def kv_identity(url: str) -> dict:
    response = requests.get(f"{url}/kv_identity", timeout=30)
    response.raise_for_status()
    return response.json()["0"]


def test_weight_changes_update_kv_identity():
    with server(extra_args=ENFORCED_OFFLOADING, port=8774) as url:
        startup = kv_identity(url)
        assert startup["components"]["weights_source"] == "hf_cache"

        requests.post(
            f"{url}/weight_checker", json={"action": "reset"}, timeout=300
        ).raise_for_status()
        assert kv_identity(url)["kv_identity"].startswith("unknown-")

        requests.post(
            f"{url}/collective_rpc", json={"method": "reload_weights"}, timeout=300
        ).raise_for_status()
        assert kv_identity(url) == startup
        assert ok(gen(url))

        requests.post(
            f"{url}/update_weight_version", json={"new_version": "step-1"}, timeout=30
        ).raise_for_status()
        versioned = kv_identity(url)
        assert versioned["components"]["weight_version"] == "step-1"
        assert versioned["kv_identity"] != startup["kv_identity"]
        assert ok(gen(url))
