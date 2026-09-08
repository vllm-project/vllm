# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability contract for transporting DSpark prompt-context KV."""

from typing import Any

DSPARK_CONTEXT_KV_TRANSPORT = "dspark_context_kv_v1"


def dspark_context_kv_transport_enabled(extra_config: dict[str, Any]) -> bool:
    """Return whether the connector explicitly enables DSpark context KV."""
    return (
        extra_config.get("dspark_context_transport_policy")
        == DSPARK_CONTEXT_KV_TRANSPORT
    )
