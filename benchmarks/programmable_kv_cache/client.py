# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small control helpers shared by the real-server examples."""

import time
from typing import Any

ACTIVE = {"received", "queued", "accepted", "running"}


def envelope(message_id: str, kind: str, payload: dict[str, Any], version: str = "1.0"):
    return {
        "protocol_version": "0.1",
        "message_id": message_id,
        "actions": [
            {
                "action_id": "action",
                "action_type": kind,
                "action_version": version,
                "payload": payload,
            }
        ],
    }


def wait_operation(
    call, message_id: str, action_id: str = "action", *, timeout: float = 120
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = call("operation", {"message_id": message_id, "action_id": action_id})
        if result["message_id"] != message_id or result["action_id"] != action_id:
            raise AssertionError("Control result lost caller correlation")
        if result["status"] not in ACTIVE:
            return result
        time.sleep(0.02)
    raise TimeoutError(f"KV operation {message_id}/{action_id} did not finish")
