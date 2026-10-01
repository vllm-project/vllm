# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise duplicate, version, event-gap and restart recovery over HTTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = requests.Session()
    engines = client.get(args.base_url + "/v1/kv/capabilities", timeout=30)
    engines.raise_for_status()
    engine = engines.json()["engines"][0]
    invocation = {
        key: engine[key] for key in ("engine_id", "engine_epoch", "namespace")
    }

    def call(method: str, payload: dict[str, Any], route: dict[str, Any] | None = None):
        response = client.post(
            args.base_url + "/v1/kv/control",
            json={
                "method": method,
                "invocation": {**invocation, **(route or {})},
                "payload": payload,
            },
            timeout=30,
        )
        return response.status_code, response.json()

    envelope = {
        "protocol_version": "0.1",
        "message_id": "recovery-signal",
        "actions": [
            {
                "action_id": "action",
                "action_type": "signal",
                "action_version": "1",
                "payload": {
                    "target": {"kind": "session", "session_id": "recovery"},
                    "signal": "tool_wait",
                },
            }
        ],
    }
    status, first = call("submit", envelope)
    if status != 200 or not first.get("results"):
        raise RuntimeError(first)
    _, replay = call("submit", envelope)
    if not replay.get("replayed"):
        raise AssertionError("duplicate envelope was not replayed")
    _, unknown = call(
        "submit",
        {
            **envelope,
            "message_id": "unknown-version",
            "actions": [{**envelope["actions"][0], "action_version": "future"}],
        },
    )
    _, gap = call("events", {"stream": "operation", "after": 1 << 60})
    _, stale = call("events", {"stream": "operation"}, {"engine_epoch": "restarted"})
    report = {
        "engine": engine,
        "duplicate": replay,
        "unknown_version": unknown,
        "event_gap": gap,
        "restart": stale,
    }
    if stale.get("error", {}).get("code") != "stale_epoch":
        raise AssertionError(stale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": True}))


if __name__ == "__main__":
    main()
