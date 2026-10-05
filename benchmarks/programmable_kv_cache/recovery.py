# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check HTTP control replay, version rejection and invalid epoch/cursor errors.

This protocol probe does not restart an engine, create transfer failures or
validate recovery of GPU residency. See http_control.py for real movement checks.
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import requests


def check_protocol(
    client: requests.Session, base_url: str, engine_id: str
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    response = client.get(base_url + "/v1/kv/capabilities", timeout=30)
    response.raise_for_status()
    engines = [e for e in response.json()["engines"] if e["engine_id"] == engine_id]
    if len(engines) != 1:
        raise ValueError("Select exactly one advertised engine_id")
    engine = engines[0]
    invocation = {
        key: engine[key] for key in ("engine_id", "engine_epoch", "namespace")
    }

    def call(method, payload, *, route=None, expected_error=None):
        response = client.post(
            base_url + "/v1/kv/control",
            json={
                "method": method,
                "invocation": {**invocation, **(route or {})},
                "payload": payload,
            },
            timeout=30,
        )
        result = response.json()
        if expected_error is None:
            response.raise_for_status()
        else:
            status, code = expected_error
            if (
                response.status_code != status
                or result.get("error", {}).get("code") != code
            ):
                raise AssertionError((expected_error, response.status_code, result))
        return result

    # An unsupported version has no semantic effects, including with policies enabled.
    envelope = {
        "protocol_version": "0.1",
        "message_id": f"protocol-probe-{uuid.uuid4().hex}",
        "actions": [
            {
                "action_id": "action",
                "action_type": "signal",
                "action_version": "future",
                "payload": {},
            }
        ],
    }
    first = call("submit", envelope)
    if first["replayed"] or first["results"][0]["status"] != "unsupported":
        raise AssertionError(first)
    replay = call("submit", envelope)
    if not replay["replayed"] or replay["results"] != first["results"]:
        raise AssertionError(replay)
    queried = call(
        "operation", {"message_id": envelope["message_id"], "action_id": "action"}
    )
    if queried != first["results"][0]:
        raise AssertionError(queried)
    gap = call(
        "events",
        {"stream": "operation", "after": 1 << 60},
        expected_error=(409, "event_gap"),
    )
    stale_epoch = uuid.uuid4().hex
    while stale_epoch == engine["engine_epoch"]:
        stale_epoch = uuid.uuid4().hex
    stale = call(
        "events",
        {"stream": "operation"},
        route={"engine_epoch": stale_epoch},
        expected_error=(409, "stale_epoch"),
    )
    return {
        "engine": engine,
        "checks": "protocol_only",
        "unknown_version": first,
        "duplicate": replay,
        "operation": queried,
        "invalid_event_cursor": gap,
        "stale_epoch": stale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with requests.Session() as client:
        report = check_protocol(client, args.base_url, args.engine_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "checks": "protocol_only", "passed": True}
        )
    )


if __name__ == "__main__":
    main()
