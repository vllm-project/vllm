# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a deterministic multi-round shared-prefix control workload.

The generated plan is useful without a server; with ``--base-url`` it sends the
same supported native actions through the HTTP control endpoint and checks every
terminal result. The workload deliberately keeps logical association, retention
and release separate from physical residency.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import requests


def make_plan(
    rounds: int, branches: int, pressure_blocks: int, seed: int
) -> list[dict[str, Any]]:
    if rounds < 1 or branches < 1 or pressure_blocks < 1:
        raise ValueError("rounds, branches and pressure_blocks must be positive")
    rng = random.Random(seed)
    plan = []
    for round_id in range(rounds):
        for branch_id in range(branches):
            session = f"rollout-{branch_id}"
            continuation = f"round-{round_id}"
            plan.append(
                {
                    "round": round_id,
                    "branch": branch_id,
                    "session_id": session,
                    "continuation_id": continuation,
                    "pause_seconds": round(rng.random() * 0.01, 6),
                    "pressure_blocks": pressure_blocks,
                    "actions": ["associate", "retain", "resume_prediction", "release"],
                }
            )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--pressure-blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    plan = make_plan(args.rounds, args.branches, args.pressure_blocks, args.seed)
    report: dict[str, Any] = {
        "seed": args.seed,
        "rounds": args.rounds,
        "branches": args.branches,
        "pressure_blocks": args.pressure_blocks,
        "plan": plan,
        "executed": False,
    }
    if args.base_url:
        client = requests.Session()
        capabilities = client.get(args.base_url + "/v1/kv/capabilities", timeout=30)
        capabilities.raise_for_status()
        engine = capabilities.json()["engines"][0]
        invocation = {
            key: engine[key] for key in ("engine_id", "engine_epoch", "namespace")
        }
        run_id = uuid.uuid4().hex
        results = []

        def submit(
            item: dict[str, Any], action_type: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            message_id = f"{run_id}-{item['round']}-{item['branch']}-{action_type}"
            body = {
                "method": "submit",
                "invocation": invocation,
                "payload": {
                    "protocol_version": "0.1",
                    "message_id": message_id,
                    "actions": [
                        {
                            "action_id": "action",
                            "action_type": action_type,
                            "action_version": "1",
                            "payload": payload,
                        }
                    ],
                },
            }
            response = client.post(
                args.base_url + "/v1/kv/control", json=body, timeout=30
            )
            response.raise_for_status()
            result = response.json()["results"][0]
            results.append(result)
            return result

        for item in plan:
            selector = {
                "kind": "hashes",
                "namespace": engine["namespace"],
                "blocks": [],
            }
            associated = submit(
                item,
                "associate",
                {
                    "session_id": item["session_id"],
                    "continuation_id": item["continuation_id"],
                    "expected_revision": 0,
                    "target": selector,
                },
            )
            if associated["status"] not in ("completed", "partial"):
                raise RuntimeError(associated)
            target = {
                "kind": "session",
                "session_id": item["session_id"],
                "continuation_id": item["continuation_id"],
            }
            retained = submit(
                item,
                "retain",
                {
                    "target": target,
                    "owner_id": item["session_id"],
                    "declaration_id": f"{item['continuation_id']}-retention",
                    "priority": 50,
                    "ttl_seconds": max(1.0, item["pause_seconds"] + 1.0),
                },
            )
            if retained["status"] != "completed":
                raise RuntimeError(retained)
            prepared = submit(
                item,
                "prefetch",
                {"target": target, "source": "mooncake"},
            )
            if prepared["status"] not in (
                "completed",
                "partial",
                "unsupported",
                "missing",
                "deferred",
            ):
                raise RuntimeError(prepared)
            time.sleep(item["pause_seconds"])
            released = submit(
                item,
                "release",
                {
                    "owner_id": item["session_id"],
                    "declaration_id": f"{item['continuation_id']}-retention",
                },
            )
            if released["status"] not in ("completed", "missing"):
                raise RuntimeError(released)
        report.update({"executed": True, "engine": engine, "results": results})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "executed": report["executed"],
                "steps": len(plan),
            }
        )
    )


if __name__ == "__main__":
    main()
