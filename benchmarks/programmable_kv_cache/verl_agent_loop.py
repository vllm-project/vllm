# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate a verl Agent Loop trace into supported programmable-KV actions.

verl remains the workload owner. This adapter only emits the merged hint
records and consumes terminal operation results when a vLLM endpoint is given;
it never registers a custom hint type or handler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import requests


def load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    for line in path.read_text().splitlines():
        if line.strip():
            episode = json.loads(line)
            if not isinstance(episode, dict) or "session_id" not in episode:
                raise ValueError("Each episode must contain session_id")
            episodes.append(episode)
    return episodes


def action_plan(
    episodes: list[dict[str, Any]], weight_version: str
) -> list[dict[str, Any]]:
    namespace = hashlib.sha256(weight_version.encode()).hexdigest()[:32]
    actions = []
    for episode in episodes:
        session = str(episode["session_id"])
        rounds = int(episode.get("rounds", 1))
        if rounds < 1:
            raise ValueError("rounds must be positive")
        for round_id in range(rounds):
            continuation = f"round-{round_id}"
            actions.append(
                {
                    "session_id": session,
                    "continuation_id": continuation,
                    "weight_version": weight_version,
                    "namespace_digest": namespace,
                    "commands": ("retain", "prefetch", "release"),
                }
            )
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--weight-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    args = parser.parse_args()
    episodes = load_episodes(args.episodes)
    plan = action_plan(episodes, args.weight_version)
    report: dict[str, Any] = {
        "weight_version": args.weight_version,
        "episodes": len(episodes),
        "plan": plan,
        "executed": False,
    }
    if args.base_url:
        response = requests.get(args.base_url + "/v1/kv/capabilities", timeout=30)
        response.raise_for_status()
        engine = response.json()["engines"][0]
        invocation = {
            key: engine[key] for key in ("engine_id", "engine_epoch", "namespace")
        }
        results = []
        for item in plan:
            payload = {
                "protocol_version": "0.1",
                "message_id": f"verl-{uuid.uuid4().hex}",
                "actions": [
                    {
                        "action_id": "action",
                        "action_type": "signal",
                        "action_version": "1",
                        "payload": {
                            "target": {
                                "kind": "session",
                                "session_id": item["session_id"],
                                "continuation_id": item["continuation_id"],
                            },
                            "signal": "resume",
                        },
                    }
                ],
            }
            result = requests.post(
                args.base_url + "/v1/kv/control",
                json={"method": "submit", "invocation": invocation, "payload": payload},
                timeout=30,
            )
            result.raise_for_status()
            results.append(result.json()["results"][0])
        report.update({"executed": True, "engine": engine, "results": results})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "executed": report["executed"]}))


if __name__ == "__main__":
    main()
