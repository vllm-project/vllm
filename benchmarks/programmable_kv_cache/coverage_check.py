# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that the programmable-KV coverage inventory has concrete artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {
    "control": [
        "vllm/v1/kv_hints/control.py",
        "tests/v1/test_kv_hint_control.py",
        "tests/entrypoints/serve/kv_control/test_api.py",
    ],
    "state": ["vllm/v1/kv_hints/state.py", "tests/v1/test_kv_hint_state.py"],
    "movement": [
        "vllm/v1/kv_hints/transfers.py",
        "vllm/v1/core/kv_cache_transfer.py",
        "benchmarks/programmable_kv_cache/http_control.py",
    ],
    "policy": [
        "vllm/v1/kv_hints/policy.py",
        "vllm/v1/kv_hints/policy_runtime.py",
        "tests/v1/core/test_scheduler.py",
    ],
    "workloads": [
        "benchmarks/programmable_kv_cache/rollout.py",
        "benchmarks/programmable_kv_cache/recovery.py",
        "benchmarks/programmable_kv_cache/evaluate.py",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        name: {
            "implemented": all((args.root / path).is_file() for path in files),
            "files": files,
        }
        for name, files in REQUIRED.items()
    }
    missing = {
        name: [path for path in files if not (args.root / path).is_file()]
        for name, files in REQUIRED.items()
        if any(not (args.root / path).is_file() for path in files)
    }
    result = {"rows": report, "missing": missing, "complete": not missing}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
