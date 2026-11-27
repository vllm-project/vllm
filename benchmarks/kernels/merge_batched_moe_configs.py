# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge per-capacity tuned config files from benchmark_batched_moe.py into
one final config file, ready to drop into VLLM_TUNED_CONFIG_FOLDER.

Adapted from xpu_kernel_autotune_tutorial.md section 8's fused-MoE merge
script: same completeness guard (a missing capacity fails loudly here
instead of silently falling back to a neighbor's config at lookup time,
fused_moe.py:1447) and the same single-triton-version guard.

Usage:
    python3 benchmarks/kernels/merge_batched_moe_configs.py \\
        --root ./tuned \\
        --num-experts 32 --intermediate-size 768 \\
        --expected 1024 2048 4096 --out ./tuned/merged
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    get_config_file_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="dir containing capacity_*/ subdirs"
    )
    parser.add_argument("--num-experts", "-E", type=int, required=True)
    parser.add_argument("--intermediate-size", "-N", type=int, required=True)
    parser.add_argument(
        "--expected",
        type=int,
        nargs="+",
        required=True,
        help="every capacity that must be present",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    filename = get_config_file_name(
        args.num_experts, args.intermediate_size, dtype=None, block_shape=None
    )
    paths = glob.glob(os.path.join(args.root, "**", filename), recursive=True)
    if not paths:
        raise SystemExit(
            f"no config files matching {filename!r} found under {args.root}"
        )

    merged: dict[int, dict] = {}
    versions: set[str] = set()
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        versions.add(data.pop("triton_version"))
        for key, config in data.items():
            capacity = int(key)
            if capacity in merged and merged[capacity] != config:
                raise SystemExit(
                    f"conflicting config for capacity {capacity} (see {path})"
                )
            merged[capacity] = config

    missing = sorted(set(args.expected) - set(merged))
    if missing:
        raise SystemExit(f"missing capacities: {missing}")
    if len(versions) != 1:
        raise SystemExit(f"mixed Triton versions across inputs: {versions}")

    result = {"triton_version": versions.pop()}
    result.update({str(c): merged[c] for c in sorted(args.expected)})

    os.makedirs(args.out, exist_ok=True)
    destination = os.path.join(args.out, filename)
    tmp = destination + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=4)
        f.write("\n")
    os.replace(tmp, destination)
    print(destination)


if __name__ == "__main__":
    main()
