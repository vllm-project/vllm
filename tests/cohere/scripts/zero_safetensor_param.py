#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mirror a HF/safetensors checkpoint with one parameter zeroed out.

All files in the source dir are symlinked into the destination, except
the safetensors shard that contains the named parameter — that shard is
rewritten with the parameter replaced by a zero tensor of identical
shape/dtype.

Used by the ``collective_rpc_reload`` test to boot a vLLM server with a
deliberately broken model that can then be fixed in place via
``/collective_rpc reload_weights``. This makes the round-trip a real
behavior test (broken model → reload → working model) rather than a
preservation test (real model → reload → real model).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _mirror_with_zeroed_param(src: Path, dst: Path, param_name: str) -> None:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        sys.exit(f"index file not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    if param_name not in weight_map:
        sys.exit(
            f"param {param_name!r} not in weight_map (checked {len(weight_map)} keys)"
        )

    target_shard = weight_map[param_name]
    print(f"Target shard for {param_name}: {target_shard}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for entry in src.iterdir():
        if entry.name == target_shard:
            continue
        os.symlink(entry, dst / entry.name)

    src_shard = src / target_shard
    dst_shard = dst / target_shard

    tensors: dict[str, torch.Tensor] = {}
    # `safe_open` exposes `.keys()` / `.get_tensor()` / `.metadata()` but
    # does not implement `__iter__`, so `for key in f:` raises TypeError.
    # Use `.keys()` explicitly.
    with safe_open(src_shard, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():  # noqa: SIM118
            t = f.get_tensor(key)
            if key == param_name:
                print(f"Zeroing {key} (shape={list(t.shape)}, dtype={t.dtype})")
                t = torch.zeros_like(t)
            tensors[key] = t

    if param_name not in tensors:
        sys.exit(
            f"param {param_name!r} was in the index but not produced by "
            f"safe_open iteration — refusing to write a mirror that would "
            f"silently leave the original weight intact"
        )

    save_file(tensors, str(dst_shard), metadata=metadata)
    print(f"Wrote corrupted shard: {dst_shard}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src-dir",
        required=True,
        help="Path to the original checkpoint directory.",
    )
    ap.add_argument(
        "--dst-dir",
        required=True,
        help="Path to write the corrupted mirror directory to.",
    )
    ap.add_argument(
        "--param-name",
        required=True,
        help="Fully-qualified parameter name to zero out.",
    )
    args = ap.parse_args()
    _mirror_with_zeroed_param(
        Path(args.src_dir).resolve(),
        Path(args.dst_dir).resolve(),
        args.param_name,
    )


if __name__ == "__main__":
    main()
