#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Make AITER's a16w4 MXFP4 MoE tile selection LDS-aware on CDNA3 (gfx942).

AITER's ``get_kernel_config`` picks one tile shape for every AMD architecture:
``block_k=256``, ``block_n=512``, ``num_stages=1``. Those suit CDNA4's 160 KiB
of LDS. CDNA3 has 64 KiB, so the prefill tile runs at very low occupancy there.

``aiter.ops.triton.utils._triton.arch_info._LDS_CAP_BYTES`` already records the
per-arch capacity and is consulted by pa_decode_sparse, gemm_config_utils and
sparse_attention_dsv4 -- the MoE GEMM is the one kernel that skips it.

Measured on MI325X with Kimi-K3 shapes (E=896, topk=16, TP=8, 4096 tokens):

    gemm1 K=3584 N=768  : 8.208 ms -> 1.303 ms  (6.30x)
    gemm2 K=384  N=7168 : 8.019 ms -> 1.701 ms  (4.72x)

End to end that is ~1.9x prefill throughput on a served Kimi-K3. Decode is
untouched: at block_m 16/32 the staging is 12-32 KiB, already well inside
64 KiB, so only tiles at or above block_m=64 are overridden.

This is a stopgap. The durable fix is upstream in ROCm/aiter, deriving
block_k from _LDS_CAP_BYTES instead of hardcoding it. Re-derive the constants
for other geometries with benchmarks/kernels/benchmark_moe_a16w4_tiles.py.

Usage:
    python tools/vllm-rocm/patch_aiter_cdna3_moe_tiles.py
    python tools/vllm-rocm/patch_aiter_cdna3_moe_tiles.py --revert
"""

import argparse
import importlib.util
import shutil
import sys

MODULE = "aiter.ops.triton.moe.moe_op_gemm_a16w4"
MARKER = "# --- CDNA3 (gfx942) LDS-aware tile override"

OVERRIDE = f'''

{MARKER} ---
# Appended by tools/vllm-rocm/patch_aiter_cdna3_moe_tiles.py. The stock
# function is left intact and still reachable as _stock_get_kernel_config.
_stock_get_kernel_config = get_kernel_config


def _cdna3_get_kernel_config(m, n, k, routing_data):
    cfg = _stock_get_kernel_config(m, n, k, routing_data)
    if cfg["block_m"] >= 64:
        # Halving block_k is the dominant win: it shrinks the A and B LDS
        # tiles together, restoring occupancy and making 2 stages viable.
        cfg["block_k"] = 128
        if n <= 1024:
            cfg["block_n"], cfg["num_warps"], cfg["num_stages"] = 128, 4, 1
        else:
            cfg["block_n"], cfg["num_warps"], cfg["num_stages"] = 512, 8, 2
    return cfg


try:
    from aiter.ops.triton.utils._triton import arch_info as _arch_info

    if _arch_info.get_arch() == "gfx942":
        get_kernel_config = _cdna3_get_kernel_config
except Exception:  # noqa: BLE001 - never break import on a probe failure
    pass
'''


def locate() -> str:
    spec = importlib.util.find_spec(MODULE)
    if spec is None or not spec.origin:
        raise SystemExit(f"cannot locate {MODULE}; is aiter installed?")
    return spec.origin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    target = locate()
    backup = target + ".orig"

    if args.revert:
        try:
            shutil.copyfile(backup, target)
        except FileNotFoundError:
            print("no backup found; nothing to revert")
            return 0
        print("reverted", target)
        return 0

    source = open(target).read()
    if MARKER in source:
        print("already patched:", target)
        return 0

    try:
        open(backup).close()
    except FileNotFoundError:
        shutil.copyfile(target, backup)
        print("backed up to", backup)

    open(target, "w").write(source + OVERRIDE)
    print("patched", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
