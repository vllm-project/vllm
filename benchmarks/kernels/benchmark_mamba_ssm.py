# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tuning script for the Mamba selective_state_update Triton kernel.

Sweeps (BLOCK_SIZE_M, num_warps) combinations for given (dstate, headdim)
shapes and writes the best config per shape to a JSON file that
vllm/model_executor/layers/mamba/ops/mamba_ssm.py will load at runtime.

Usage:
    # Tune for default shapes and save configs to the in-tree configs dir:
    python benchmarks/kernels/benchmark_mamba_ssm.py --save-dir \
        vllm/model_executor/layers/mamba/configs/

    # Tune specific shapes:
    python benchmarks/kernels/benchmark_mamba_ssm.py \
        --dstates 64 128 256 \
        --headdims 64 128 \
        --batch-size 32

    # Dry-run: print best configs without saving:
    python benchmarks/kernels/benchmark_mamba_ssm.py --no-save
"""

import argparse
import json
import os
from itertools import product
from typing import Optional

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    _selective_scan_update_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Candidate kernel configs to sweep
CANDIDATE_BLOCK_SIZE_M = [4, 8, 16, 32, 64]
CANDIDATE_NUM_WARPS = [1, 2, 4, 8]

# Default shapes to tune for (common Mamba2 model configs)
DEFAULT_DSTATES = [16, 32, 64, 128, 256]
DEFAULT_HEADDIMS = [64, 128]


def get_config_file_name(dstate: int, headdim: int) -> str:
    device_name = current_platform.get_device_name().replace(" ", "_")
    return f"dstate={dstate},headdim={headdim},device_name={device_name}.json"


def benchmark_one_config(
    block_size_m: int,
    num_warps: int,
    dstate: int,
    headdim: int,
    nheads: int,
    batch_size: int,
    dtype: torch.dtype,
) -> Optional[float]:
    """
    Benchmark a single (BLOCK_SIZE_M, num_warps) config.

    Returns the median latency in milliseconds, or None if the config is
    invalid (e.g. BLOCK_SIZE_M > headdim or headdim % BLOCK_SIZE_M != 0).
    """
    if headdim % block_size_m != 0:
        return None

    device = torch.device("cuda")
    ngroups = 1

    # Allocate tensors matching the shapes expected by selective_state_update.
    # We use 4-D tensors since selective_state_update unsqueezes internally,
    # but here we call the kernel directly so we build the right shapes.
    state = torch.randn(batch_size, nheads, headdim, dstate, dtype=dtype,
                        device=device)
    x = torch.randn(batch_size, nheads, headdim, dtype=dtype, device=device)
    dt = torch.randn(batch_size, nheads, headdim, dtype=dtype, device=device)
    # A must be negative (stability), stored as float32
    A = -torch.rand(nheads, headdim, dstate, dtype=torch.float32,
                    device=device)
    B = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch_size, ngroups, dstate, dtype=dtype, device=device)
    dt_bias = torch.zeros(nheads, headdim, dtype=dtype, device=device)
    out = torch.zeros_like(x)
    # dst_state mirrors state for in-place update
    dst_state = state.clone()

    grid = (triton.cdiv(headdim, block_size_m), batch_size, nheads)

    def run():
        _selective_scan_update_kernel[grid](
            state,
            None,   # rand_seed_ptr
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            None,   # D_ptr
            None,   # z_ptr
            out,
            None,   # state_batch_indices_ptr
            dst_state,
            -1,     # null_block_id
            None,   # num_accepted_tokens_ptr
            None,   # cu_seqlens_ptr
            # Matrix dimensions
            batch_size,
            nheads,
            headdim,
            dstate,
            nheads // ngroups,
            # Strides for state (batch, head, dim, dstate)
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            # Strides for x (batch, head, dim)
            x.stride(0),
            x.stride(1),
            x.stride(2),
            # Strides for dt (batch, head, dim)
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            # Strides for dt_bias (head, dim)
            dt_bias.stride(0),
            dt_bias.stride(1),
            # Strides for A (head, dim, dstate)
            A.stride(0),
            A.stride(1),
            A.stride(2),
            # Strides for B (batch, group, dstate)
            B.stride(0),
            B.stride(1),
            B.stride(2),
            # Strides for C (batch, group, dstate)
            C.stride(0),
            C.stride(1),
            C.stride(2),
            # D strides (unused)
            0,
            0,
            # z strides (unused)
            0,
            0,
            0,
            # Strides for out (batch, head, dim)
            out.stride(0),
            out.stride(1),
            out.stride(2),
            # state_batch_indices strides (unused)
            0,
            0,
            # dst_state_batch_indices strides
            dst_state.stride(0),
            dst_state.stride(1),
            # Meta-parameters
            False,          # DT_SOFTPLUS
            False,          # TIE_HDIM
            block_size_m,   # BLOCK_SIZE_M
            num_warps=num_warps,
            USE_RS_ROUNDING=False,
            PHILOX_ROUNDS=0,
        )

    try:
        # Warmup + benchmark
        ms = triton.testing.do_bench(run, warmup=25, rep=100)
        return ms
    except Exception:
        return None


def tune_shape(
    dstate: int,
    headdim: int,
    batch_size: int,
    dtype: torch.dtype,
) -> dict:
    """Find the best (BLOCK_SIZE_M, num_warps) for a given (dstate, headdim)."""
    # Use enough heads to have representative occupancy
    nheads = max(1, 512 // headdim)

    best_ms = float("inf")
    best_config = {"BLOCK_SIZE_M": 4, "num_warps": 8}

    print(
        f"  Tuning dstate={dstate}, headdim={headdim}, "
        f"nheads={nheads}, batch_size={batch_size}"
    )

    for block_size_m, num_warps in product(
        CANDIDATE_BLOCK_SIZE_M, CANDIDATE_NUM_WARPS
    ):
        ms = benchmark_one_config(
            block_size_m=block_size_m,
            num_warps=num_warps,
            dstate=dstate,
            headdim=headdim,
            nheads=nheads,
            batch_size=batch_size,
            dtype=dtype,
        )
        if ms is None:
            continue
        if ms < best_ms:
            best_ms = ms
            best_config = {"BLOCK_SIZE_M": block_size_m, "num_warps": num_warps}
        print(
            f"    BLOCK_SIZE_M={block_size_m:3d}, num_warps={num_warps}: "
            f"{ms:.3f} ms"
            + (" <-- best" if best_config["BLOCK_SIZE_M"] == block_size_m
               and best_config["num_warps"] == num_warps else "")
        )

    print(
        f"  Best: BLOCK_SIZE_M={best_config['BLOCK_SIZE_M']}, "
        f"num_warps={best_config['num_warps']} ({best_ms:.3f} ms)\n"
    )
    return best_config


def main():
    parser = FlexibleArgumentParser(
        description="Tune Mamba selective_state_update kernel configs"
    )
    parser.add_argument(
        "--dstates",
        nargs="+",
        type=int,
        default=DEFAULT_DSTATES,
        help="State dimensions to tune for",
    )
    parser.add_argument(
        "--headdims",
        nargs="+",
        type=int,
        default=DEFAULT_HEADDIMS,
        help="Head dimensions to tune for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (number of decode tokens) to use during tuning",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for tuning",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../vllm/model_executor/layers/mamba/configs",
        ),
        help="Directory to write JSON config files",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print results without writing config files",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this tuning script")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    device_name = current_platform.get_device_name()
    print(f"Tuning Mamba SSM kernel on: {device_name}")
    print(f"dtype={args.dtype}, batch_size={args.batch_size}\n")

    results = {}
    for dstate, headdim in product(args.dstates, args.headdims):
        config = tune_shape(
            dstate=dstate,
            headdim=headdim,
            batch_size=args.batch_size,
            dtype=dtype,
        )
        results[(dstate, headdim)] = config

    if args.no_save:
        print("Results (not saved):")
        for (dstate, headdim), config in results.items():
            print(f"  dstate={dstate}, headdim={headdim}: {config}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    for (dstate, headdim), config in results.items():
        filename = get_config_file_name(dstate, headdim)
        filepath = os.path.join(args.save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")
        print(f"Wrote {filepath}")


if __name__ == "__main__":
    main()
