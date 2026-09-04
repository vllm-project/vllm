# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the MI300 fused MXFP4 SwiGLU stage-1 fast path.

Times ``triton_kernel_moe_forward`` at the gpt-oss-20b decode geometry
(hidden=3072, intermediate=3072, 32 experts, top-4) with the fast path
in ``vllm/model_executor/layers/fused_moe/experts/gpt_oss_mi300_swiglu_stage1.py``
enabled vs. disabled (via the ``VLLM_DISABLE_MI300_GPTOSS_SWIGLU=1``
kill-switch).

Stage 2 (the ``w2`` matmul + reduce) is identical between the two paths,
so the per-shape delta is entirely the stage-1 fused-MXFP4-SwiGLU saving.

Usage:

    python benchmarks/kernels/benchmark_gpt_oss_mi300_swiglu_stage1.py \\
        --num-tokens 1 4 8 16 32 64 \\
        --save-json /tmp/mi300_stage1.json

Requirements: gfx942 (MI300X / MI325X) host with ``triton_kernels`` installed
(the same prerequisites the fast path itself gates on).
"""

import gc
import json
import os
import sys

import torch
import torch.utils.benchmark as benchmark

from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.import_utils import has_triton_kernels

# gpt-oss-20b decode geometry; the fast path is gated to exactly this shape.
HIDDEN_DIM = 3072
INTERMEDIATE_DIM = 3072
NUM_EXPERTS = 32
EXPERTS_PER_TOKEN = 4

# Default sweep: gather_rows = M * topk = {4, 16, 32, 64, 128, 256}, all in
# the validated set ``_MI300_GATHER_ROWS`` inside the kernel module.
DEFAULT_NUM_TOKENS = [1, 4, 8, 16, 32, 64]

KILL_SWITCH = "VLLM_DISABLE_MI300_GPTOSS_SWIGLU"


def _check_platform() -> None:
    """Fail fast (with a useful message) on non-target hardware."""
    if not current_platform.is_rocm():
        sys.exit(
            "error: this benchmark requires ROCm; the MI300 fast path is AMD-only."
        )
    if not current_platform.is_device_capability((9, 4)):
        sys.exit(
            "error: this benchmark targets gfx942 (MI300X / MI325X). "
            "The fast path will silently fall through on any other arch, "
            "making the comparison meaningless."
        )
    if not has_triton_kernels():
        sys.exit("error: `triton_kernels` is not installed in this environment.")


def _build_inputs(M: int, num_warps: int):
    # Reuse the canonical test fixture so the inputs match exactly what the
    # production code path consumes, including the MXFP4 weight layout and
    # bias dtypes.  Same precedent as ``benchmark_cutlass_moe_nvfp4.py``
    # importing from ``tests.kernels.moe.utils``.
    from tests.kernels.moe.test_gpt_oss_triton_kernels import init_compute_data

    return init_compute_data(
        M,
        HIDDEN_DIM,
        INTERMEDIATE_DIM,
        NUM_EXPERTS,
        "bf16",
        "mx4",
        num_warps=num_warps,
    )


def _build_quant_config(pc1, pc2, w1_bias_tri, w2_bias_tri):
    from vllm.model_executor.layers.fused_moe.config import (
        mxfp4_w4a16_moe_quant_config,
    )

    return mxfp4_w4a16_moe_quant_config(
        w1_scale=pc1,
        w2_scale=pc2,
        w1_bias=w1_bias_tri,
        w2_bias=w2_bias_tri,
    )


def _time_pair(
    M: int, min_run_time: float, seed: int
) -> tuple[benchmark.Measurement, benchmark.Measurement]:
    """Time baseline and optimized paths for a single (M,) shape.

    Both measurements are taken inside the same process with identical
    inputs; only the ``VLLM_DISABLE_MI300_GPTOSS_SWIGLU`` env var differs
    between them, which is exactly the toggle ``run_mi300_swiglu_stage1``
    checks on every call.
    """
    torch.manual_seed(seed)
    (
        _x_ref,
        _w1_ref,
        _w1b_ref,
        _w2_ref,
        _w2b_ref,
        _exp_ref,
        x_tri,
        w1_tri,
        w2_tri,
        exp_data_tri,
        w1_bias_tri,
        w2_bias_tri,
        pc1,
        pc2,
    ) = _build_inputs(M, num_warps=8)

    quant_config = _build_quant_config(pc1, pc2, w1_bias_tri, w2_bias_tri)

    from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
        triton_kernel_moe_forward,
    )

    def _forward():
        return triton_kernel_moe_forward(
            hidden_states=x_tri,
            w1=w1_tri,
            w2=w2_tri,
            gating_output=exp_data_tri,
            topk=EXPERTS_PER_TOKEN,
            renormalize=True,
            quant_config=quant_config,
        )

    globals_dict = {"fn": _forward}

    # Optimized path: kill switch cleared.  Warm up so triton autotuner
    # caches plans; otherwise the first measured iteration would include
    # compilation cost.
    os.environ.pop(KILL_SWITCH, None)
    for _ in range(3):
        _forward()
    torch.accelerator.synchronize()
    opt = benchmark.Timer(
        stmt="fn()",
        globals=globals_dict,
        label="gpt-oss-20b stage-1",
        sub_label=f"M={M}, gather_rows={M * EXPERTS_PER_TOKEN}",
        description="optimized (MI300 fused MXFP4 SwiGLU)",
    ).blocked_autorange(min_run_time=min_run_time)

    # Baseline path: kill switch forced on.  Warm up again so the generic
    # ``matmul_ogs`` plan gets cached too.
    os.environ[KILL_SWITCH] = "1"
    try:
        for _ in range(3):
            _forward()
        torch.accelerator.synchronize()
        base = benchmark.Timer(
            stmt="fn()",
            globals=globals_dict,
            label="gpt-oss-20b stage-1",
            sub_label=f"M={M}, gather_rows={M * EXPERTS_PER_TOKEN}",
            description="baseline (matmul_ogs + fused_swiglu)",
        ).blocked_autorange(min_run_time=min_run_time)
    finally:
        os.environ.pop(KILL_SWITCH, None)

    return base, opt


def main() -> None:
    parser = FlexibleArgumentParser(
        description=(
            "Microbenchmark for the MI300 fused MXFP4 SwiGLU stage-1 fast "
            "path. Times triton_kernel_moe_forward at the gpt-oss-20b decode "
            "shape with the fast path on vs. off."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        "-M",
        type=int,
        nargs="+",
        default=DEFAULT_NUM_TOKENS,
        help="Tokens per micro-batch (gather_rows = M * topk). Defaults to "
        "the validated sweep {1,4,8,16,32,64}.",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=1.0,
        help="Minimum run-time per Timer (seconds). Higher = tighter "
        "confidence intervals; default 1.0s.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Path to dump per-shape JSON (M, gather_rows, baseline_us, "
        "optimized_us, speedup).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-shape torch.utils.benchmark.Measurement summaries.",
    )
    args = parser.parse_args()

    _check_platform()

    cap = current_platform.get_device_capability()
    print(f"GPU:                    {torch.cuda.get_device_name(0)}")
    print(f"Compute capability:     {cap}")
    print(
        f"Shape:                  hidden={HIDDEN_DIM}, "
        f"inter={INTERMEDIATE_DIM}, E={NUM_EXPERTS}, "
        f"topk={EXPERTS_PER_TOKEN}"
    )
    print(f"Min run-time per Timer: {args.min_run_time:.2f}s")
    print()

    rows: list[dict] = []
    header = (
        f"{'M':>4} {'gather_rows':>12} {'baseline_us':>14} "
        f"{'optimized_us':>14} {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))

    for M in args.num_tokens:
        gc.collect()
        torch.accelerator.empty_cache()
        base, opt = _time_pair(M, args.min_run_time, args.seed)
        b_us = base.median * 1e6
        o_us = opt.median * 1e6
        speedup = b_us / o_us if o_us > 0 else float("nan")
        rows.append(
            {
                "M": M,
                "gather_rows": M * EXPERTS_PER_TOKEN,
                "baseline_us": b_us,
                "optimized_us": o_us,
                "speedup": speedup,
            }
        )
        print(
            f"{M:>4d} {M * EXPERTS_PER_TOKEN:>12d} "
            f"{b_us:>14.2f} {o_us:>14.2f} {speedup:>8.3f}x"
        )
        if args.verbose:
            print()
            print(base)
            print(opt)
            print()

    print()
    print("Note: timings are full triton_kernel_moe_forward (stage-1 + stage-2).")
    print(
        "      Stage-2 is identical between the two paths, so absolute "
        "(baseline_us - optimized_us) "
    )
    print("      equals the stage-1 fast-path saving.")

    if args.save_json:
        with open(args.save_json, "w") as fp:
            json.dump(rows, fp, indent=2)
        print(f"\nWrote per-shape results to {args.save_json}")


if __name__ == "__main__":
    main()
