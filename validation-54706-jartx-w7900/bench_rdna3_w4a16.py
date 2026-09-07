#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark + accuracy harness for PR #54706 (RDNA3 W4A16 split-K determinism).

Measures the public op path ``torch.ops._rocm_C.gptq_gemm_rdna3`` (which
forwards to the WMMA kernels for bf16 M>=16 / fp16 M>=64) on gfx1100.

For every (dtype, M, N, K) point this script records:
  * median / mean / p50 latency via CUDA events (warmup + measured iters)
  * the selected kernel path, derived by replicating the C++ dispatch
    chain (NOT inferred from M alone)
  * k_split and FP32 scratch bytes for the selected path
  * max_abs_error / mean_abs_error against an FP32 dequantized reference

Reference semantics (uint4b8, GPTQv1, synthesized zeros):
    W[k, n] = (q[k, n] - 8) * round_to_dtype(scale[g, n]),  g = k // group
    ref[m, n] = sum_k round_to_dtype(x[m, k]) * W[k, n]     (fp32 matmul)
The kernel additionally rounds the dequantized B tile to the activation
dtype (one T-rounding per weight), which bounds the achievable accuracy;
tolerances reported alongside are chosen from that analysis, not tuned to
pass.
"""

import argparse
import contextlib
import csv
import statistics
import sys

import torch

# ---------------------------------------------------------------------------
# Dispatch replication (mirrors csrc/rocm/q_gemm_rdna3{,_wmma}.cu)
# ---------------------------------------------------------------------------

BLOCK_KN_SIZE = 256  # scalar K-split granularity


def compute_wmma_k_split(size_k: int) -> int:
    if size_k >= 1024 and size_k % 64 == 0:
        return 4
    if size_k >= 512 and size_k % 32 == 0:
        return 2
    return 1


def compute_wmma_k_split_mn(m: int, n: int, k: int, m_tile: int,
                            n_tile: int) -> int:
    blocks_xy = ((n + n_tile - 1) // n_tile) * ((m + m_tile - 1) // m_tile)
    target = 1500
    if blocks_xy >= target:
        return 1
    if blocks_xy * 2 >= target and k >= 512 and k % 32 == 0:
        return 2
    if blocks_xy * 4 >= target and k >= 1024 and k % 64 == 0:
        return 4
    return compute_wmma_k_split(k)


def scalar_dispatch(m: int, k: int):
    z = (k + BLOCK_KN_SIZE - 1) // BLOCK_KN_SIZE
    if m == 1:
        variant = "scalar_mcount1"
    elif m <= 3:
        variant = "scalar_mcount2"
    elif m <= 7:
        variant = "scalar_mcount4"
    else:
        variant = "scalar_mcount8"
    scratch = z * min(64, m) * 4  # bytes per N column; x N below
    return variant, z, scratch


def wmma_dispatch(m: int, n: int, k: int, groupsize: int):
    """Replicate launch_gemm_q4_wmma_64x64_4w's fallback chain.

    Returns (kernel_name, k_split, scratch_rows) where scratch_rows is the
    M-extent of the FP32 partials tensor (TILE_M=512 for the tiled v5/v7/v8
    launchers, size_m for v1-v4).
    """
    # launch_gemm_q4_wmma_64x64_4w
    if m < 64 or n < 64:
        # -> launch_gemm_q4_wmma_64x32_4w
        if m < 64 or n < 32:
            # -> launch_gemm_q4_wmma_64x16_4w
            if m < 64:
                # -> launch_gemm_q4_wmma_32x16_2w
                if m < 32:
                    # -> launch_gemm_q4_wmma_16x16_1w
                    ks = compute_wmma_k_split(k)
                    return "wmma_16x16_1w", ks, m
                ks = compute_wmma_k_split(k)
                return "wmma_32x16_2w", ks, m
            ks = compute_wmma_k_split_mn(m, n, k, 64, 16)
            return "wmma_64x16_4w", ks, m
        ks = compute_wmma_k_split_mn(m, n, k, 64, 32)
        return "wmma_64x32_4w", ks, m
    # 64x64_4w: big-M tiled path (no act-order in this benchmark)
    if m >= 128:
        ks = compute_wmma_k_split_mn(m, n, k, 128, 64)
        if k % 32 == 0 and groupsize >= 32 and (k // ks) % 32 == 0:
            name = "wmma_128x64_k32"
        else:
            name = "wmma_128x64_k16"
        return name, ks, min(512, m)
    ks = compute_wmma_k_split_mn(m, n, k, 64, 64)
    return "wmma_64x64_4w", ks, min(512, m)


def dispatch_for(dtype: torch.dtype, m: int, n: int, k: int, groupsize: int):
    is_bf16 = dtype == torch.bfloat16
    if (is_bf16 and m >= 16) or ((not is_bf16) and m >= 64):
        name, ks, rows = wmma_dispatch(m, n, k, groupsize)
        scratch_bytes = ks * rows * n * 4 if ks > 1 else 0
        return f"wmma:{name}", ks, scratch_bytes
    variant, z, scratch_per_n = scalar_dispatch(m, k)
    scratch_bytes = z * min(64, m) * n * 4 if z > 1 else 0
    return f"scalar:{variant}", z, scratch_bytes


# ---------------------------------------------------------------------------
# Weight/op plumbing (mirrors the PR's regression test)
# ---------------------------------------------------------------------------

GROUP = 128


def build_inputs(m, k, n, seed, dtype, device):
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig)
    from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (
        RDNA3W4A16LinearKernel)
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32)
    from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                               PackedvLLMParameter)
    from vllm.scalar_type import scalar_types

    torch.manual_seed(seed)
    q_int4_kn = torch.randint(0, 16, (k, n), dtype=torch.int32)
    scales_gn = (torch.randn(k // GROUP, n) * 0.01 + 0.02).to(dtype)

    no_loader = lambda *a, **kw: None  # noqa: E731

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(data=pack_quantized_values_into_int32(
            q_int4_kn, scalar_types.uint4b8, packed_dim=0),
            weight_loader=no_loader,
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=8))
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(data=scales_gn,
                                 weight_loader=no_loader,
                                 input_dim=0,
                                 output_dim=1))
    layer.to(device)

    cfg = MPLinearLayerConfig(full_weight_shape=(k, n),
                              partition_weight_shape=(k, n),
                              weight_type=scalar_types.uint4b8,
                              act_type=dtype,
                              group_size=GROUP,
                              zero_points=False,
                              has_g_idx=False)
    kernel = RDNA3W4A16LinearKernel(cfg,
                                    w_q_param_name="qweight",
                                    w_s_param_name="scales",
                                    w_zp_param_name=None,
                                    w_gidx_param_name=None)
    kernel.process_weights_after_loading(layer)
    w_q, w_s, w_zp, w_g_idx = kernel._get_weight_params(layer)

    torch.manual_seed(seed + 1)
    x = torch.randn(m, k, device=device, dtype=dtype)

    # FP32 reference: dequantize with the *stored* (dtype-rounded) scale,
    # uint4b8 effective zero point = 8 (synthesized zp 7 + GPTQv1 offset 1).
    w_f32 = (q_int4_kn.to(device).float() - 8.0) * scales_gn.to(device).repeat_interleave(
        GROUP, dim=0).float()
    ref = x.float() @ w_f32
    return x, w_q, w_zp, w_s, w_g_idx, ref


def run_op(x, w_q, w_zp, w_s, w_g_idx):
    return torch.ops._rocm_C.gptq_gemm_rdna3(x, w_q, w_zp, w_s, w_g_idx, False)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _setup_vllm_env():
    """Config + single-process distributed init (test-suite pattern)."""
    import contextlib
    import os

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.utils.network_utils import get_open_port

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(get_open_port()))
    if not torch.distributed.is_initialized():
        init_distributed_environment(backend="cpu:gloo,cuda:hccl",
                                     world_size=1, rank=0, local_rank=0,
                                     distributed_init_method="env://")
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True,
                   help="label for this run (e.g. before / after)")
    p.add_argument("--out", default=None, help="output CSV path")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5,
                   help="repeatability check iterations")
    args = p.parse_args()

    device = "cuda"
    import vllm  # noqa: F401  (records import success in crash logs)

    with _setup_vllm_env():

        # Shape matrix. Covers required M values in both dtypes, N=4096 plus a
        # wide-N point, K in {4096, 6656} (production-like), and the JartX
        # same-M/different-N pair (M=512: N=4096 -> k_split>1, N=25600 -> 1).
        shapes = [
            # (M, N, K)
            (1, 4096, 4096),
            (8, 4096, 4096),
            (16, 4096, 4096),
            (64, 4096, 4096),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (512, 4096, 6656),
            (512, 25600, 6656),
            (1, 4096, 6656),
            (128, 4096, 6656),
        ]

        dtypes = [torch.bfloat16, torch.float16]
        rows = []
        for dtype in dtypes:
            for (m, n, k) in shapes:
                path, k_split, scratch_mb = dispatch_for(dtype, m, n, k, GROUP)
                scratch_mb = scratch_mb / (1024 * 1024)
                x, w_q, w_zp, w_s, w_g_idx, ref = build_inputs(
                    m, k, n, seed=hash((m, n, k)) % 10000, dtype=dtype,
                    device=device)

                # Accuracy (single call, against FP32 reference).
                out = run_op(x, w_q, w_zp, w_s, w_g_idx)
                err = (out.float() - ref).abs()
                max_abs = err.max().item()
                mean_abs = err.mean().item()

                # Bit-repeatability across calls.
                repeat_ok = all(
                    torch.equal(out, run_op(x, w_q, w_zp, w_s, w_g_idx))
                    for _ in range(args.repeats))

                # Latency via CUDA events.
                for _ in range(args.warmup):
                    run_op(x, w_q, w_zp, w_s, w_g_idx)
                torch.cuda.synchronize()
                times = []
                for _ in range(args.iters):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    run_op(x, w_q, w_zp, w_s, w_g_idx)
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e) * 1000.0)  # ms -> us
                med = statistics.median(times)
                mean = statistics.fmean(times)
                p50 = med

                rows.append(dict(
                    tag=args.tag, dtype="bf16" if dtype == torch.bfloat16 else "fp16",
                    M=m, N=n, K=k, kernel_path=path, k_split=k_split,
                    scratch_MB=round(scratch_mb, 2),
                    median_us=round(med, 2), mean_us=round(mean, 2),
                    p50_us=round(p50, 2), warmup=args.warmup, iters=args.iters,
                    max_abs_error=round(max_abs, 6),
                    mean_abs_error=round(mean_abs, 6),
                    bit_repeatable=repeat_ok,
                    gpu=torch.cuda.get_device_name(0),
                    rocm=torch.version.hip,
                    commit=_git_sha(),
                ))
                print(rows[-1], flush=True)
                del x, w_q, w_zp, w_s, w_g_idx, ref, out, err
                torch.cuda.empty_cache()

        out_path = args.out or f"04-benchmark-{args.tag}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_path} ({len(rows)} rows)")


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd="/workspace/vllm", text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
