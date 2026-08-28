# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Benchmarks FP8 vs BF16 ViT attention via FlashInfer cuDNN backend.
#
# == Usage Examples ==
#
# Benchmark mode (default, FlashInfer CUDAGraph Bench)
#   python3 benchmark_vit_fp8_attn.py
#
# Profile mode (PyTorch profiler, saves TensorBoard traces):
#   python3 benchmark_vit_fp8_attn.py --profile
#   python3 benchmark_vit_fp8_attn.py --profile --profile-output-dir ./profile_traces
#
# Custom seq_lens:
#   python3 benchmark_vit_fp8_attn.py --seq-lens 4096 8192 16384

from functools import partial

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from vllm.utils.argparse_utils import FlexibleArgumentParser

# Qwen3-VL defaults
NUM_HEADS = 16
HEAD_DIM = 72
DEFAULT_SEQ_LENS = [2304, 4096, 8192, 16384]


def _setup_fp8_attention(num_heads: int, head_dim: int) -> tuple:
    """Create FP8 and BF16 attention modules + workspace."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
        _get_flashinfer_workspace_buffer,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    backend_patch = patch(
        "vllm.model_executor.layers.attention.mm_encoder_attention"
        ".get_vit_attn_backend",
        return_value=AttentionBackendEnum.FLASHINFER,
    )

    # FP8 attention
    mm_config_fp8 = MultiModalConfig(mm_encoder_attn_dtype="fp8")
    vllm_config_fp8 = VllmConfig()
    vllm_config_fp8.model_config = SimpleNamespace(multimodal_config=mm_config_fp8)
    with set_current_vllm_config(vllm_config_fp8), backend_patch:
        attn_fp8 = MMEncoderAttention(
            num_heads=num_heads,
            head_size=head_dim,
            prefix="visual.blocks.0.attn",
        ).to("cuda")

    # BF16 attention (no FP8)
    with set_current_vllm_config(VllmConfig()), backend_patch:
        attn_bf16 = MMEncoderAttention(
            num_heads=num_heads,
            head_size=head_dim,
            prefix="visual.blocks.0.attn",
        ).to("cuda")

    torch.set_default_dtype(old_dtype)

    workspace = _get_flashinfer_workspace_buffer()
    return attn_fp8, attn_bf16, workspace


def _build_meta(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    fp8: bool,
):
    """Build cu_seqlens, max_seqlen, sequence_lengths."""
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    from vllm.utils.math_utils import round_up
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    cu_np = np.array([0, seq_len], dtype=np.int32)
    fp8_padded = num_heads * round_up(head_dim, 16) if fp8 else None

    seq_lengths = MMEncoderAttention.maybe_compute_seq_lens(
        AttentionBackendEnum.FLASHINFER, cu_np, torch.device("cuda")
    )
    max_seqlen = torch.tensor(
        MMEncoderAttention.compute_max_seqlen(AttentionBackendEnum.FLASHINFER, cu_np),
        dtype=torch.int32,
    )
    cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
        AttentionBackendEnum.FLASHINFER,
        cu_np,
        num_heads * head_dim,
        1,
        torch.device("cuda"),
        fp8_padded_hidden_size=fp8_padded,
    )
    return cu_seqlens, max_seqlen, seq_lengths


def run_benchmark(
    seq_lens: list[int],
    num_heads: int,
    head_dim: int,
    method: str,
):
    """Benchmark FP8 vs BF16 attention across seq_lens.

    Uses FlashInfer GPU-level timing to measure pure kernel time,
    excluding CPU launch overhead.
    """
    if method == "cupti":
        from flashinfer.testing import bench_gpu_time_with_cupti as bench_fn

        bench_fn = partial(bench_fn, use_cuda_graph=True, cold_l2_cache=False)
    elif method == "cudagraph":
        from flashinfer.testing import (
            bench_gpu_time_with_cudagraph as bench_fn,
        )

        bench_fn = partial(bench_fn, cold_l2_cache=False)
    else:
        raise ValueError(f"Invalid method: {method}")

    attn_fp8, attn_bf16, workspace = _setup_fp8_attention(num_heads, head_dim)

    print(f"Timing method: {method}")
    print(f"{'seq_len':>8} {'BF16 (us)':>12} {'FP8 (us)':>12} {'Speedup':>10}")
    print("-" * 46)

    for seq_len in seq_lens:
        torch.manual_seed(42)

        q = torch.randn(
            seq_len,
            num_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        cu_fp8, max_s, seq_l = _build_meta(seq_len, num_heads, head_dim, fp8=True)
        # we can reuse cu_fp8 for cu_bf16 since q, k, and v are contiguous
        cu_bf16 = cu_fp8.clone()

        def bf16_fn(q=q, k=k, v=v, cu=cu_bf16, ms=max_s, sl=seq_l):
            attn_bf16._forward_flashinfer(q, k, v, cu, ms, sl)

        def fp8_fn(q=q, k=k, v=v, cu=cu_fp8, ms=max_s, sl=seq_l):
            attn_fp8._forward_flashinfer(q, k, v, cu, ms, sl)

        # bench_fn returns List[float] of per-iteration times in ms
        bf16_times = bench_fn(bf16_fn)
        fp8_times = bench_fn(fp8_fn)

        bf16_us = np.median(bf16_times) * 1e3  # ms -> us
        fp8_us = np.median(fp8_times) * 1e3
        speedup = bf16_us / fp8_us if fp8_us > 0 else float("inf")

        print(f"{seq_len:>8} {bf16_us:>12.1f} {fp8_us:>12.1f} {speedup:>9.2f}x")


def _make_trace_handler(output_dir: str, worker_name: str, label: str):
    """Create a trace handler that saves to TensorBoard and prints summary."""

    def handler(prof):
        torch.profiler.tensorboard_trace_handler(output_dir, worker_name)(prof)
        print(f"\n{'=' * 80}")
        print(label)
        print(f"{'=' * 80}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    return handler


def run_profile(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    warmup: int,
    output_dir: str,
):
    """Profile FP8 vs BF16 attention with PyTorch profiler."""
    attn_fp8, attn_bf16, workspace = _setup_fp8_attention(num_heads, head_dim)

    torch.manual_seed(42)
    q = torch.randn(
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    cu_fp8, max_s, seq_l = _build_meta(seq_len, num_heads, head_dim, fp8=True)
    # we can reuse cu_fp8 for cu_bf16 since q, k, and v are contiguous
    cu_bf16 = cu_fp8.clone()

    sched = torch.profiler.schedule(wait=0, warmup=warmup, active=1)

    # Profile BF16 (warmup handled by profiler schedule)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=_make_trace_handler(
            output_dir,
            f"bf16_h{head_dim}_s{seq_len}",
            f"BF16 Attention (seq_len={seq_len}, heads={num_heads}, "
            f"head_dim={head_dim})",
        ),
    ) as prof_bf16:
        for _ in range(warmup + 1):
            with record_function("bf16_attention"):
                attn_bf16._forward_flashinfer(
                    q.clone(), k.clone(), v.clone(), cu_bf16, max_s, seq_l
                )
                torch.accelerator.synchronize()
            prof_bf16.step()

    # Profile FP8 (warmup handled by profiler schedule)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=_make_trace_handler(
            output_dir,
            f"fp8_h{head_dim}_s{seq_len}",
            f"FP8 Attention (seq_len={seq_len}, heads={num_heads}, "
            f"head_dim={head_dim})",
        ),
    ) as prof_fp8:
        for _ in range(warmup + 1):
            with record_function("fp8_attention"):
                attn_fp8._forward_flashinfer(
                    q.clone(), k.clone(), v.clone(), cu_fp8, max_s, seq_l
                )
                torch.accelerator.synchronize()
            prof_fp8.step()

    print(f"\nTensorBoard traces saved to: {output_dir}")
    print(f"View with: tensorboard --logdir={output_dir}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark FP8 vs BF16 ViT attention.")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=NUM_HEADS,
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=HEAD_DIM,
    )
    parser.add_argument(
        "--method",
        choices=["cupti", "cudagraph"],
        default="cudagraph",
        help="GPU timing method: cupti (CUPTI kernel timing) or "
        "cudagraph (CUDA graph capture/replay). Default: cudagraph",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (profile mode only)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run PyTorch profiler instead of benchmark",
    )
    parser.add_argument(
        "--profile-seq-len",
        type=int,
        default=8192,
        help="Sequence length for profiling (default: 8192)",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default="./profile_traces",
        help="Output directory for TensorBoard traces (default: ./profile_traces)",
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(
            args.profile_seq_len,
            args.num_heads,
            args.head_dim,
            args.warmup,
            args.profile_output_dir,
        )
    else:
        run_benchmark(
            args.seq_lens,
            args.num_heads,
            args.head_dim,
            args.method,
        )
