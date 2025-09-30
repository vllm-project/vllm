# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import statistics
import time

import torch

from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    apply_rotary_pos_emb_vision_2c,
)
from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser


def benchmark_vision_rotary(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    warmup_iter: int = 10,
    benchmark_iter: int = 100,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)

    # Qwen2-VL uses rotary over half the head dim
    rotary_dim = head_size // 2
    rope = Qwen2VisionRotaryEmbedding(rotary_dim)
    rope = rope.to(dtype=torch.float32, device=torch.get_default_device())
    freqs = rope(seq_len)

    q = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype)
    k = torch.randn_like(q)

    # warmup
    for _ in range(warmup_iter):
        apply_rotary_pos_emb_vision(q, freqs)
        apply_rotary_pos_emb_vision(k, freqs)
        apply_rotary_pos_emb_vision_2c(q, k, freqs)
    torch.cuda.synchronize()

    def time_op_cuda_events(fn) -> float:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        return start_event.elapsed_time(end_event)  # ms

    def time_op_cpu_timer(fn) -> float:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - start) * 1000.0  # ms

    timer = time_op_cuda_events if torch.cuda.is_available() else time_op_cpu_timer

    # 1c path timing: apply to q and k separately
    lat_1c: list[float] = []
    for _ in range(benchmark_iter):
        lat_1c.append(
            timer(
                lambda: (
                    apply_rotary_pos_emb_vision(q, freqs),
                    apply_rotary_pos_emb_vision(k, freqs),
                )
            )
        )

    # 2c path timing: apply to q and k together
    lat_2c: list[float] = []
    for _ in range(benchmark_iter):
        lat_2c.append(timer(lambda: apply_rotary_pos_emb_vision_2c(q, k, freqs)))

    mean_1c = statistics.mean(lat_1c)
    mean_2c = statistics.mean(lat_2c)
    med_1c = statistics.median(lat_1c)
    med_2c = statistics.median(lat_2c)

    print("== Vision Rotary Benchmark (1c vs 2c) ==")
    print(
        f"Config: batch={batch_size}, seqlen={seq_len}, "
        f"heads={num_heads}, head_dim={head_size}, dtype={dtype}"
    )
    print(f"Iters: warmup={warmup_iter}, bench={benchmark_iter}")
    print(f"1c (separated q and k): mean={mean_1c:.4f} ms, median={med_1c:.4f} ms")
    print(f"2c (fused q and k):  mean={mean_2c:.4f} ms, median={med_2c:.4f} ms")
    print(f"Fusion speedup: {mean_1c / mean_2c:.3f}x")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the 1c vs 2c vision rotary embedding paths."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument(
        "--head-size",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iter", type=int, default=10)
    parser.add_argument("--benchmark-iter", type=int, default=1000)
    args = parser.parse_args()

    benchmark_vision_rotary(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_size=args.head_size,
        dtype=getattr(torch, args.dtype),
        seed=args.seed,
        device=args.device,
        warmup_iter=args.warmup_iter,
        benchmark_iter=args.benchmark_iter,
    )
