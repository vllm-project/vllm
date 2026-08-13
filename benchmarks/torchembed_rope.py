"""
Benchmark: vLLM C++ rotary embedding vs. torchembed Triton kernel.

Run with:
    VLLM_USE_TORCHEMBED=1 python benchmarks/torchembed_rope.py

Requires torchembed and triton to be installed:
    pip install torchembed[triton]

Config mirrors Llama 3.1 8B GQA: 32 Q heads, 8 KV heads, head_dim=128.

Performance note
----------------
In vLLM's standard packed-token inference layout ``(num_tokens, heads*dim)``,
the built-in CUDA kernel operates in-place with no format conversion, so it
is typically 3–8× faster than the torchembed path on current hardware.
torchembed's fused Triton kernel excels in training pipelines or when tensors
are already in ``(batch, heads, seq, dim)`` layout.
"""
import math
import sys
import time

import torch

from vllm import _custom_ops as ops
from vllm.utils.torchembed import is_torchembed_available

HEAD_SIZE = 128
ROTARY_DIM = 128
MAX_POS = 32768
BASE = 10000.0
DTYPE = torch.float16
NUM_QH, NUM_KVH = 32, 8
IS_NEOX = True
WARMUP = 30
RUNS = 200


def build_cos_sin_cache(
    max_pos: int, rotary_dim: int, base: float, dtype: torch.dtype, device: str
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype).to(device)


def bench(fn, *args, warmup: int = WARMUP, runs: int = RUNS) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def run_vllm_cpp(positions, q, k, cos_sin_cache):
    ops.rotary_embedding(positions, q, k, HEAD_SIZE, cos_sin_cache, IS_NEOX)
    return q, k


def run_torchembed(positions, q_in, k_in, cos_sin_cache):
    from torchembed._triton import fused_rope_forward

    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cs = cos_sin_cache.index_select(0, positions)
    c, s = cs.chunk(2, dim=-1)

    q = q_in.view(num_tokens, NUM_QH, ROTARY_DIM)
    k = k_in.view(num_tokens, NUM_KVH, ROTARY_DIM)

    q_t = q.transpose(0, 1).contiguous()
    k_t = k.transpose(0, 1).contiguous()
    q_out_t, k_out_t = fused_rope_forward(q_t, k_t, c, s)
    q_out = q_out_t.transpose(0, 1).contiguous().reshape(q_in.shape)
    k_out = k_out_t.transpose(0, 1).contiguous().reshape(k_in.shape)
    return q_out, k_out


def main():
    device = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"GQA: {NUM_QH}Q / {NUM_KVH}KV heads / head_dim={HEAD_SIZE} / float16")
    print()

    if not is_torchembed_available():
        print(
            "torchembed not enabled.  Set VLLM_USE_TORCHEMBED=1 and install "
            "torchembed[triton] to run the full comparison."
        )
        sys.exit(1)

    cos_sin_cache = build_cos_sin_cache(MAX_POS, ROTARY_DIM, BASE, DTYPE, device)

    # Correctness check
    S = 512
    pos = torch.randint(0, MAX_POS, (S,), device=device)
    q0 = torch.randn(S, NUM_QH * HEAD_SIZE, dtype=DTYPE, device=device)
    k0 = torch.randn(S, NUM_KVH * HEAD_SIZE, dtype=DTYPE, device=device)
    q_cpp, k_cpp = run_vllm_cpp(pos, q0.clone(), k0.clone(), cos_sin_cache)
    q_tc, k_tc = run_torchembed(pos, q0.clone(), k0.clone(), cos_sin_cache)
    qdiff = (q_cpp - q_tc).abs().max().item()
    kdiff = (k_cpp - k_tc).abs().max().item()
    status = "PASS" if qdiff < 0.05 and kdiff < 0.05 else "MISMATCH"
    print(f"Correctness (S=512): max|q diff|={qdiff:.4f}  max|k diff|={kdiff:.4f}  [{status}]")
    print()

    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    print(f"{'S':<7} | {'vLLM C++ (ms)':>14} | {'Triton (ms)':>12} | {'speedup':>9}")
    print("-" * 52)
    for S in seq_lens:
        positions = torch.randint(0, MAX_POS, (S,), device=device)
        q = torch.randn(S, NUM_QH * HEAD_SIZE, dtype=DTYPE, device=device)
        k = torch.randn(S, NUM_KVH * HEAD_SIZE, dtype=DTYPE, device=device)

        t_cpp = bench(run_vllm_cpp, positions, q.clone(), k.clone(), cos_sin_cache)
        t_tc = bench(run_torchembed, positions, q.clone(), k.clone(), cos_sin_cache)
        ratio = t_cpp / t_tc
        note = f"{'Triton faster' if ratio > 1 else 'C++ faster':>14}"
        print(f"{S:<7} | {t_cpp:>13.3f} | {t_tc:>11.3f} | {ratio:>8.2f}×  {note}")

    print()
    print(
        "Tip: the vLLM C++ kernel wins in inference because it operates in-place\n"
        "on the packed-token layout.  torchembed wins in training or (B,H,S,D) contexts."
    )


if __name__ == "__main__":
    main()
