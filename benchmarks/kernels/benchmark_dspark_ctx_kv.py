# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench DSpark context-KV write (proj + kv_norm + insert).

Shapes match DeepSeek-V4-Flash-0731. GEMM uses bf16 F.linear on the same
[out, in] layout as the block-FP8 weights (timing proxy). Insert ops are the
real CUDA kernels from production ``dspark._insert_context_kv``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm.models.deepseek_v4.nvidia import dspark as dspark_mod

HIDDEN = 4096
Q_LORA = 1024
HEAD_DIM = 512
N_HEADS = 64
N_LAYERS = 3
BLOCK = (128, 128)
CACHE_BLOCK = 64
HEAD_BYTES = 448 + 128 + 8  # fp8_ds_mla packed row
EPS = 1e-6
ROPE_DIM = 64


class Bf16ApplyKernel:
    """Stand-in for fp8_linear.apply_weights; each instance is a distinct object."""

    def apply_weights(self, layer, x, bias):
        weight = layer.weight
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        return F.linear(x, weight, bias)


class FakeQuantMethod:
    def __init__(self, block_size: tuple[int, int]):
        self.weight_block_size = list(block_size)
        self.fp8_linear = Bf16ApplyKernel()


class FakeProj(torch.nn.Module):
    def __init__(
        self,
        out_features: int,
        in_features: int,
        block_size: tuple[int, int],
        device: torch.device,
    ):
        super().__init__()
        self.quant_method = FakeQuantMethod(block_size)
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        br, bc = block_size
        self.weight_scale_inv = torch.nn.Parameter(
            torch.ones(
                out_features // br,
                in_features // bc,
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor):
        return self.quant_method.fp8_linear.apply_weights(self, x, None), None


class FakeRMSNorm:
    def __init__(self, dim: int, device: torch.device):
        self.weight = torch.ones(dim, dtype=torch.bfloat16, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, self.weight, EPS)
        return out


class FakeSWA:
    def __init__(self, kv_cache: torch.Tensor, block_size: int):
        self.kv_cache = kv_cache
        self.block_size = block_size


class FakeAttn:
    def __init__(
        self,
        *,
        cache_dtype: torch.dtype,
        n_tokens: int,
        device: torch.device,
        distinct_fp8_linear: bool,
        shared_kernel: Bf16ApplyKernel | None,
    ):
        self.q_lora_rank = Q_LORA
        self.head_dim = HEAD_DIM
        self.n_local_heads = N_HEADS
        self.padded_heads = 64
        self.eps = EPS
        self.fused_wqa_wkv = FakeProj(
            Q_LORA + HEAD_DIM, HIDDEN, BLOCK, device
        )
        if not distinct_fp8_linear:
            assert shared_kernel is not None
            self.fused_wqa_wkv.quant_method.fp8_linear = shared_kernel
        self.kv_norm = FakeRMSNorm(HEAD_DIM, device)
        max_pos = max(n_tokens + 16, 128)
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=_make_cos_sin_cache(max_pos, device)
        )
        num_blocks = (n_tokens + CACHE_BLOCK - 1) // CACHE_BLOCK + 2
        if cache_dtype == torch.uint8:
            cache = torch.zeros(
                num_blocks, CACHE_BLOCK * HEAD_BYTES, dtype=torch.uint8, device=device
            )
        else:
            cache = torch.zeros(
                num_blocks, CACHE_BLOCK, HEAD_DIM, dtype=cache_dtype, device=device
            )
        self.swa_cache_layer = FakeSWA(cache, CACHE_BLOCK)
        self._flashinfer_fp8_kv_scale = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )
        self._flashinfer_fp8_q_scale_inv = torch.tensor(
            [1.0], dtype=torch.float32, device=device
        )


class FakeDraft:
    def __init__(
        self,
        *,
        cache_dtype: torch.dtype,
        n_tokens: int,
        device: torch.device,
        distinct_fp8_linear: bool,
    ):
        self.hidden_size = HIDDEN
        self.num_dspark_layers = N_LAYERS
        self._fused_wkv_attempted = False
        self._fused_wkv_ready = False
        self._fused_wkv_weight = None
        self._fused_wkv_scale = None
        self._fused_wkv_layer = None
        self._wkv_kernel = None
        self._wkv_head_dim = 0
        shared = None if distinct_fp8_linear else Bf16ApplyKernel()
        self.layers: list[SimpleNamespace] = []
        for _ in range(N_LAYERS):
            attn = FakeAttn(
                cache_dtype=cache_dtype,
                n_tokens=n_tokens,
                device=device,
                distinct_fp8_linear=distinct_fp8_linear,
                shared_kernel=shared,
            )
            self.layers.append(SimpleNamespace(attn=attn))

    _build_fused_wkv_buffer = dspark_mod.DSparkDeepseekV4Model._build_fused_wkv_buffer
    precompute_and_store_context_kv = (
        dspark_mod.DSparkDeepseekV4Model.precompute_and_store_context_kv
    )


def _make_cos_sin_cache(max_pos: int, device: torch.device) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device) / ROPE_DIM
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _median_ms(samples: list[float]) -> float:
    return float(statistics.median(samples))


def _time_cuda(fn, warmup: int, iters: int) -> tuple[float, list[float]]:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))
    return _median_ms(times), times


def _cache_dtype(name: str) -> torch.dtype:
    if name == "uint8":
        return torch.uint8
    if name == "bf16":
        return torch.bfloat16
    if name == "fp8":
        return torch.float8_e4m3fn
    raise ValueError(name)


def run_one(
    *,
    n_tokens: int,
    cache_name: str,
    distinct_fp8_linear: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict:
    cache_dtype = _cache_dtype(cache_name)
    draft = FakeDraft(
        cache_dtype=cache_dtype,
        n_tokens=n_tokens,
        device=device,
        distinct_fp8_linear=distinct_fp8_linear,
    )
    main_x = torch.randn(n_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    positions = torch.arange(n_tokens, dtype=torch.int64, device=device)
    slots = [
        torch.arange(n_tokens, dtype=torch.int64, device=device)
        for _ in range(N_LAYERS)
    ]

    wkv_ok = draft._build_fused_wkv_buffer()

    def _run_full():
        draft.precompute_and_store_context_kv(main_x, positions, slots)

    # Segment: projection only (fused or 3x), then insert-only on cached kv.
    def _proj_only():
        if draft._fused_wkv_ready:
            assert draft._wkv_kernel is not None
            draft._wkv_kernel.apply_weights(draft._fused_wkv_layer, main_x, None)
        else:
            for layer in draft.layers:
                layer.attn.fused_wqa_wkv(main_x)

    kv_list = []
    if draft._fused_wkv_ready:
        fused = draft._wkv_kernel.apply_weights(
            draft._fused_wkv_layer, main_x, None
        ).view(n_tokens, N_LAYERS, HEAD_DIM)
        kv_list = [fused[:, i, :].contiguous() for i in range(N_LAYERS)]
    else:
        for layer in draft.layers:
            qr_kv, _ = layer.attn.fused_wqa_wkv(main_x)
            kv_list.append(qr_kv[..., Q_LORA:].contiguous())

    def _insert_only():
        for i, layer in enumerate(draft.layers):
            dspark_mod._insert_context_kv(layer.attn, kv_list[i], positions, slots[i])

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    alloc_before = torch.cuda.memory_allocated(device)
    _run_full()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    dummy_q_bytes = n_tokens * N_HEADS * HEAD_DIM * 2
    if cache_name == "fp8":
        dummy_q_bytes += n_tokens * N_HEADS * HEAD_DIM  # dummy_q_fp8
    dummy_q_bytes *= 1  # allocated per layer, freed between layers → one copy peak

    full_ms, _ = _time_cuda(_run_full, warmup, iters)
    proj_ms, _ = _time_cuda(_proj_only, warmup, iters)
    insert_ms, _ = _time_cuda(_insert_only, warmup, iters)

    return {
        "n_tokens": n_tokens,
        "cache": cache_name,
        "distinct_fp8_linear": distinct_fp8_linear,
        "fused_wkv_enabled": bool(wkv_ok),
        "full_ms": full_ms,
        "proj_ms": proj_ms,
        "insert_ms": insert_ms,
        "peak_alloc_delta_mib": (peak - alloc_before) / (1024**2),
        "dummy_q_nominal_mib": dummy_q_bytes / (1024**2),
        "insert_path": "production_dspark._insert_context_kv",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caches", default="uint8,bf16,fp8")
    p.add_argument("--tokens", default="128,512,2048,8192")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--shared-fp8-linear",
        action="store_true",
        help="Reuse one fp8_linear instance (is-check succeeds). Default is "
        "per-layer instances, matching checkpoints that skip fused WKV.",
    )
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--label", default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda")
    caches = [c.strip() for c in args.caches.split(",") if c.strip()]
    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    distinct = not args.shared_fp8_linear
    rows = []
    print(
        f"commit probe caches={caches} tokens={tokens} "
        f"distinct_fp8_linear={distinct} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    for cache in caches:
        for n in tokens:
            row = run_one(
                n_tokens=n,
                cache_name=cache,
                distinct_fp8_linear=distinct,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            rows.append(row)
            print(
                f"  {cache:5s} N={n:5d} fused_wkv={row['fused_wkv_enabled']} "
                f"full={row['full_ms']:.3f}ms proj={row['proj_ms']:.3f}ms "
                f"insert={row['insert_ms']:.3f}ms "
                f"peak_delta={row['peak_alloc_delta_mib']:.1f}MiB "
                f"dummy_q~{row['dummy_q_nominal_mib']:.1f}MiB",
                flush=True,
            )
    payload = {
        "label": args.label,
        "hidden": HIDDEN,
        "q_lora": Q_LORA,
        "head_dim": HEAD_DIM,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "block": list(BLOCK),
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
    }
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
