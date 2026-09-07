#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone G001 accuracy + microbench runner.

Loads ``fused_indexer_k_store`` from the vLLM 0.27.1 checkout without importing
the full vLLM package (this host's default torch is CUDA 13.0 while the driver
is 12.8). Use a cu128 torch interpreter, for example:

  /mnt/nvme1n1/ml_research/models/envs/minicpm_stream_torch210/bin/python \\
      tests/kernels/run_fused_indexer_k_store.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "vllm/models/deepseek_v32/common/kernels.py"

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
EPS = 1e-6
ROPE_DIM = 64
INDEX_HEAD_DIM = 128


def _load_kernels():
    vllm = types.ModuleType("vllm")
    tu = types.ModuleType("vllm.triton_utils")
    tu.tl = tl
    tu.triton = triton
    sys.modules["vllm"] = vllm
    sys.modules["vllm.triton_utils"] = tu
    spec = importlib.util.spec_from_file_location("vllm_kernels_g001", KERNELS)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def make_cos_sin(max_pos: int, rot_dim: int, device) -> torch.Tensor:
    half = rot_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    return (xf - mean) * torch.rsqrt(var + EPS) * w.float() + b.float()


def rope(x, pos, cos_sin, interleave: bool) -> torch.Tensor:
    rot = cos_sin.shape[-1]
    half = rot // 2
    cs = cos_sin[pos.long()]
    cos, sin = cs[..., :half], cs[..., half:]
    out = x.float().clone()
    r = out[..., :rot]
    if interleave:
        x1, x2 = r[..., 0::2].clone(), r[..., 1::2].clone()
        r[..., 0::2] = x1 * cos - x2 * sin
        r[..., 1::2] = x2 * cos + x1 * sin
    else:
        x1, x2 = r[..., :half].clone(), r[..., half:].clone()
        r[..., :half] = x1 * cos - x2 * sin
        r[..., half:] = x2 * cos + x1 * sin
    return out


def ue8m0_quant(vals: torch.Tensor):
    amax = vals.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (vals.float() / scale).to(FP8)
    return q, scale.squeeze(-1)


def fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return int((key(a) - key(b)).abs().max().item())


def pack_from_cache(idx_cache, block_size, num_tokens):
    flat = idx_cache[0].reshape(-1)
    vals = (
        flat[: block_size * INDEX_HEAD_DIM]
        .view(FP8)
        .reshape(block_size, INDEX_HEAD_DIM)
    )
    scales = flat[block_size * INDEX_HEAD_DIM :].view(torch.float32)
    return vals[:num_tokens], scales[:num_tokens]


def run_accuracy(K):
    rows = []
    for num_tokens in (1, 4, 17, 128, 512):
        for interleave in (True, False):
            torch.manual_seed(0)
            dev = "cuda"
            max_pos = 8192
            pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos
            ik = torch.randn(num_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
            ikw = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
            ikb = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
            idx_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)
            bs = max(num_tokens, 64)
            idx_row = INDEX_HEAD_DIM + INDEX_HEAD_DIM // 128 * 4
            idx_cache = torch.zeros(1, bs, idx_row, device=dev, dtype=torch.uint8)
            slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)
            K.fused_indexer_k_store(
                pos, ik, ikw, ikb, EPS, idx_cos_sin, slot, idx_cache, interleave
            )
            ik_ref = rope(layer_norm(ik, ikw, ikb), pos, idx_cos_sin, interleave)
            q_ref, s_ref = ue8m0_quant(ik_ref)
            vals, scales = pack_from_cache(idx_cache, bs, num_tokens)
            ulp = fp8_ulp(vals, q_ref)
            scale_ok = torch.allclose(scales, s_ref, rtol=0, atol=0)
            ok = ulp <= 1 and scale_ok
            rows.append(
                {
                    "tokens": num_tokens,
                    "interleave": interleave,
                    "fp8_ulp": ulp,
                    "scale_exact": bool(scale_ok),
                    "pass": ok,
                }
            )
            if not ok:
                raise SystemExit(
                    f"FAIL tokens={num_tokens} interleave={interleave} ulp={ulp} scale={scale_ok}"
                )
    return rows


def run_padding(K):
    torch.manual_seed(1)
    dev = "cuda"
    n = 8
    pos = torch.arange(n, device=dev, dtype=torch.int64)
    ik = torch.randn(n, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    ikw = torch.ones(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    ikb = torch.zeros(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    idx_cos_sin = make_cos_sin(1024, ROPE_DIM, dev)
    bs = 16
    idx_row = INDEX_HEAD_DIM + INDEX_HEAD_DIM // 128 * 4
    idx_cache = torch.full((1, bs, idx_row), 7, device=dev, dtype=torch.uint8)
    slot = torch.arange(n, device=dev, dtype=torch.int64)
    slot[-2:] = -1
    K.fused_indexer_k_store(pos, ik, ikw, ikb, EPS, idx_cos_sin, slot, idx_cache, True)
    if not (idx_cache[0, n - 2 : n] == 7).all():
        raise SystemExit("FAIL padding slots were written")


def run_microbench(K):
    rows = []
    for num_tokens in (128, 1024, 4096):
        torch.manual_seed(2)
        dev = "cuda"
        max_pos = 8192
        pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos
        ik = torch.randn(num_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
        ikw = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
        ikb = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
        idx_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)
        bs = num_tokens
        idx_row = INDEX_HEAD_DIM + INDEX_HEAD_DIM // 128 * 4
        idx_cache = torch.zeros(1, bs, idx_row, device=dev, dtype=torch.uint8)
        slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)

        def unfused():
            return ue8m0_quant(
                rope(layer_norm(ik, ikw, ikb), pos, idx_cos_sin, True)
            )

        for _ in range(5):
            K.fused_indexer_k_store(
                pos, ik, ikw, ikb, EPS, idx_cos_sin, slot, idx_cache, True
            )
            unfused()
        torch.cuda.synchronize()
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            K.fused_indexer_k_store(
                pos, ik, ikw, ikb, EPS, idx_cos_sin, slot, idx_cache, True
            )
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) * 1e3 / iters
        t0 = time.perf_counter()
        for _ in range(iters):
            unfused()
        torch.cuda.synchronize()
        unfused_ms = (time.perf_counter() - t0) * 1e3 / iters
        rows.append(
            {
                "tokens": num_tokens,
                "fused_ms": round(fused_ms, 4),
                "unfused_ref_ms": round(unfused_ms, 4),
                "speedup": round(unfused_ms / fused_ms, 3),
            }
        )
        print(
            f"microbench tokens={num_tokens}: fused={fused_ms:.3f}ms "
            f"unfused_ref={unfused_ms:.3f}ms speedup={unfused_ms / fused_ms:.2f}x"
        )
    return rows


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this interpreter")
    print("device", torch.cuda.get_device_name(0), "torch", torch.__version__)
    K = _load_kernels()
    acc = run_accuracy(K)
    run_padding(K)
    bench = run_microbench(K)
    print("accuracy_pass", len(acc), "cases")
    out = Path("/tmp/g001_fused_indexer_k_store_results.json")
    import json

    out.write_text(json.dumps({"accuracy": acc, "microbench": bench}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
