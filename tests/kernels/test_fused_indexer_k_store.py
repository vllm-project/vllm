# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accuracy + microbench for fused_indexer_k_store (GLM-5.2 DSA G001).

Compares the pid-0 store fuse (LayerNorm + RoPE + UE8M0 fp8 cache write)
against the unfused PyTorch reference used by deepseek_v2.Indexer.
"""

import time

import pytest
import torch

from vllm.models.deepseek_v32.common import kernels as K
from vllm.platforms import current_platform

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
EPS = 1e-6
ROPE_DIM = 64
INDEX_HEAD_DIM = 128


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


def rope(
    x: torch.Tensor, pos: torch.Tensor, cos_sin: torch.Tensor, interleave: bool
) -> torch.Tensor:
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


def ue8m0_quant(vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = vals.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (vals.float() / scale).to(FP8)
    return q, scale.squeeze(-1)


def _fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    def key(t):
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return int((key(a) - key(b)).abs().max().item())


def assert_fp8(got: torch.Tensor, ref: torch.Tensor, msg: str):
    assert _fp8_ulp(got, ref) <= 1, f"{msg}: >1 fp8 ULP"

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="fused_indexer_k_store requires CUDA with fp8 (SM89+)",
)


def _pack_from_cache(idx_cache: torch.Tensor, block_size: int, num_tokens: int):
    flat = idx_cache[0].reshape(-1)
    vals = flat[: block_size * INDEX_HEAD_DIM].view(FP8).reshape(block_size, INDEX_HEAD_DIM)
    scales = flat[block_size * INDEX_HEAD_DIM :].view(torch.float32)
    return vals[:num_tokens], scales[:num_tokens]


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 128, 512])
@pytest.mark.parametrize("index_interleave", [True, False])
def test_fused_indexer_k_store_matches_unfused(num_tokens: int, index_interleave: bool):
    torch.manual_seed(0)
    dev = "cuda"
    max_pos = 8192
    pos = torch.arange(num_tokens, device=dev, dtype=torch.int64) % max_pos
    ik = torch.randn(num_tokens, INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    ikw = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    ikb = torch.randn(INDEX_HEAD_DIM, device=dev, dtype=torch.float32)
    idx_cos_sin = make_cos_sin(max_pos, ROPE_DIM, dev)

    bs = max(num_tokens, 64)
    idx_row = INDEX_HEAD_DIM + INDEX_HEAD_DIM // 128 * 4  # 132
    idx_cache = torch.zeros(1, bs, idx_row, device=dev, dtype=torch.uint8)
    slot = torch.arange(num_tokens, device=dev, dtype=torch.int64)

    K.fused_indexer_k_store(
        pos,
        ik,
        ikw,
        ikb,
        EPS,
        idx_cos_sin,
        slot,
        idx_cache,
        index_rope_interleave=index_interleave,
    )

    ik_ref = layer_norm(ik, ikw, ikb)
    ik_ref = rope(ik_ref, pos, idx_cos_sin, interleave=index_interleave)
    q_ref, s_ref = ue8m0_quant(ik_ref)
    vals, scales = _pack_from_cache(idx_cache, bs, num_tokens)
    assert_fp8(vals, q_ref, "indexer-K fp8")
    torch.testing.assert_close(scales, s_ref, rtol=0, atol=0)


def test_fused_indexer_k_store_padding_slots_are_skipped():
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

    K.fused_indexer_k_store(
        pos, ik, ikw, ikb, EPS, idx_cos_sin, slot, idx_cache, True
    )
    untouched = idx_cache[0, n - 2 : n]
    assert (untouched == 7).all(), "padding slots must not be written"


@pytest.mark.parametrize("num_tokens", [128, 1024, 4096])
def test_fused_indexer_k_store_microbench(num_tokens: int):
    """Warm kernel vs unfused LN+RoPE+quant. Writes no serving claim."""
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
        ik_ref = layer_norm(ik, ikw, ikb)
        ik_ref = rope(ik_ref, pos, idx_cos_sin, interleave=True)
        return ue8m0_quant(ik_ref)

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

    print(
        f"\nG001 microbench tokens={num_tokens}: "
        f"fused={fused_ms:.3f} ms  unfused_ref={unfused_ms:.3f} ms  "
        f"speedup={unfused_ms / fused_ms:.2f}x"
    )
    assert fused_ms > 0 and unfused_ms > 0
