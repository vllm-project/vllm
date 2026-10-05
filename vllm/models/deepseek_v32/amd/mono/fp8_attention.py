# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline FP8 E4M3 block-128 quantization of the GLM-5.2 MonoKernel's attention
weights.

Kernel contract (``AttentionWeight.FP8_BLOCK128``):
  * ``w_<name>``: ``torch.float8_e4m3fn`` [rows, K], row-major (``pack_fp8`` then tiles
    it 16x64);
  * ``s_<name>``: FP32 [ceil(rows / 128), K // bk], row-major; the kernel reads
      BK == 64  (``unit_fp8``):   s[(rg * 16 // 128) * (K // 64) + kc]
      BK == 128 (``unit_fp8x2``): s[(rg * 16 // scale_rows) * (K // 128) + kc // 2]
    for 16-row group ``rg`` and 64-k chunk ``kc``; ``scale_rows`` = 128 for w_uv
    (derived from the weight / scale row counts by ``Glm5MonoKernel``);
  * ``bk`` = 64 for ``w_uk`` (K = 192), 128 for qkv_a, q_b, uv, o.
Dequantization (== FlyDSL ``reference.dequant``): ``q.float() * s[r // 128, k // bk]``.
Activations stay BF16 in the kernel (W8A16), so this is the only numeric change vs BF16
attention.

Pure torch: no HIP, no vLLM imports (CPU-testable)."""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn  # OCP E4M3 on gfx950
FP8_MAX = 448.0
SCALE_BM = 128
# (key, bk): K block size per attention matrix; rows are always blocked by 128.
ATTENTION_FP8_BLOCKS = (
    ("qkv_a", 128),
    ("q_b", 128),
    ("uk", 64),
    ("uv", 128),
    ("o", 128),
)


@torch.no_grad()
def quant_fp8_block(
    w: torch.Tensor, bm: int = SCALE_BM, bk: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """[rows, K] (bf16/fp32) -> (fp8 [rows, K], fp32 scales [ceil(rows/bm), K//bk]).

    scale = amax(|block|) / 448 (1.0 for an all-zero block); q = RNE(w / scale) clamped
    to +-448 (clamping keeps an exact-amax value from rounding past 448 into NaN)."""
    if w.dim() != 2:
        raise ValueError(f"expected a matrix, got {tuple(w.shape)}")
    rows, K = w.shape
    if K % bk:
        raise ValueError(f"K={K} not divisible by bk={bk}")
    R = -(-rows // bm) * bm
    w32 = torch.zeros(R, K, dtype=torch.float32, device=w.device)
    w32[:rows] = w.float()
    blk = w32.view(R // bm, bm, K // bk, bk)
    amax = blk.abs().amax(dim=(1, 3))
    s = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (blk / s[:, None, :, None]).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q.view(R, K)[:rows].contiguous(), s.contiguous()


def dequant_fp8_block(
    q: torch.Tensor, s: torch.Tensor, bk: int, bm: int = SCALE_BM
) -> torch.Tensor:
    rows, K = q.shape
    return q.float() * s.repeat_interleave(bm, 0)[:rows].repeat_interleave(bk, 1)


@torch.no_grad()
def quantize_attention_fp8(t: dict[str, torch.Tensor]) -> dict[str, float]:
    """In place on a ``LayerWeights.t`` dict holding BF16 ``w_{qkv_a,q_b,uk,uv,o}``:
    replaces each with its FP8 tensor and adds ``s_<name>``. Returns per-matrix rel-L2
    quantization error."""
    errs = {}
    for name, bk in ATTENTION_FP8_BLOCKS:
        w = t[f"w_{name}"]
        if w.dtype is FP8_DTYPE:
            raise ValueError(f"w_{name} is already FP8")
        q, s = quant_fp8_block(w, SCALE_BM, bk)
        if not (torch.isfinite(s).all() and (s > 0).all()):
            raise ValueError(f"w_{name}: non-finite or non-positive FP8 block scale")
        d = dequant_fp8_block(q, s, bk)
        wf = w.float()
        errs[name] = float((d - wf).norm() / wf.norm().clamp_min(1e-30))
        t[f"w_{name}"], t[f"s_{name}"] = q, s
    return errs
