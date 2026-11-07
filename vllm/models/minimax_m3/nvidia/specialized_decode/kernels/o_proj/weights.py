# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-hoist weights of the TP4 o_proj router CTAs.

The MoE router runs on the post-attention normed row, `logits = W_r n` with
`n = h * rstd * (1 + g)` and `h = residual + sum_k o_k` (o_k = rank k's
o_proj partial `W_o,k a_k`). `rstd` is a per-row scalar, so

    logits = rstd * (W' h),   W' = W_r diag(1 + g)
    W' h   = sum_k [ W'' _k a_k  +  W'[:, H_k] residual[H_k] ],   W''_k = W' deq(W_o,k)

Each rank's router CTAs compute their bracket (`a` from the decode's MXFP8
output, `r` from this rank's quarter of the residual columns) while the
GEMV runs, push it to every rank's rx slot, and the post multiplies the rank
sum by rstd. `W''` is folded in fp64 and rounded once to fp32.

Blob layouts (per rank, fp32), as the kernel's router CTAs index them:
  blob_a `[r][e][m][tid][4]` = W''_k[8 r + e, 8 tid + 4 m + j]          (2 MB)
  blob_r `[r][e][m][tid][4]` = W'[8 r + e, c0 + 1024 m + 4 tid + j]      (2 MB,
         columns >= 1536 zero-padded; c0 = rank * 1536)
"""

from __future__ import annotations

import torch

H = 6144
E = 128
TOP_K = 4
NR = 16  # router CTAs
RE = 8  # experts per router CTA
KO = 2048  # o_proj input width per rank (16 heads x 128)
HK = 1536  # residual columns per rank
HKP = 2048  # padded r-part columns
THR = 256
ROUTE_SCALE = 2.0
SUM_EPS = 1e-20
EPS = 1e-6


# ----------------------------------------------------------------------------- MXFP8
# scale layout (128x4 swizzle)
def sf128x4_index(n: torch.Tensor, kb: torch.Tensor, kb_total: int) -> torch.Tensor:
    kt = (kb_total + 3) // 4
    return (
        (n // 128) * kt * 512
        + (kb // 4) * 512
        + (n % 32) * 16
        + ((n % 128) // 32) * 4
        + kb % 4
    )


def swizzle_128x4(sf: torch.Tensor) -> torch.Tensor:
    """[N, KB] uint8 -> flat swizzled uint8 (rows padded to 128, KB to a multiple of
    4)."""
    n, kb = sf.shape
    npad, kbpad = (n + 127) // 128 * 128, (kb + 3) // 4 * 4
    out = torch.zeros(npad * kbpad, dtype=torch.uint8, device=sf.device)
    nn, kk = torch.meshgrid(
        torch.arange(n, device=sf.device),
        torch.arange(kb, device=sf.device),
        indexing="ij",
    )
    out[sf128x4_index(nn, kk, kb).reshape(-1)] = sf.reshape(-1)
    return out


def unswizzle_128x4(swz: torch.Tensor, n: int, kb: int) -> torch.Tensor:
    flat = swz.reshape(-1).view(torch.uint8)
    nn, kk = torch.meshgrid(
        torch.arange(n, device=flat.device),
        torch.arange(kb, device=flat.device),
        indexing="ij",
    )
    return flat[sf128x4_index(nn, kk, kb)].reshape(n, kb)


def e8m0_scale(codes_u8: torch.Tensor) -> torch.Tensor:
    return torch.exp2(codes_u8.to(torch.float64) - 127.0)


def mxfp8_dequant(w_e4m3: torch.Tensor, sf_u8: torch.Tensor) -> torch.Tensor:
    """[N, K] e4m3 x UE8M0 per 32 columns -> fp64 (exact)."""
    n, k = w_e4m3.shape
    return w_e4m3.to(torch.float64) * e8m0_scale(sf_u8[:, : k // 32]).repeat_interleave(
        32, dim=1
    )


def mxfp8_quant_rows(x: torch.Tensor):
    """Row-wise MXFP8 (code = e8m0 roundup of amax/448, x / 2^code -> e4m3 RN sat), for
    tests."""
    t, k = x.shape
    xb = x.float().reshape(t, k // 32, 32)
    amax = xb.abs().amax(-1)
    e = torch.ceil(torch.log2(torch.clamp(amax / 448.0, min=2.0**-127)))
    code = torch.clamp(e + 127, 0, 254).to(torch.uint8)
    sc = torch.where(
        amax > 0, torch.exp2(-(code.float() - 127.0)), torch.zeros_like(amax)
    )
    q = (xb * sc.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(t, k)
    return q, code


# ----------------------------------------------------------------------------- folding
def fold_router(router: torch.Tensor, post_norm: torch.Tensor) -> torch.Tensor:
    """W' = W_r diag(1 + g): (1 + g) in fp32, then one fp32 product."""
    return router.float() * (1.0 + post_norm.float()).unsqueeze(0)


def fold_oproj(wp: torch.Tensor, o_deq: torch.Tensor) -> torch.Tensor:
    """W'' = W' deq(W_o_k), fp64 accumulate, one rounding to fp32. o_deq: [H, Ko]."""
    return (wp.double() @ o_deq.double()).float()


def r_cols(rank: int, world: int) -> tuple[int, int]:
    w = H // world
    return rank * w, (rank + 1) * w


def blobs(router, post_norm, o_w_e4m3, o_sf_u8, rank: int, world: int = 4):
    """(blob_a, blob_r, W', W''_k). o_sf_u8: the unswizzled UE8M0 [H, KO / 32] of this
    rank's o_proj slice."""
    wp = fold_router(router.cpu(), post_norm.cpu())
    w2 = fold_oproj(wp, mxfp8_dequant(o_w_e4m3.cpu(), o_sf_u8.cpu()))
    assert wp.shape == (E, H) and w2.shape == (E, KO), (wp.shape, w2.shape)
    a = w2.reshape(NR, RE, THR, KO // 1024, 4).permute(0, 1, 3, 2, 4).contiguous()
    c0, c1 = r_cols(rank, world)
    assert c1 - c0 == HK, (c0, c1)
    rp = torch.zeros(E, HKP, dtype=wp.dtype)
    rp[:, :HK] = wp[:, c0:c1]
    r = rp.reshape(NR, RE, HKP // 1024, THR, 4).contiguous()
    return a, r, wp, w2


def emulate(blob_a, blob_r, xq_f, sc, res, rank: int, t: int):
    """CPU model of one rank's router CTAs with the kernel's flat indexing (fp64: layout
    check, not rounding).
    xq_f `[t][KO]` e4m3 values, sc `[t][KO / 32]` scale factors, res `[t][H]`.
    Returns (z_a, z_r) `[t][E]`."""
    ba, br = blob_a.reshape(-1).double(), blob_r.reshape(-1).double()
    za = torch.zeros(t, E, dtype=torch.float64)
    zr = torch.zeros(t, E, dtype=torch.float64)
    tid = torch.arange(THR)
    for r in range(NR):
        for e in range(RE):
            for m in range(HKP // 1024):
                w = br[
                    (((r * RE + e) * (HKP // 1024) + m) * THR * 4)
                    + (tid[:, None] * 4 + torch.arange(4))
                ]
                off = m * 1024 + 4 * tid[:, None] + torch.arange(4)
                off = torch.where(off < HK, off, 4 * tid[:, None] + torch.arange(4))
                col = rank * HK + off
                for tt in range(t):
                    zr[tt, r * RE + e] += (w * res[tt].double()[col]).sum()
            for m in range(KO // 1024):
                w = ba[
                    (((r * RE + e) * (KO // 1024) + m) * THR * 4)
                    + (tid[:, None] * 4 + torch.arange(4))
                ]
                col = 8 * tid[:, None] + 4 * m + torch.arange(4)
                for tt in range(t):
                    x = xq_f[tt].double()[col] * sc[tt].double()[tid // 4][:, None]
                    za[tt, r * RE + e] += (w * x).sum()
    return za, zr


# --------------------------------------------------------------------------------------
# references
def route_minimax2(logits: torch.Tensor, bias: torch.Tensor):
    """The kernels' MiniMax2 routing, op for op: key = (0.5 tanh(0.5 x) + 0.5) + bias;
    top-4 by key, ties to the
    lower id; sk = key - bias; w = bf16(2 sk / ((sk0 + sk2) + (sk1 + sk3) + 1e-20)).
    Returns ids [T, 4] int64 (rank order), w [T, 4] fp32 (bf16 values), keys [T, E]."""
    x = logits.float()
    sig = (0.5 * torch.tanh(0.5 * x) + 0.5).float()
    keys = sig + bias.float().unsqueeze(0)
    ids = torch.sort(keys, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    sk = keys.gather(1, ids) - bias.float()[ids]
    total = (sk[:, 0] + sk[:, 2]) + (sk[:, 1] + sk[:, 3])
    w = ((sk * ROUTE_SCALE) / (total + SUM_EPS).unsqueeze(1)).to(torch.bfloat16).float()
    return ids, w, keys


def hoisted_logits_ref(parts, rstd):
    """parts: per source rank [T, E] fp32 partials (a + r); rstd [T]. Rank-order fp32
    sum times rstd."""
    s = None
    for p in parts:
        s = p.float() if s is None else s + p.float()
    return s * rstd.float().unsqueeze(1)


__all__ = [
    "E",
    "H",
    "HK",
    "HKP",
    "KO",
    "NR",
    "blobs",
    "emulate",
    "fold_oproj",
    "fold_router",
    "hoisted_logits_ref",
    "mxfp8_dequant",
    "mxfp8_quant_rows",
    "r_cols",
    "route_minimax2",
    "swizzle_128x4",
    "unswizzle_128x4",
]
