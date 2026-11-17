# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MXFP8 shared expert of the glue paths, on AITER's preshuffled-weight GEMM.

Weights are derived once, after the checkpoint is loaded (raw [N, K] FP8 rows and
per-row [N, K / 32] E8M0 scales, as the Quark OCP-MX loader stores them):

* decode: this rank's TP shard, (16, 16)-preshuffled (gate_up [2 I/TP, H],
  down [H, I/TP]); the output is a TP partial, like the stock MLP's.
* SP prefill: the full, replicated expert, all-gathered from the TP shards
  (gate_up = [all ranks' gate rows | all ranks' up rows], down = all ranks'
  columns), preshuffled; the output of a rank's own rows is complete.
* decode, fused (`fused`): the TP shard for AITER's a16w4 MoE kernels' dense
  shared-expert prologue (routed + shared in the two MoE launches; no separate
  shared GEMMs). "mxfp8": the checkpoint's FP8 codes, (16, 16)-preshuffled, and
  their E8M0 scales in AITER's `shuffle_scale` layout; the kernels scale the codes
  to bf16 in registers. "bf16" (an AITER that takes only bf16 there): the exact
  bf16 upcast (e4m3 x 2^k fits bf16), (16, 16)-preshuffled, twice the bytes. Both
  give the same bits."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass

import torch

OUT_DTYPE = torch.bfloat16
SWIGLU_BLOCK_M = 16  # rows a program of the fused swiglu + MXFP8 quant
# decode shared expert policy (measured, TP4 MI355X): AITER's down GEMM wins at
# these widths, vLLM's Triton dot_scaled one at the others
AITER_DOWN_MAX_ROWS = 72
AITER_DOWN_EXTRA_ROWS = (160,)


@dataclass
class SharedWeights:
    gu: torch.Tensor | None = None  # TP gate_up, preshuffled FP8
    gu_s: torch.Tensor | None = None  # its per-row E8M0 (uint8)
    dn: torch.Tensor | None = None  # TP down, preshuffled FP8
    dn_s: torch.Tensor | None = None
    gu_full: torch.Tensor | None = None  # replicated gate_up / down (SP)
    gu_full_s: torch.Tensor | None = None
    dn_full: torch.Tensor | None = None
    dn_full_s: torch.Tensor | None = None
    # decode, fused: AITER's stage-1 / stage-2 shared-weight arguments (no scratch)
    fused1: dict[str, torch.Tensor] | None = None
    fused2: dict[str, torch.Tensor] | None = None


def fp8_rows(linear) -> tuple[torch.Tensor, torch.Tensor]:
    """`linear`'s FP8 weight [N, K] and per-row E8M0 scales [N, K / 32] (uint8)."""
    w = linear.weight.data
    n, k = w.shape
    assert w.element_size() == 1, f"FP8 weight expected, got {w.dtype}"
    s = linear.weight_scale.data.view(torch.uint8)
    if s.shape[0] != n:  # one scale row per 32 weight rows
        s = s.repeat_interleave(32, dim=0)
    return w.contiguous(), s[:n, : k // 32].contiguous()


def swiglu_params(mlp) -> tuple[float, float, float | None]:
    beta = 1.0 if mlp.swiglu_beta is None else float(mlp.swiglu_beta)
    return float(mlp.swiglu_alpha), beta, mlp.swiglu_limit


def dequant_bf16(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 [N, K] x per-32 E8M0 [N, K / 32] -> bf16 [N, K], exact (a 3-bit-mantissa
    value times a power of two is a bf16 value)."""
    scale = torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=1)
    return (w.float() * scale).to(torch.bfloat16)


# the a16w4 kernels' SwiGLU: clamp(gate) * sigmoid(1.702 clamp(gate)) * (clamp(up) + 1)
FUSED_SWIGLU = (1.702, 1.0)


def fused_shared_supported(mlp) -> str | None:
    """The shared-weight format ("mxfp8" or "bf16") in which `mlp` runs inside
    the a16w4 MoE kernels: their SwiGLU is `mlp`'s and the installed AITER takes
    a dense shared expert there; else None."""
    alpha, beta, limit = swiglu_params(mlp)
    if (alpha, beta) != FUSED_SWIGLU or limit is None:
        return None
    return aiter_fused_shared_format()


@functools.cache
def aiter_fused_shared_format() -> str | None:
    """The dense shared expert AITER's A16W4 MoE stages take: "mxfp8" (FP8 codes
    + E8M0 scales: `shared_w1` / `shared_w1_scale`, `shared_w2` / `shared_w2_scale`),
    "bf16" (`shared_w1_bf16` / `shared_w2_bf16`, ROCm/aiter#7773 as first
    posted), or None."""
    try:
        from aiter import fused_moe as fm

        p1 = inspect.signature(fm._flydsl_stage1_wrapper).parameters
        p2 = inspect.signature(fm._flydsl_stage2_wrapper).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return None
    if {"shared_w1", "shared_w1_scale"} <= p1.keys() and {
        "shared_w2",
        "shared_w2_scale",
    } <= p2.keys():
        return "mxfp8"
    if "shared_w1_bf16" in p1 and "shared_w2_bf16" in p2:
        return "bf16"
    return None


def _gather(t: torch.Tensor, dim: int, group) -> torch.Tensor:
    """All-gather of a byte tensor over the TP group along `dim`."""
    out = group.all_gather(t.contiguous().view(torch.uint8), dim)
    return out.contiguous().view(t.dtype)


def build_shared_weights(
    mlp, decode: bool, sp: bool, group, fused: str | None = None
) -> SharedWeights:
    """Collective when `sp` (every TP rank, same layer order). `fused` ("mxfp8" or
    "bf16", from `fused_shared_supported`): decode runs the shared expert inside
    the MoE kernels, on weights in that format (and not the separate GEMMs')."""
    from aiter.ops.shuffle import shuffle_weight

    gu, gu_s = fp8_rows(mlp.gate_up_proj)
    dn, dn_s = fp8_rows(mlp.down_proj)
    out = SharedWeights()
    if decode and fused == "mxfp8":
        from aiter.ops.shuffle import shuffle_scale

        e4m3 = torch.float8_e4m3fn  # OCP MXFP8 codes (the checkpoint's F8_E4M3)
        out.fused1 = {
            "shared_w1": shuffle_weight(gu.view(e4m3), layout=(16, 16)),
            "shared_w1_scale": shuffle_scale(gu_s),
        }
        out.fused2 = {
            "shared_w2": shuffle_weight(dn.view(e4m3), layout=(16, 16)),
            "shared_w2_scale": shuffle_scale(dn_s),
        }
    elif decode and fused == "bf16":
        out.fused1 = {
            "shared_w1_bf16": shuffle_weight(dequant_bf16(gu, gu_s), layout=(16, 16))
        }
        out.fused2 = {
            "shared_w2_bf16": shuffle_weight(dequant_bf16(dn, dn_s), layout=(16, 16))
        }
    elif decode:
        out.gu, out.gu_s = shuffle_weight(gu, layout=(16, 16)), gu_s
        out.dn, out.dn_s = shuffle_weight(dn, layout=(16, 16)), dn_s
    if sp:
        inter = gu.shape[0] // 2  # this rank's [gate | up] split
        gate, up = _gather(gu[:inter], 0, group), _gather(gu[inter:], 0, group)
        gate_s, up_s = _gather(gu_s[:inter], 0, group), _gather(gu_s[inter:], 0, group)
        out.gu_full = shuffle_weight(torch.cat([gate, up]), layout=(16, 16))
        out.gu_full_s = torch.cat([gate_s, up_s]).contiguous()
        del gate, up
        out.dn_full = shuffle_weight(_gather(dn, 1, group), layout=(16, 16))
        out.dn_full_s = _gather(dn_s, 1, group)
    return out


def _gemm(x_q, w, x_s, w_s):
    import aiter
    from aiter import dtypes

    e8 = dtypes.fp8_e8m0
    return aiter.gemm_a8w8_mxfp8_bpreshuffle(
        x_q, w, x_s.view(e8), w_s.view(e8), OUT_DTYPE
    )


def shared_decode(
    sw: SharedWeights, mlp, xq: torch.Tensor, xsc: torch.Tensor
) -> torch.Tensor:
    """TP-partial shared expert of `xq` / `xsc` (the glue's MXFP8 normed rows)."""
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_gemm,
    )
    from vllm.models.minimax_m3.amd.ops.swiglu_oai import swiglu_oai_quantize_mxfp8

    s = xq.shape[0]
    alpha, beta, limit = swiglu_params(mlp)
    gate_up = _gemm(xq, sw.gu, xsc, sw.gu_s)
    a_q, a_s = swiglu_oai_quantize_mxfp8(
        gate_up, alpha, beta, limit, block_m=SWIGLU_BLOCK_M
    )
    dn = mlp.down_proj
    if (
        s <= AITER_DOWN_MAX_ROWS
        or s in AITER_DOWN_EXTRA_ROWS
        or dn.weight_scale.shape[0]
        != dn.weight.shape[0]  # block scales: no Triton route
    ):
        return _gemm(a_q, sw.dn, a_s, sw.dn_s)
    return _mxfp8_dot_scaled_gemm(a_q, a_s, dn.weight, dn.weight_scale, OUT_DTYPE)


def shared_full(
    sw: SharedWeights, mlp, x_q: torch.Tensor, x_s: torch.Tensor
) -> torch.Tensor:
    """The complete shared expert of a rank's own rows (replicated weights)."""
    from vllm.models.minimax_m3.amd.ops.swiglu_oai import swiglu_oai_quantize_mxfp8

    alpha, beta, limit = swiglu_params(mlp)
    gate_up = _gemm(x_q, sw.gu_full, x_s, sw.gu_full_s)
    a_q, a_s = swiglu_oai_quantize_mxfp8(
        gate_up, alpha, beta, limit, block_m=SWIGLU_BLOCK_M
    )
    return _gemm(a_q, sw.dn_full, a_s, sw.dn_full_s)
