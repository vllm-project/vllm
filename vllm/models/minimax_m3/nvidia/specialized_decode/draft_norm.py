# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft layers' Gemma RMSNorms fused with the MXFP8 quantize of
the linear after them.

Each draft layer all-reduces two row-parallel outputs (`o_proj`,
`down_proj`), normalizes the sum right after (with the residual add), and
the next linear (`gate_up_proj`, the next layer's `qkv_proj`) quantizes
it to MXFP8: the row norm and the quantize are two launches, ten times per
draft step.

Here (`FusedGemmaNormQuant`) the stock all-reduce stays; the residual add,
the norm and the quantize run in one launch, and the result is handed to the
linear as a `QuantizedActivation` (also for the context states' norm before
the context K/V projection). The norm is exactly that of
`drafter.GemmaRMSNormRows` (the residual add rounded to the residual's
dtype and stored, the sum of squares over the same 8192-wide block,
`x * rstd * (1 + w)`), and the quantize is FlashInfer's MXFP8 quantize bit
for bit (UE8M0 round-up, RN e4m3, the F8_128x4 scale layout).

Fusing the all-reduce as well (a peer-memory Lamport exchange inside the
norm) measured slower at TP4 than FlashInfer's MNNVL all-reduce, which uses
NVLS multicast, so it is not done.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Rows per call: a draft block of 8 requests x 8 query tokens.
MAX_ROWS = 64
_BLOCK = 8192  # the row norm's block (next power of two of hidden 6144)


@triton.jit
def _norm_quant_kernel(
    x_ptr,
    x_stride,
    res_ptr,
    res_stride,
    w_ptr,
    scratch_ptr,  # bf16 [rows, n], the norm before the quantize
    n,
    eps,
    q_ptr,  # e4m3 [rows, n]
    sf_ptr,  # UE8M0 scales, F8_128x4-swizzled
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
    # The row norm (drafter._gemma_rmsnorm_kernel).
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * res_stride + cols, mask=mask, other=0.0)
        x = (x.to(tl.float32) + r.to(tl.float32)).to(r.dtype)
    tl.store(res_ptr + row * res_stride + cols, x, mask=mask)
    xf = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / n + eps)
    y = xf * rstd * (1.0 + w)
    # FlashInfer's MXFP8 quantize of the BF16 norm output: per 32 values, s = UE8M0
    # rounded up of amax * (1 / 448) (exponent bits, +1 for a non-zero mantissa), codes
    # = e4m3(clamp(v * 2^(127 - s), +-448)). Through a scratch row, re-read as [groups,
    # 32].
    tl.store(scratch_ptr + row * n + cols, y.to(tl.bfloat16), mask=mask)
    tl.debug_barrier()
    grp = tl.arange(0, BLOCK // 32)
    gmask = grp < n // 32
    e32 = tl.arange(0, 32)
    offs = grp[:, None] * 32 + e32[None, :]
    yb = tl.load(scratch_ptr + row * n + offs, mask=gmask[:, None], other=0.0)
    yb = yb.to(tl.float32)
    amax = tl.max(tl.abs(yb), axis=1)
    nm = amax * (1.0 / 448.0)
    nbits = nm.to(tl.int32, bitcast=True)
    expo = (nbits >> 23) & 255
    mant = nbits & 0x7FFFFF
    bump = (mant != 0) & ~((expo == 0) & (mant <= 0x400000))
    s = tl.minimum(expo + bump.to(tl.int32), 254)
    s = tl.where(nm <= 0.0, 0, s)
    inv = (tl.maximum(254 - s, 0) << 23).to(tl.float32, bitcast=True)
    inv = tl.where(s == 0, 0.0, inv)
    qv = yb * inv[:, None]
    qv = tl.minimum(tl.maximum(qv, -448.0), 448.0)
    tl.store(q_ptr + row * n + offs, qv.to(tl.float8e4nv), mask=gmask[:, None])
    k_pad = (n // 32 + 3) // 4 * 4
    sf_off = (
        (grp % 4)
        + (grp // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * 128 * k_pad
    )
    tl.store(sf_ptr + sf_off, s.to(tl.uint8), mask=gmask)


def _quant_buffers(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    return (
        torch.empty(rows, n, dtype=torch.float8_e4m3fn, device=x.device),
        torch.empty(
            (rows + 127) // 128 * 128 * ((n // 32 + 3) // 4 * 4),
            dtype=torch.uint8,
            device=x.device,
        ),
    )


def _quantized(quant, x: torch.Tensor):
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    return QuantizedActivation(
        quant[0], quant[1], x.dtype, quant[0].shape, kMxfp8Dynamic
    )


def _takes_mxfp8(consumer: nn.Module | None) -> bool:
    from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    return consumer is not None and get_input_quant_key(consumer) == kMxfp8Dynamic


_SCRATCH: dict[torch.device, torch.Tensor] = {}


class FusedGemmaNormQuant:
    """The forward of a draft Gemma RMSNorm after the stock all-reduce, with
    the next linear's MXFP8 quantize in the same launch: residual add, norm
    and quantize, bit for bit as the row norm kernel followed by FlashInfer's
    quantize."""

    def __init__(self, norm: nn.Module, consumer: nn.Module | None, stock, guard=None):
        self._norm = norm
        self._consumer = consumer
        self._quant: bool | None = None
        self._guard = guard  # () -> bool: the consumer is the norm's only reader
        self.stock = stock

    def __call__(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if self._quant is None:
            self._quant = _takes_mxfp8(self._consumer) and (
                self._guard is None or self._guard()
            )
        if (
            not self._quant
            or x.dim() != 2
            or not 0 < x.shape[0] <= MAX_ROWS
            or x.dtype != torch.bfloat16
            or x.stride(-1) != 1
            or x.shape[1] % 32
            or x.shape[1] > _BLOCK
            or (
                residual is not None
                and (residual.shape != x.shape or residual.stride(-1) != 1)
            )
        ):
            return self.stock(x, residual)
        scratch = _SCRATCH.get(x.device)
        if scratch is None or scratch.shape[1] != x.shape[1]:
            scratch = _SCRATCH[x.device] = torch.empty(
                MAX_ROWS, x.shape[1], dtype=torch.bfloat16, device=x.device
            )
        quant = _quant_buffers(x)
        norm = self._norm
        res = residual if residual is not None else x  # x is rewritten unchanged
        _norm_quant_kernel[(x.shape[0],)](
            x,
            x.stride(0),
            res,
            res.stride(0),
            norm.weight,
            scratch,
            x.shape[1],
            norm.variance_epsilon,
            quant[0],
            quant[1],
            HAS_RESIDUAL=residual is not None,
            BLOCK=_BLOCK,
            num_warps=8,
        )
        out = _quantized(quant, x)
        if residual is None:
            return out
        return out, residual


def fused_norms_unsupported(model: nn.Module) -> str:
    """Why the draft norms keep their stock forward ("" when they fuse)."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return "no draft layers"
    for layer in layers:
        for norm in (layer.input_layernorm, layer.post_attention_layernorm):
            if type(norm).__name__ != "MiniMAXGemmaRMSNorm":
                return f"a {type(norm).__name__} draft norm"
    if type(getattr(model.model, "norm", None)).__name__ != "MiniMAXGemmaRMSNorm":
        return "a final norm that is not a Gemma RMSNorm"
    if model.model.norm.weight.dtype != torch.bfloat16:
        return f"a {model.model.norm.weight.dtype} draft model"
    return ""


def attach_fused_norms(model: nn.Module, stock_forward) -> int:
    """Fuse each draft norm with the MXFP8 quantize of the linear it feeds.
    `stock_forward(norm)`: the (already patched) forward of a norm. Returns
    the number of fused norms."""
    reason = fused_norms_unsupported(model)
    if reason:
        logger.info("DSpark draft norms stay separate: %s.", reason)
        return 0
    inner = model.model
    count = 0
    for layer in inner.layers:
        attn = layer.self_attn
        pairs = [
            # The draft attention reads its input's dtype for the attention output
            # unless it states one (FLASHINFER_TS with an FP8 query): only then may the
            # input norm hand it a QuantizedActivation.
            (
                layer.input_layernorm,
                attn.qkv_proj,
                lambda attn=attn: getattr(attn, "_attn_output_dtype", None) is not None,
            ),
            (layer.post_attention_layernorm, layer.mlp.gate_up_proj, None),
        ]
        for norm, consumer, guard in pairs:
            norm.forward = FusedGemmaNormQuant(
                norm, consumer, stock_forward(norm), guard=guard
            )
            count += 1
    # The context states' norm feeds the fused context K/V projection.
    hidden_norm = getattr(inner, "hidden_norm", None)
    ctx_proj = getattr(inner, "_context_kv_proj", None)
    if hidden_norm is not None and ctx_proj is not None:
        hidden_norm.forward = FusedGemmaNormQuant(
            hidden_norm,
            ctx_proj,
            stock_forward(hidden_norm),
            # A BF16 fused K/V weight takes precedence over the projection.
            guard=lambda: getattr(inner, "_fused_kv_weight", None) is None,
        )
        count += 1
    logger.info("DSpark draft norms: %d with the MXFP8 quantize fused.", count)
    return count
