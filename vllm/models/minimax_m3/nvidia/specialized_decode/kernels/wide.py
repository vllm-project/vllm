# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernels of steps wider than the fused kernels' rows (9..64 rows).

The fused o_proj / post / MoE engine kernels hold at most 8 rows (the engine's
slot tiling, the router CTAs' register shape). Wider steps (c = 2..8 requests
x 8 rows at K = 7) run these slots instead:

==================  ===========================================================
stage               wide implementation
==================  ===========================================================
front (17..64)      `WideFront`: after the stock input boundary (FlashInfer
                    AR + norm), one MXFP8 quantize launch and the same GEMV on
                    the QKV / index weight, then the stock q/k norm + RoPE +
                    FP8 KV / index-K insert op.
o_proj              `WideOProj`: the same MXFP8 GEMV (tcgen05, swapped AB:
                    the weight is the M128 operand, the rows are the MMA's N =
                    16 / 32 / 48 / 64 columns), so each o_proj weight tile is
                    read once per step at any row count; it reads the kernel
                    sparse decode's MXFP8 output in place (no requantize).
                    BF16 partial out, no push.
allreduce_norm      stock (FlashInfer's fused all-reduce + Gemma RMSNorm,
                    the stock router GEMM).
moe                 stock (above 8 rows the routed GEMMs stream nearly every
                    expert's weights once either way).
==================  ===========================================================

Why not the push + post at these rows: the producer push writes 4-byte words
to every peer, which is slower than FlashInfer's all-reduce + norm at 64 rows,
and the MoE engine reads routes per slot. Why not the fused engine: at 32-64
rows x top-4 most of the 128 experts are hit and the routed GEMMs stream
nearly every expert's weights once either way, which FlashInfer's grouped
GEMMs already do near the HBM roofline. A separate router + NVFP4 launch with
FlashInfer's routed entry point measured slower end to end at c = 4 / 8 than
the stock router GEMM + monolithic MoE.
"""

from __future__ import annotations

from typing import Any

import torch

H = 6144
KO = 2048  # o_proj input per rank at TP4 (16 heads x 128)
MAX_ROWS = 64


def _rows_ext():
    from .post.host import row_ops

    return row_ops()


class WideError(RuntimeError):
    pass


class WideOProj:
    """The o_proj MXFP8 GEMV at 9..64 rows on one layer's weight (no router
    CTAs, no push): bf16 partial [rows, 6144] from the kernel sparse decode's
    MXFP8 output."""

    def __init__(
        self, o_weight: torch.Tensor, o_weight_scale: torch.Tensor, out: torch.Tensor
    ):
        from .o_proj import host as gemv

        self.gemv = gemv
        if o_weight.shape != (H, KO) or o_weight.dtype != torch.float8_e4m3fn:
            raise WideError(f"o_proj weight must be e4m3 [{H}, {KO}]")
        self.w = o_weight.contiguous()
        self.ws = o_weight_scale.contiguous().view(torch.uint8).reshape(-1)
        self.out = out  # bf16 [64, 6144], shared by the layers (stream order)

    def build(self, rows: tuple[int, ...]) -> None:
        """Compile the MMA widths `rows` need (outside capture)."""
        self.gemv.check_cute_dsl()
        for m in rows:
            if (
                self.gemv.precompile(self.w, self.ws, m, x_bf16=False, use_pdl=True)
                is None
            ):
                raise WideError(f"o_proj GEMV declines {m} rows")

    def __call__(self, xq: torch.Tensor, xsf: torch.Tensor) -> torch.Tensor:
        rows = xq.shape[0]
        y = self.gemv.m3_mx_gemv(
            xq, xsf, self.w, self.ws, use_pdl=True, out=self.out[:rows]
        )
        if y is None:
            raise WideError(
                f"o_proj GEMV declined {rows} rows (uncompiled key "
                f"{self.gemv.last_uncompiled})"
            )
        return y


class WideFront:
    """The QKV / index projection of a sparse layer at 17..64 rows: MXFP8 of
    the normed BF16 rows (`mxfp8_rows`, trtllm `mxfp8_quantize` rounding)
    and the GEMV on the layer's MXFP8 `qkv_proj` weight (rows =
    q | k | v | index q | index k heads x 128)."""

    def __init__(
        self,
        qkv_weight: torch.Tensor,
        qkv_weight_scale: torch.Tensor,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        out: torch.Tensor,
    ):
        from .o_proj import host as gemv

        self.gemv = gemv
        if (
            qkv_weight.dim() != 2
            or qkv_weight.shape[1] != H
            or qkv_weight.dtype != torch.float8_e4m3fn
        ):
            raise WideError(f"qkv weight must be e4m3 [N, {H}]")
        self.w = qkv_weight.contiguous()
        self.ws = qkv_weight_scale.contiguous().view(torch.uint8).reshape(-1)
        self.n = self.w.shape[0]
        self.xq, self.xsf = (
            xq,
            xsf,
        )  # shared MXFP8 input buffers [64, H] / [128 * H / 32]
        if out.shape[1] != self.n:
            raise WideError(f"qkv out width {out.shape[1]} != {self.n}")
        self.out = out  # bf16 [64, N], shared (stream order)
        self._ext: Any = None

    def build(self, rows: tuple[int, ...], compile_gemv: bool = True) -> None:
        """Load the quantize extension; compile the GEMV for `rows` (once
        per process: the compile key does not depend on the weight)."""
        self._ext = _rows_ext()
        self._ext.precompile()  # its first launch is inside a FULL graph capture
        if not compile_gemv:
            return
        self.gemv.check_cute_dsl()
        for m in rows:
            if (
                self.gemv.precompile(self.w, self.ws, m, x_bf16=False, use_pdl=True)
                is None
            ):
                raise WideError(f"QKV GEMV declines {m} rows")

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        """Qkv bf16 [rows, N] of the normed rows `hidden` (bf16 [rows, 6144])."""
        rows = hidden.shape[0]
        self._ext.mxfp8_rows(hidden, self.xq, self.xsf, rows)
        return self.quantized(self.xq[:rows], self.xsf)

    def quantized(self, xq: torch.Tensor, xsf: torch.Tensor) -> torch.Tensor:
        """Qkv bf16 [rows, N] of rows already in MXFP8: e4m3 `xq` [rows,
        6144] and UE8M0 `xsf` in the 128x4-swizzled layout (at least 128 x
        192 bytes), as `mxfp8_rows` and the stock fused MoE finalize +
        all-reduce + norm write them."""
        rows = xq.shape[0]
        y = self.gemv.m3_mx_gemv(
            xq, xsf, self.w, self.ws, use_pdl=True, out=self.out[:rows]
        )
        if y is None:
            raise WideError(
                f"QKV GEMV declined {rows} rows (uncompiled key "
                f"{self.gemv.last_uncompiled})"
            )
        return y
