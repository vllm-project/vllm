# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""o_proj stage (TP4): MXFP8 o_proj GEMV + router CTAs, one CuTe DSL launch.

Grid = 16 router CTAs + (6144 / 128) x split GEMV CTAs (split 2: 112 CTAs),
fixed by the layer shapes, not by the SM count. The GEMV CTAs (2-CTA clusters,
tcgen05 block-scaled MMA, A = the o_proj weight, B = the decode's MXFP8 output)
push their bf16 rows into every rank's boundary exchange
(`_exchange.push_workspace(transport, 8, "g3")`, push mode 4) for the
allreduce_norm stage. The router CTAs compute this rank's share of the MoE
router logits from the folded weights (`weights.py`) and push them, tagged,
into every rank's `RouterExchangeSlot`.

Nothing in this grid waits on another CTA outside its own cluster, so it has
no co-residency precondition; dependents launch after each CTA's push. The
allreduce_norm stage (`PostRouteStage`) must run after every call of this
stage on the same layer (the rx tags pair one o_proj call with one post call).

Explicit buffers and transport; compiled at build (`build`), never on a live call;
rows are powers of two
(kernel assertion).
"""

from __future__ import annotations

import torch

from .. import _arch
from .._comm import PeerTransport
from .._exchange import bucket, push_workspace
from ..allreduce_norm import RCTR_STRIDE, RouterExchangeSlot
from . import host as gemv
from . import weights as hw

H, KO, NR = hw.H, hw.KO, hw.NR
WORLD = 4
SPLIT = 2
# (split, ring, contig, nacc, a_tmem, st_async, stagger, early_w, qfirst, burst, ptrig,
# pfx): the shipped TP4 o_proj config (K 2048: 8 k-tiles per CTA, ring 8 = 128 KB).
# Then (router CTAs, pairing check 1 = trap).
BASE_CFG = (SPLIT, 8, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1)
RA_CHECK = 1
PUSH_MODE = 4  # unicast into the boundary exchange (4 column slices per row)
RA_TAB_WORDS = 8
O_PROJ_SMEM_BOUND = (
    224 * 1024
)  # router CTAs: max dynamic smem (W'' + W' slices, 224 KB carveout)


def grid_ctas(n_out: int = H, split: int = SPLIT, ra: int = NR) -> int:
    return (n_out // 128) * split + ra


class OProjRouterStage:
    """o_proj + router CTAs for one sparse layer on one rank.

    Build-time inputs (addresses are baked into the router table, so the
    activation buffers are fixed for the stage's lifetime, as in a CUDA graph):
        transport: peer transport (world 4).
        rx: the layer's RouterExchangeSlot (shared with PostRouteStage).
        o_weight: e4m3 [6144, 2048] (this rank's column slice of W_o).
        o_weight_scale: uint8 UE8M0, 128x4-swizzled (FlashInfer F8_128x4, as
            vLLM's MXFP8 linear stores it).
        router_weight: [128, 6144] (fp32 or bf16), post_norm_weight: bf16 [6144].
        residual_mid: bf16 [rows, 6144] (the front's residual output).
        xq: e4m3 [rows, 2048], xsf: uint8 128x4-swizzled scales (>= 128 * 64 B),
            the sparse_decode stage's MXFP8 output.
        rows: 1, 2, 4 or 8.
        blobs: optional precomputed `(blob_a, blob_r)` of `weights.blobs`
            for these weights and this rank (shared across row counts).
    """

    stage = "o_proj"

    def __init__(
        self,
        *,
        transport: PeerTransport,
        rx: RouterExchangeSlot,
        o_weight: torch.Tensor,
        o_weight_scale: torch.Tensor,
        router_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        residual_mid: torch.Tensor,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        rows: int,
        blobs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if rows not in (1, 2, 4, 8):
            raise ValueError(
                f"OProjRouterStage: rows {rows} must be 1, 2, 4 or 8 (router CTA row "
                "passes)"
            )
        if transport.world != WORLD:
            raise ValueError(f"OProjRouterStage: world {transport.world}, need {WORLD}")
        if o_weight.shape != (H, KO) or o_weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"o_weight must be e4m3 [{H}, {KO}], got {o_weight.dtype} "
                f"{tuple(o_weight.shape)}"
            )
        if (
            residual_mid.dtype != torch.bfloat16
            or residual_mid.shape[-1] != H
            or not residual_mid.is_contiguous()
        ):
            raise ValueError("residual_mid must be contiguous bf16 [rows, 6144]")
        if (
            xq.dtype != torch.float8_e4m3fn
            or tuple(xq.shape) != (rows, KO)
            or not xq.is_contiguous()
        ):
            raise ValueError(f"xq must be contiguous e4m3 [{rows}, {KO}]")
        if xq.data_ptr() % 16 or xsf.numel() < 128 * (KO // 32):
            raise ValueError(
                "xq must be 16-byte aligned and xsf hold 128 x 64 swizzled scale bytes"
            )
        # the router CTAs size the rx parity half by the exchange bucket
        if rx.maxt != bucket(rows):
            raise ValueError(
                f"rx slot maxt {rx.maxt} != exchange bucket {bucket(rows)}"
            )
        self.transport, self.rx, self.rows = transport, rx, rows
        self.rank = transport.rank
        dev = xq.device
        self.w = o_weight.contiguous()
        self.ws_scale = o_weight_scale.contiguous().view(torch.uint8).reshape(-1)
        self.residual, self.xq, self.xsf = residual_mid, xq, xsf
        if blobs is None:
            sf = hw.unswizzle_128x4(self.ws_scale, H, KO // 32)
            a, r, _, _ = hw.blobs(
                router_weight, post_norm_weight, self.w, sf, self.rank, WORLD
            )
        else:
            # Integration: the blobs depend only on the layer's weights and the rank, so
            # the stages of one layer (one per row count) share them (``hw.blobs`` folds
            # on the CPU in fp64 and holds 4 MB per rank).
            a, r = blobs
        self.blob_a, self.blob_r = a.to(dev), r.to(dev)
        self.ws = push_workspace(transport, bucket(rows), "g3")
        self.ar_peers = self.ws.exch.peer_ptrs_tensor()
        self.y = torch.empty(rows, H, dtype=torch.bfloat16, device=dev)
        self.tab = torch.tensor(
            [
                self.blob_a.data_ptr(),
                self.blob_r.data_ptr(),
                residual_mid.data_ptr(),
                xq.data_ptr(),
                xsf.data_ptr(),
                rx.bases.data_ptr(),
                rx.oep.data_ptr(),
                rx.rxc.data_ptr(),
            ],
            dtype=torch.int64,
            device=dev,
        )
        assert self.tab.numel() == RA_TAB_WORDS
        assert rx.oep.numel() >= NR * RCTR_STRIDE
        self.cfg = BASE_CFG + (NR, RA_CHECK)
        # (exchange peer table, parity flags, rank, world, bucket rows, push mode)
        self.ar = (
            self.ar_peers.data_ptr(),
            self.ws.flags.data_ptr(),
            self.rank,
            WORLD,
            self.ws.maxt,
            PUSH_MODE,
        )
        self._ptrs = (residual_mid.data_ptr(), xq.data_ptr(), xsf.data_ptr())
        self._built = False

    def grid_use(self) -> _arch.GridUse:
        # ring 8 x 16 KB weight tiles + activation / scales / mailbox; the router CTAs
        # reuse it (<= 224 KB)
        return _arch.GridUse("o_proj", grid_ctas(), O_PROJ_SMEM_BOUND, 256)

    def build(self) -> None:
        """Compile outside capture (fail early, never on a live call) and check
        the launch fits the device."""
        gemv.check_cute_dsl()
        spec = _arch.device_spec()
        if spec.smem_per_block_opt_in < O_PROJ_SMEM_BOUND:
            raise _arch.ResidencyError(
                f"o_proj: needs up to 224 KB dynamic smem per CTA, {spec.name} allows "
                f"{spec.smem_per_block_opt_in}"
            )
        gemv.compile_call(
            self.xq,
            self.xsf,
            self.w,
            self.ws_scale,
            self.cfg,
            self.ar,
            int(self.tab.data_ptr()),
            self.y,
        )
        self._built = True

    def __call__(self) -> None:
        if not self._built:
            raise RuntimeError("OProjRouterStage: build() first")
        if (
            self.residual.data_ptr(),
            self.xq.data_ptr(),
            self.xsf.data_ptr(),
        ) != self._ptrs:
            raise RuntimeError(
                "OProjRouterStage: residual / xq / xsf moved after the build"
            )
        y = gemv.m3_mx_gemv(
            self.xq,
            self.xsf,
            self.w,
            self.ws_scale,
            cfg=self.cfg,
            use_pdl=True,
            ar=self.ar,
            ra_tab=int(self.tab.data_ptr()),
            out=self.y,
        )
        if y is None:
            raise RuntimeError(
                f"o_proj GEMV declined (uncompiled key {gemv.last_uncompiled})"
            )


__all__ = ["OProjRouterStage", "grid_ctas"]
