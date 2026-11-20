# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""allreduce_norm stage: the post-attention boundary of a sparse layer (TP4).

One launch (CuTe DSL `post/kernel.py` `post_route_kernel`) after the o_proj stage:

1. lands the four ranks' bf16 o_proj partials the o_proj GEMV pushed into this
   rank's boundary exchange (`_exchange.push_workspace(transport, 8, "g3")`),
   sums them in rank order, adds the residual, Gemma RMSNorm;
2. writes the routed experts' NVFP4 activation with the mode-Q per-token scale
   (the M3.1 QAT FC1 row quantizer) and, optionally, an MXFP8 copy;
3. routes: a dedicated warp polls the o_proj router CTAs' tagged partial logits
   (`RouterExchangeSlot`), scales them by the row's 1/rms, runs the
   MiniMax2 top-4 (sigmoid + correction bias, scaled-sum normalize x 2) and
   publishes tagged route lines the fused MoE engine polls.

The hand-off to the MoE stage is `MoeHandoff` (exact layout below).

Buffers come from `_comm.PeerTransport`; every input is an explicit tensor.

Why not upstream `finalize_allreduce_norm.FinalizeAllReduceGemmaRMSNormMxfp8`:
its arithmetic (rank-order fp32 sum, bf16 residual, Gemma RMSNorm, UE8M0
MXFP8 in the 128x4 layout) is the same contract, but its device protocol is
different and cannot be swapped in without rewriting the producers: upstream's
consumer reads a three-generation multicast mailbox filled by its own publish
kernel (BF16 -0.0 sentinel), while here the producers (the o_proj GEMV and the
fused MoE engine) push unicast into a two-parity exchange whose parity flags the
consumer flips, and the same launch also performs routing and the NVFP4 tail.
What is reused: the test oracle uses the same UE8M0 / 128x4 math, and
`_comm.SymmMemPeerTransport` (the mechanism upstream uses) can provide the
peer pointers instead of CUDA IPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from . import _arch
from ._comm import PeerTransport
from ._exchange import HIDDEN, bucket, push_workspace

H = HIDDEN
E = 128
TOP_K = 4
WORLD = 4  # the TP4 default
WORLDS = (4, 8)  # the post kernel's worlds (a compile-time parameter of post/kernel.py)
NPART = 1  # rx parts per source rank: the o_proj router CTAs push a + r as one part
REPL = 16  # route-line replicas: engine CTA bx polls replica bx % REPL
ROUTE_LINE_U64 = 16  # one 128-byte line per (replica, row); words 0..3 = top-4
RCTR_STRIDE = 32  # int32 words per row counter (its own 128-byte line)
NR = 16  # o_proj router CTAs (8 experts each)
RX_PARITIES = 2  # router exchange halves, by tag parity (RouterExchangeSlot)
X4_SF_ROWS = 16  # e4m3 block-scale rows of the x4 scale allocation
X4_ROW_BYTES = H // 16  # 384
S_T_BYTE = X4_SF_ROWS * X4_ROW_BYTES  # fp32 s_t[t] at byte 6144 + 4 t
FC1_QAT_MAX = 6.0 * 448.0  # s_t = bf16(amax / 2688)
EPS = 1e-6
RX_DEADLINE_NS = 2_000_000_000  # routing warp's rx poll deadline (then __trap)
DEFAULT_CL = 8  # CTAs per row (cluster width); 4 / 8 / 16 are instantiated

# Upper bound of the post kernel's static shared memory (the largest instantiation, CL =
# 16: csq 16 x 4 floats + warp partials + mbarrier). The exact value is read from the
# compiled kernel by ``kernel_info``.
POST_SMEM_BOUND = 1024


def _ext(world: int = WORLD):
    if world not in WORLDS:
        raise ValueError(f"post_route: world {world} not in {WORLDS}")
    from .post.host import post_ops

    return post_ops(world)


class RouterExchangeSlot:
    """One layer's router exchange between its o_proj router CTAs (writers,
    every rank's) and its post routing warp (reader, this rank).

    `rx`: u64 `[2][maxt][world][NPART][E]` per rank = (tag << 32) | fp32 bits,
    zero = no call (tags start at 1); call `tag` lives in half `tag & 1`.
    Peer-addressable (each rank's o_proj router CTAs store into every rank's
    slot).

    Why two halves: rank A's o_proj(k + 1) stores into rank B's slot as soon
    as A's own post(k) has finished, which needs B's o_proj(k) data but not
    B's post(k) read, so with one buffer it could overwrite B's unread tag-k
    words (B's post then never sees its tag and traps at the deadline). It
    cannot get two calls ahead: A's o_proj(k + 2) waits (grid dependency) for
    A's post(k + 1), which needs B's o_proj(k + 1) words, stored only after
    B's post(k) finished (B's own grid dependency).

    `rxc` int32 `[maxt][32]`: word 0
    of row t = tag of the last consumed rx call (post side). `oep` int32
    `[NR][32]`: word 0 of router CTA r = its call counter (o_proj side). Every
    o_proj call must be followed by the post call of the same layer, or the
    counters drift and the post traps at its deadline; `reset` re-aligns.
    """

    def __init__(self, transport: PeerTransport, layer_idx: int, maxt: int = 8):
        if transport.world not in WORLDS:
            raise ValueError(
                f"router exchange: world {transport.world} not in {WORLDS}"
            )
        self.transport, self.layer_idx, self.maxt = transport, layer_idx, maxt
        self.world = transport.world
        # u64 words of one tag-parity half
        self.half_words = maxt * self.world * NPART * E
        self.rx = transport.alloc(
            f"minimax_m3_sd.rx.L{layer_idx}.t{maxt}",
            RX_PARITIES * self.half_words * 8,
            align=256,
        )
        dev = self.rx.local.device
        self.bases = (
            self.rx.peer_ptrs_tensor()
        )  # int64 [MAX_PEERS], entries >= world are 0
        self.rxc = torch.zeros(maxt * RCTR_STRIDE, dtype=torch.int32, device=dev)
        self.oep = torch.zeros(NR * RCTR_STRIDE, dtype=torch.int32, device=dev)

    @property
    def rx_local(self) -> int:
        return int(self.rx.local.data_ptr())

    def reset(self) -> None:
        """Collective (every rank, same point, outside graph capture): zero the
        rx words and both counters."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("RouterExchangeSlot.reset during graph capture")
        torch.accelerator.synchronize()
        self.transport.barrier()
        self.rx.local.zero_()
        self.rxc.zero_()
        self.oep.zero_()
        torch.accelerator.synchronize()
        self.transport.barrier()


@dataclass
class MoeHandoff:
    """What the post writes and the fused MoE engine reads (one per layer and
    row count; rows = the engine's maxt, fixed at build).

    =============  ======================  =========================================
    field          dtype / shape           content
    =============  ======================  =========================================
    normed         bf16 [rows, 6144]       Gemma-RMSNorm output (shared-expert input)
    residual_out   bf16 [rows, 6144]       bf16(bf16(sum partials) + residual_in)
    x4_q           uint8 [16, 3072]        NVFP4 codes of normed rows, row-major,
                                           low nibble = even column; rows >= rows
                                           untouched
    x4_sf_full     uint8 [17, 384]         rows 0..15: e4m3 block-16 scales,
                                           row-major `[row][col / 16]`; row 16: fp32
                                           s_t[0..95] (bytes 6144 + 4 t)
    x4_sf          uint8 [16, 384]         view of x4_sf_full[:16]
    x4_pts         fp32 [96]               view of x4_sf_full[16] (s_t per row)
    x4_sfsc        fp32 [1]                1.0 (unused by mode Q; struct field)
    route          int64 [16, rows, 16]    `[replica][row][word]`; words 0..3 =
                                           (tag << 32) | (id << 16) | bf16(w) of
                                           the top-4 in rank order; engine CTA bx
                                           polls replica bx % 16 for tag ==
                                           rctr-epoch + 1
    rctr           int32 [rows, 32]        word 0 of row t = tag of the last
                                           publish (post-owned)
    mx_q, mx_sf    uint8 or None           optional MXFP8 normed (codes [rows,
                                           6144], 128x4-swizzled UE8M0); None in M1
    =============  ======================  =========================================

    Mode Q dequant: value = e2m1(code) * e4m3(sf) * s_t (no global scale); FC1
    is scaled by s_t per token after the GEMM; FC2 uses the fixed scale 16
    folded into the weights (moe stage's weight prep).
    """

    rows: int
    normed: torch.Tensor
    residual_out: torch.Tensor
    x4_q: torch.Tensor
    x4_sf_full: torch.Tensor
    x4_sf: torch.Tensor
    x4_pts: torch.Tensor
    x4_sfsc: torch.Tensor
    route: torch.Tensor
    rctr: torch.Tensor
    mx_q: torch.Tensor | None = None
    mx_sf: torch.Tensor | None = None

    @classmethod
    def allocate(cls, rows: int, device, mxfp8: bool = False) -> MoeHandoff:
        if not 1 <= rows <= X4_SF_ROWS:
            raise ValueError(f"rows {rows} outside 1..{X4_SF_ROWS}")
        x4r = max(rows, X4_SF_ROWS)
        sf_full = torch.zeros(x4r + 1, X4_ROW_BYTES, dtype=torch.uint8, device=device)
        mx_q = mx_sf = None
        if mxfp8:
            mx_q = torch.zeros(rows, H, dtype=torch.uint8, device=device)
            mx_sf = torch.zeros(
                ((rows + 127) // 128 * 128) * (H // 32),
                dtype=torch.uint8,
                device=device,
            )
        return cls(
            rows=rows,
            normed=torch.zeros(rows, H, dtype=torch.bfloat16, device=device),
            residual_out=torch.zeros(rows, H, dtype=torch.bfloat16, device=device),
            x4_q=torch.zeros(x4r, H // 2, dtype=torch.uint8, device=device),
            x4_sf_full=sf_full,
            x4_sf=sf_full[:x4r],
            x4_pts=sf_full[x4r].view(torch.float32),
            x4_sfsc=torch.ones(1, dtype=torch.float32, device=device),
            route=torch.zeros(
                REPL, rows, ROUTE_LINE_U64, dtype=torch.int64, device=device
            ),
            rctr=torch.zeros(rows, RCTR_STRIDE, dtype=torch.int32, device=device),
            mx_q=mx_q,
            mx_sf=mx_sf,
        )


def unpack_route(words: torch.Tensor):
    """(tag, ids, weights fp32) of route words."""
    words = words.to(torch.int64)
    wb = ((words & 0xFFFF) << 16).to(torch.int32)
    return (words >> 32) & 0xFFFFFFFF, (words >> 16) & 0xFFFF, wb.view(torch.float32)


class PostRouteStage:
    """allreduce_norm stage (TP4 or TP8, mode Q). Build once per (layer, rows).

    Args:
        transport: the TP group's peer transport (world 4 or 8).
        rx: the layer's `RouterExchangeSlot` (shared with `OProjRouterStage`).
        post_norm_weight: bf16 [6144], Gemma weight (`1 + w` applied in-kernel).
        router_bias: fp32 [128] e_score_correction_bias.
        rows: decode rows per rank (1..8; the o_proj router path needs a power of 2).
        cl: CTAs per row (4, 8 or 16).

    """

    stage = "allreduce_norm"

    def __init__(
        self,
        *,
        transport: PeerTransport,
        rx: RouterExchangeSlot,
        post_norm_weight: torch.Tensor,
        router_bias: torch.Tensor,
        rows: int,
        cl: int = DEFAULT_CL,
        eps: float = EPS,
        mxfp8_out: bool = False,
        rx_deadline_ns: int = RX_DEADLINE_NS,
        two_round_rx: bool = True,
    ):
        if not 1 <= rows <= 8:
            raise ValueError(f"PostRouteStage: rows {rows} outside 1..8")
        if transport.world not in WORLDS:
            raise ValueError(f"PostRouteStage: world {transport.world} not in {WORLDS}")
        if rx.maxt != bucket(rows):
            raise ValueError(
                f"PostRouteStage: rx slot maxt {rx.maxt} != exchange bucket "
                f"{bucket(rows)}"
            )
        if post_norm_weight.dtype != torch.bfloat16 or post_norm_weight.numel() != H:
            raise ValueError("post_norm_weight must be bf16 [6144]")
        self.transport, self.rx, self.rows, self.cl, self.eps = (
            transport,
            rx,
            rows,
            cl,
            eps,
        )
        self.ws = push_workspace(transport, bucket(rows), "g3")
        self._pp = self.ws.params_tensor().clone()  # host bytes, copied into the launch
        dev = rx.rxc.device
        self.gamma = post_norm_weight.contiguous()
        self.bias = router_bias.float().contiguous().to(dev)
        if self.bias.numel() != E:
            raise ValueError("router_bias must have 128 entries")
        self.out = MoeHandoff.allocate(rows, dev, mxfp8=mxfp8_out)
        self.deadline_ns = int(rx_deadline_ns)
        self.rx2 = 1 if two_round_rx else 0
        self._ext: Any = None
        # (static smem, regs, threads, CTAs/SM, cluster, max active clusters) after
        # build
        self.kernel_info: tuple[int, ...] | None = None

    # -- build ---------------------------------------------------------------
    def grid_use(self) -> _arch.GridUse:
        """The post's footprint for `_arch.check_co_resident` with the MoE
        engine it launches early (exact after `build`)."""
        info = self.kernel_info
        smem = POST_SMEM_BOUND if info is None else int(info[0])
        threads = {4: 224, 8: 128, 16: 96}[self.cl]
        regs = 0 if info is None else int(info[1])
        return _arch.GridUse(
            "allreduce_norm.post",
            self.rows * self.cl,
            smem,
            threads,
            regs_per_thread=regs,
        )

    def build(self) -> None:
        """Compile, check residency, load the module with an idle launch
        (rows = 0). Must run outside graph capture and before the first live
        call: a live post whose module loads lazily can miss the 2 s rx
        deadline while peers wait."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PostRouteStage.build during graph capture")
        spec = _arch.device_spec()
        self._ext = _ext(self.transport.world)
        info = self.kernel_info = tuple(
            int(v)
            for v in self._ext.kernel_info(self.cl, self.rx2, self.out.mx_q is not None)
        )
        occ = info[3]
        # The post never waits on its own grid, but its row clusters spin on peer pushes
        # while the fused MoE engine (launched at the post's entry trigger) waits for
        # the post: every post cluster must be resident at once so none queues behind
        # engine CTAs. Fail closed.
        _arch.check_persistent_grid(
            "allreduce_norm.post", self.rows * self.cl, occ, spec
        )
        if info[5] < self.rows:
            raise _arch.ResidencyError(
                f"allreduce_norm.post: {self.rows} clusters of {self.cl} CTAs must be "
                f"co-resident, {spec.name} "
                f"({spec.sm_count} SMs) holds {info[5]} at once"
            )
        self._launch(0)
        torch.accelerator.synchronize()

    # -- run -----------------------------------------------------------------
    def _launch(self, rows: int, residual: torch.Tensor | None = None) -> None:
        o = self.out
        res = o.residual_out if residual is None else residual
        self._ext.post_route(
            self._pp,
            rows,
            res,
            self.gamma,
            self.eps,
            1,
            o.normed,
            o.residual_out,
            o.mx_q,
            o.mx_sf,
            self.cl,
            self.rx.rx_local,
            self.rx.rxc,
            o.route,
            o.rctr,
            self.bias,
            REPL,
            o.x4_q,
            o.x4_sf,
            o.x4_sfsc,
            o.x4_pts,
            self.deadline_ns,
            0,
            self.rx2,
            int(self.ws.exch.local.data_ptr()),
        )

    def __call__(
        self, residual_mid: torch.Tensor, rows: int | None = None
    ) -> MoeHandoff:
        """Run the post on `residual_mid` (bf16 [rows, 6144], final before
        the o_proj launch). Returns the hand-off buffers (owned by the stage)."""
        if self._ext is None:
            raise RuntimeError("PostRouteStage: build() first")
        rows = self.rows if rows is None else rows
        if rows != self.rows:
            raise ValueError(
                f"PostRouteStage built for {self.rows} rows, called with {rows}"
            )
        if residual_mid.dtype != torch.bfloat16 or not residual_mid.is_contiguous():
            raise ValueError("residual_mid must be contiguous bf16")
        self._launch(rows, residual_mid)
        return self.out


__all__ = [
    "E",
    "H",
    "NR",
    "REPL",
    "RCTR_STRIDE",
    "ROUTE_LINE_U64",
    "TOP_K",
    "MoeHandoff",
    "PostRouteStage",
    "RX_PARITIES",
    "RouterExchangeSlot",
    "unpack_route",
]
