# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `kernels/` registry entries behind the `stages.py` interfaces.

Each adapter below is the kernel implementation of one stage slot. It keeps the
slot's contract towards the stock stages on either side wherever the kernels
allow, builds the caller-side layouts the kernels need, and declares the
partners it cannot run without (`compatible`):

==============  ===========================  =================================
slot            adapter                      runs beside
==============  ===========================  =================================
front           `KernelFront`              anything; fuses the input
                                             boundary of the sparse layers
                                             (AR land + residual + input norm)
index_score     `KernelIndexScore`         anything (stock contract)
page_topk       `KernelPageTopK`           `KernelSparseDecode` only
                                             (physical page table)
page_topk       `KernelPageTopKLogical`    anything (stock top-k buffer)
sparse_decode   `KernelSparseDecode`       anything (reads either table)
o_proj          `KernelOProj`              `KernelAllReduceNorm` only
allreduce_norm  `KernelAllReduceNorm`      `KernelOProj` only (the pair)
moe             `KernelMoE`                `KernelAllReduceNorm` only
                                             (routes + NVFP4 hand-off)
front           `KernelFrontWide`          anything (stock contract)
o_proj          `KernelOProjWide`          anything (BF16 partial)
==============  ===========================  =================================

The `*Wide` adapters serve the steps above the pair's 8 rows (the front's
16) up to 64 (`kernels/wide.py`): the MXFP8 GEMV QKV projection and
o_proj. Above 8 rows the post-attention boundary and the MoE run stock.

The kernels cover the sparse MoE layers (3-59, "covered" layers); on every
other layer each adapter runs its stock stage. `KernelContext` owns what the
adapters share: the TP peer transport and its exchanges, per-layer activation
buffers whose addresses the kernels bind at build, the kernel instances, and
the per-step metadata upstream does not produce (physical page rows, sparse
lengths).

Boundary semantics against the stock chain (per covered layer `L`):

- input boundary: stock `forward_input_norm` all-reduces the previous FFN's
  partial (finalizing a deferred MoE), adds the residual and norms. The kernel
  front does the same in its boundary CTAs: from the previous kernel MoE's
  push (`PushedFFN`, prepushed mode) or from a BF16 partial it pushes itself
  (a stock MoE or dense MLP, never left unfinalized: the executor does not
  defer a MoE whose next front fuses the input).
- post-attention boundary: stock `fused_allreduce_gemma_rms_norm` + the FP32
  router GEMM. The kernel pair is the o_proj GEMV (+ router CTAs on the folded
  router weight) pushing into the post kernel, which lands the all-reduce, adds
  the residual, norms, quantizes the NVFP4 FC1 input and publishes the top-4
  routes. It returns no router logits: a stock MoE behind it computes them
  from the normed rows; the kernel MoE reads the routes.
- MoE: the kernel MoE never defers its finalize; it pushes into the next
  layer's kernel front when that front polls (`polls_push`), else returns
  its BF16 partial for the stock boundary.

Build (`KernelContext.build`, from the first kernel stage's `warm_up`,
before memory profiling and capture) is collective and fail closed: all peer
memory is allocated first in one fixed order, then the CuTe DSL kernel modules load,
then the kernel instances build; after each phase the ranks agree (MIN over
the TP CPU group) on which stages built, and a stage that failed on any rank
runs stock on every rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.distributed as dist

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

from . import PAGE_SIZE, CacheDtypes, DecodeConfig
from .kernels import FRONT_MAX_ROWS, MAX_ROWS, PAIR_MAX_ROWS, fallback_stages
from .kernels._mxfp8 import mxfp8_linear_nk
from .stages import (
    AllReduceNormStage,
    BoundaryOutput,
    FrontOutput,
    FrontStage,
    IndexScoreStage,
    MoEStage,
    OProjStage,
    PageTopKStage,
    PushedFFN,
    SparseDecodeStage,
    StepStages,
)
from .stock import (
    StockAllReduceNorm,
    StockFront,
    StockIndexScore,
    StockMoE,
    StockOProj,
    StockPageTopK,
    StockSparseDecode,
    _finalize_moe_output,
    _indexer_metadata,
)

if TYPE_CHECKING:
    from ..model import (
        MiniMaxM3DecoderLayer,
        MiniMaxM3Model,
        MiniMaxM3MoE,
    )
    from .kernels._comm import PeerTransport
    from .runtime.decode import DecodeStep
    from .stages import LayerInput, PendingFFN

logger = init_logger(__name__)

# Rows of the largest step any kernel stage runs (the largest registry ``max_rows``; per
# stage: front 16, attention trio 64, o_proj / post / MoE 8).
KERNEL_MAX_ROWS = MAX_ROWS
# Row counts of the o_proj router CTAs (powers of two); the post and the MoE are built
# per row count behind them.
PAIR_ROWS = tuple(r for r in (1, 2, 4, 8, 16, 32, 64) if r <= PAIR_MAX_ROWS)
# Exchange buckets of the kernel front (its bucket-8 and bucket-16 builds).
FRONT_BUCKETS = tuple(b for b in (8, 16) if b <= FRONT_MAX_ROWS)
PAGE = PAGE_SIZE  # the deployment check refuses any other block size
HIDDEN = 6144
HEAD_DIM = 128
# The kernels read and write fp8 (E4M3) KV and index-K caches only.
FP8_CACHES = frozenset({CacheDtypes(kv="fp8", index="fp8")})
# Kernel names, in build order (dependencies first): the registry stages, and the wide
# variants of the pair's slots ("<slot>_wide", registry variant "wide").
KERNEL_STAGES = (
    "front",
    "index_score",
    "page_topk",
    "sparse_decode",
    "allreduce_norm",
    "o_proj",
    "moe",
    "o_proj_wide",
    "front_wide",
)
# Stages built per row count behind the o_proj router CTAs.
PAIR_STAGES = ("o_proj", "allreduce_norm", "moe")
# The wide variants (rows above the pair's / the front's).
WIDE_STAGES = ("o_proj_wide", "front_wide")


def slot_of(kernel: str) -> tuple[str, str]:
    """(stage slot, registry variant) of a kernel name."""
    if kernel.endswith("_wide"):
        return kernel[: -len("_wide")], "wide"
    return kernel, ""


# Peer-arena alignment of every buffer (the slack is this per allocation).
_ARENA_ALIGN = 256


def rx_slot_key(layer_id: int, rows: int) -> int:
    """`RouterExchangeSlot`'s `layer_idx` (it only names the buffer): one
    slot per (layer, row count), so that the o_proj and post call counters of
    different row counts never interleave on one slot."""
    return layer_id * 16 + rows


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16).contiguous()


def _relerr(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    g, w = got.float(), want.float()
    return (g - w).abs().amax() / w.abs().amax().clamp_min(1e-12)


@dataclass
class LayerKernels:
    """One covered layer's kernel weights views, buffers and instances."""

    layer_id: int
    layer: Any  # MiniMaxM3DecoderLayer
    # The previous local layer is covered: its kernel MoE may push into this layer's
    # front (which then also has a prepushed instance).
    after_covered: bool
    # The next local layer is covered: this layer's MoE may push.
    before_covered: bool
    # Buffers (M = the largest kernel step's rows, at most KERNEL_MAX_ROWS; steps use
    # row-prefix views, so every bound address is the same for every row count).
    r_mid: torch.Tensor | None = None  # bf16 [M, H]: residual after the input boundary
    q_fp8: torch.Tensor | None = None  # e4m3 [M, nq * 128]
    q_bf16: torch.Tensor | None = None  # bf16 [M, nq * 128] (Triton stock decode only)
    index_q: torch.Tensor | None = None  # e4m3 [M, niq * 128]
    table: torch.Tensor | None = None  # int32 [Hkv, M, 16]: physical top-k pages
    out: torch.Tensor | None = None  # bf16 [M, nq * 128]: attention output
    xq: torch.Tensor | None = None  # e4m3 [M, nq * 128]: its MXFP8 values
    xsf: torch.Tensor | None = None  # uint8 [128 * nq * 4]: 128x4-swizzled UE8M0
    sfp: torch.Tensor | None = None  # uint8 [M, nq * 4]: row-major UE8M0
    front_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    # Instances. QkvFrontStage, prepushed=False (takes a partial), by exchange bucket
    # (8: rows 1..8, 16: rows 9..16).
    front_x: dict[int, Any] = field(default_factory=dict)
    # QkvFrontStage, prepushed=True (polls the push), bucket 8: the kernel MoE (which
    # pushes) runs up to 8 rows.
    front_pushed: Any = None
    sparse: Any = None  # ClusterSparseDecodeStage
    rx: dict[int, Any] = field(default_factory=dict)  # rows -> RouterExchangeSlot
    post: dict[int, Any] = field(default_factory=dict)  # rows -> PostRouteStage
    o_proj: dict[int, Any] = field(default_factory=dict)  # rows -> OProjRouterStage
    moe: dict[int, Any] = field(default_factory=dict)  # rows -> FusedMoeStage
    ffn_out: dict[int, torch.Tensor] = field(default_factory=dict)
    # Wide variants (rows above the pair's): kernels/wide.py.
    wide_front: Any = None  # WideFront
    wide_o_proj: Any = None  # WideOProj

    @property
    def attn(self):
        return self.layer.self_attn

    @property
    def block_sparse_moe(self):
        return self.layer.block_sparse_moe


class OProjPending(NamedTuple):
    """`KernelOProj`'s output: the o_proj launch, made by
    `KernelAllReduceNorm.post_attention` once the residual is known (the
    router CTAs read it), immediately before the post it pairs with."""

    lk: LayerKernels
    attn_out: torch.Tensor


class KernelContext:
    """What the kernel stages of one executor share (see the module
    docstring). Host-only until `build`."""

    def __init__(
        self,
        backbone: MiniMaxM3Model,
        config: DecodeConfig,
        enabled: frozenset[str],
    ) -> None:
        from vllm.distributed import get_tp_group

        from ..model import MiniMaxM3DecoderLayer, MiniMaxM3SparseAttention
        from .kernels import _arch

        self.backbone = backbone
        self.config = config
        self.enabled = frozenset(enabled)
        self.step_rows = tuple(r for r in config.step_rows if 1 <= r <= KERNEL_MAX_ROWS)
        group = get_tp_group()
        self.cpu_group = group.cpu_group
        self.device_group = group.device_group
        self.rank = group.rank_in_group
        self.tp = group.world_size
        try:
            self.arch: str | None = _arch.device_spec().arch
        except Exception as err:  # UnsupportedArch, no CUDA
            logger.warning("MiniMax-M3.1 specialized decode kernels off: %s", err)
            self.arch = None
        self.built = False
        self.ok: set[str] = set()
        self.rows_ok: dict[str, tuple[int, ...]] = {}
        self.transport: PeerTransport | None = None
        self.index_score = None
        self.page_topk = None
        self.identity_rows: torch.Tensor | None = None
        self.sf_index: torch.Tensor | None = None
        self.row_ids: torch.Tensor | None = None
        # The wide o_proj's BF16 partial [64, H] (shared by the layers).
        self.wide_y: torch.Tensor | None = None
        # Per step (set by begin_step).
        self._step: object | None = None
        self._page_rows: dict[int, torch.Tensor] = {}
        self.lens: torch.Tensor | None = None

        # Coverage: sparse-attention MoE layers of the geometry the kernels were written
        # for (TP4: 16 query heads per KV head, one KV / index head, 128-dim heads,
        # hidden 6144).
        self.layers: dict[int, LayerKernels] = {}
        layer_ids = list(range(backbone.start_layer, backbone.end_layer))
        covered: list[int] = []
        for layer_id in layer_ids:
            layer = backbone.layers[layer_id]
            assert isinstance(layer, MiniMaxM3DecoderLayer)
            attn = layer.self_attn
            if not (
                isinstance(attn, MiniMaxM3SparseAttention)
                and layer.is_moe_layer
                and not layer.block_sparse_moe.use_mega_moe
            ):
                continue
            geometry = (
                attn.hidden_size == HIDDEN
                and attn.head_dim == HEAD_DIM
                and attn.idx_head_dim == HEAD_DIM
                and attn.num_heads == 16 * attn.num_kv_heads
                and attn.num_idx_heads == attn.num_kv_heads
            )
            if not geometry:
                logger.warning_once(
                    "MiniMax-M3.1 specialized decode: layer %d has a geometry the "
                    "kernels do not cover; it runs the stock stages.",
                    layer_id,
                )
                continue
            covered.append(layer_id)
        covered_set = set(covered)
        for layer_id in covered:
            self.layers[layer_id] = LayerKernels(
                layer_id=layer_id,
                layer=backbone.layers[layer_id],
                after_covered=layer_id - 1 in covered_set
                and layer_id - 1 >= backbone.start_layer,
                before_covered=layer_id + 1 in covered_set
                and layer_id + 1 < backbone.end_layer,
            )
        self._by_attn = {id(lk.attn): lk for lk in self.layers.values()}
        self._by_layer = {id(lk.layer): lk for lk in self.layers.values()}
        self._by_moe = {id(lk.block_sparse_moe): lk for lk in self.layers.values()}

    # -- lookup ---------------------------------------------------------------
    def by_attn(self, attn) -> LayerKernels | None:
        return self._by_attn.get(id(attn))

    def by_layer(self, layer) -> LayerKernels | None:
        return self._by_layer.get(id(layer))

    def by_moe(self, moe) -> LayerKernels | None:
        return self._by_moe.get(id(moe))

    def local_index(self, lk: LayerKernels) -> int:
        return lk.layer_id - self.backbone.start_layer

    # -- envelope -------------------------------------------------------------
    def _envelope(self, stage: str, rows: int) -> bool:
        """Registry envelope (arch, TP, rows) of `stage` at `rows`."""
        from . import kernels

        if self.arch is None or not self.layers:
            return False
        if stage in PAIR_STAGES and rows not in PAIR_ROWS:
            return False
        slot, variant = slot_of(stage)
        return kernels.select(slot, self.arch, self.tp, rows, variant) is not None

    def supports(self, stage: str, rows: int) -> bool:
        if stage not in self.enabled or not self._envelope(stage, rows):
            return False
        if self.built:
            return stage in self.ok and rows in self.rows_ok.get(stage, ())
        return True

    # -- build ----------------------------------------------------------------
    def build(self) -> None:
        """Build every enabled kernel stage for the executor's step rows
        (collective; see the module docstring). Idempotent."""
        if self.built:
            return
        self.built = True
        want = [
            s
            for s in KERNEL_STAGES
            if s in self.enabled and any(self._envelope(s, r) for r in self.step_rows)
        ]
        # Every rank must build (and agree on, ``_agree``) the same stages: a rank-local
        # input that differs would size the collectives below differently and hang. Keep
        # the stages every rank wants.
        mask = torch.tensor([int(s in want) for s in KERNEL_STAGES], dtype=torch.int32)
        dist.all_reduce(mask, op=dist.ReduceOp.MIN, group=self.cpu_group)
        agreed = [s for s, m in zip(KERNEL_STAGES, mask.tolist()) if m]
        if agreed != want:
            logger.warning(
                "specialized decode: kernel stages %s are not enabled on every "
                "rank; they run stock",
                ", ".join(s for s in want if s not in agreed),
            )
            want = agreed
        if not want:
            return
        for stage in want:
            self.rows_ok[stage] = tuple(
                r for r in self.step_rows if self._envelope(stage, r)
            )
        ok = {stage: True for stage in want}
        # 1. Peer memory, in one fixed order on every rank (the transport's
        # first allocation and every new exchange are collective and agree
        # among themselves, PeerTransport.agree, so a local failure raises on
        # every rank together). The ranks then agree once more; on failure
        # every stage runs stock everywhere.
        try:
            self._allocate_peer_memory(want)
            peer_ok = True
        except Exception:
            logger.exception("specialized decode: peer memory setup failed")
            peer_ok = False
        if not self._all_ranks(peer_ok):
            self._fail_all(want, "peer memory setup failed on some rank")
            return
        # 2. Kernel modules (CuTe DSL availability; kernels compile at build).
        for stage in want:
            try:
                self._load_extension(stage)
            except Exception:
                logger.exception("specialized decode: %s kernel did not load", stage)
                ok[stage] = False
        self._agree(want, ok)
        # 3. Buffers and instances (local; agreed below).
        try:
            self._allocate_buffers()
            buffers_ok = True
        except Exception:
            logger.exception("specialized decode: buffer allocation failed")
            buffers_ok = False
        if not self._all_ranks(buffers_ok):
            self._fail_all(want, "buffer allocation failed on some rank")
            return
        for stage in want:
            if not ok[stage] or not self._dependencies_ok(stage, ok):
                ok[stage] = False
                continue
            try:
                getattr(self, f"_build_{stage}")()
            except Exception:
                logger.exception("specialized decode: %s kernel did not build", stage)
                ok[stage] = False
        self._agree(want, ok)
        for stage in want:
            if not self._dependencies_ok(stage, ok):
                ok[stage] = False
        self.ok = {stage for stage in want if ok[stage]}
        logger.info(
            "MiniMax-M3.1 specialized decode kernel stages: %s (rows %s); stock: %s",
            ", ".join(s for s in KERNEL_STAGES if s in self.ok) or "none",
            {s: self.rows_ok[s] for s in KERNEL_STAGES if s in self.ok},
            ", ".join(s for s in want if s not in self.ok) or "none failed",
        )
        self._log_row_fallbacks()

    def row_fallbacks(self) -> dict[int, tuple[str, ...]]:
        """{step rows: enabled kernel stages that run stock at those rows
        because the rows are outside their kernel's envelope (registry
        `max_rows`, or the pair's powers of two)}; only steps with some."""
        out: dict[int, tuple[str, ...]] = {}
        if self.arch is None or not self.layers:
            return out
        for rows in self.config.step_rows:
            stock = set(fallback_stages(self.arch, self.tp, rows))
            stock.update(
                s
                for s in PAIR_STAGES
                if rows not in PAIR_ROWS and rows <= PAIR_MAX_ROWS
            )
            # Above the pair's rows the o_proj runs its wide variant, if enabled; the
            # post and the MoE run stock.
            if rows > PAIR_MAX_ROWS:
                stock.update(("allreduce_norm", "moe"))
                if "o_proj_wide" not in self.enabled:
                    stock.add("o_proj")
            if rows > FRONT_MAX_ROWS and "front_wide" not in self.enabled:
                stock.add("front")
            names = tuple(s for s in KERNEL_STAGES if s in stock and s in self.enabled)
            if names:
                out[rows] = names
        return out

    def _log_row_fallbacks(self) -> None:
        fallbacks = self.row_fallbacks()
        if fallbacks:
            logger.info(
                "MiniMax-M3.1 specialized decode: stages outside their kernel's "
                "rows, run stock (per step rows): %s",
                "; ".join(f"{r}: {', '.join(n)}" for r, n in fallbacks.items()),
            )
        multi = {
            lk.sparse.multi_wave_rows[0]
            for lk in self.layers.values()
            if lk.sparse is not None and lk.sparse.multi_wave_rows
        }
        if multi:
            logger.info(
                "MiniMax-M3.1 specialized decode: the sparse decode runs in more "
                "than one wave of clusters from %d rows on",
                min(multi),
            )

    @staticmethod
    def _dependencies_ok(stage: str, ok: dict[str, bool]) -> bool:
        # The o_proj and the post only run as a pair; the MoE reads the post.
        if stage in ("o_proj", "allreduce_norm"):
            return ok.get("o_proj", False) and ok.get("allreduce_norm", False)
        if stage == "moe":
            return (
                ok.get("moe", False)
                and ok.get("o_proj", False)
                and ok.get("allreduce_norm", False)
            )
        return ok.get(stage, False)

    def _all_ranks(self, ok: bool) -> bool:
        """MIN over the TP group of one flag."""
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.cpu_group)
        return bool(flag.item())

    def _fail_all(self, want: list[str], why: str) -> None:
        """Every stage runs stock on every rank (collective failure)."""
        self.ok = set()
        logger.warning(
            "MiniMax-M3.1 specialized decode: %s; kernel stages %s run stock",
            why,
            ", ".join(want),
        )

    def _agree(self, want: list[str], ok: dict[str, bool]) -> None:
        """MIN over the TP group: a stage runs only if it built everywhere."""
        flags = torch.tensor([1 if ok[s] else 0 for s in want], dtype=torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=self.cpu_group)
        for stage, flag in zip(want, flags.tolist()):
            ok[stage] = bool(flag)

    def _allocate_peer_memory(self, want: list[str]) -> None:
        from .kernels._comm import make_transport
        from .kernels._exchange import exchange_words, push_workspace

        needs_g3 = any(s in want for s in ("front", "o_proj", "allreduce_norm", "moe"))
        front_buckets = self.front_buckets() if "front" in want else ()
        needs_front = bool(front_buckets)
        pair_rows = [
            r for r in PAIR_ROWS if r in self.rows_ok.get("allreduce_norm", ())
        ]
        needs_rx = "o_proj" in want and "allreduce_norm" in want and pair_rows
        if not (needs_g3 or needs_front or needs_rx):
            return
        from .kernels.allreduce_norm import NPART, RX_PARITIES, RouterExchangeSlot
        from .kernels.allreduce_norm import E as NUM_EXPERTS

        exch_bytes = 4 * exchange_words(self.tp, 8)
        # both tag-parity halves of the largest (8-row) router exchange slot
        rx_bytes = RX_PARITIES * 8 * self.tp * NPART * NUM_EXPERTS * 8
        layers = sorted(self.layers)
        n_rx = len(layers) * len(pair_rows) if needs_rx else 0
        n_allocs = int(needs_g3) + len(front_buckets) + n_rx
        capacity = (
            int(needs_g3) * exch_bytes
            + sum(4 * exchange_words(self.tp, b) for b in front_buckets)
            + n_rx * rx_bytes
            + n_allocs * _ARENA_ALIGN
        )
        self.transport = make_transport(
            "auto",
            group=self.device_group,
            cpu_group=self.cpu_group,
            rank=self.rank,
            world=self.tp,
            capacity=capacity,
        )
        logger.info(
            "specialized decode: %s peer transport, %d ranks, %d B arena",
            self.transport.kind,
            self.tp,
            capacity,
        )
        if needs_g3:
            push_workspace(self.transport, 8, "g3")
        for b in front_buckets:
            push_workspace(self.transport, b, "front")
        if needs_rx:
            for layer_id in layers:
                lk = self.layers[layer_id]
                for rows in pair_rows:
                    lk.rx[rows] = RouterExchangeSlot(
                        self.transport, rx_slot_key(layer_id, rows), maxt=8
                    )

    def front_buckets(self) -> tuple[int, ...]:
        """Exchange buckets the kernel front is built for: those of its rows."""
        from .kernels._exchange import bucket

        rows = self.rows_ok.get("front", ())
        return tuple(b for b in FRONT_BUCKETS if any(bucket(r) == b for r in rows))

    def max_rows_of(self, stage: str) -> int:
        """The largest step rows `stage` is built for."""
        return max(self.rows_ok.get(stage, ()), default=1)

    def _load_extension(self, stage: str) -> None:
        from .kernels import _dtypes

        if stage == "front":
            from .kernels import front

            front._ext(_dtypes.CacheFormat.FP8_E4M3, self.tp)
        elif stage == "page_topk":
            from .kernels import page_topk

            page_topk._ext()
        elif stage == "sparse_decode":
            from .kernels import sparse_decode

            sparse_decode._ext()
        elif stage == "allreduce_norm":
            from .kernels import allreduce_norm

            allreduce_norm._ext(self.tp)
        elif stage == "o_proj":
            from .kernels.o_proj import host

            host.check_cute_dsl()
        elif stage == "moe":
            from .kernels.moe import host  # type: ignore[no-redef]

            supported, why = host.is_supported()  # type: ignore[attr-defined]
            if not supported:
                raise RuntimeError(why)
        elif stage == "o_proj_wide":
            from .kernels.o_proj import host

            host.check_cute_dsl()
        elif stage == "front_wide":
            from .kernels import wide

            wide._rows_ext()
        # index_score: upstream CuTe DSL op, compiled on its first call.

    def _allocate_buffers(self) -> None:
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        # Rows of the largest step a kernel stage runs (front 16, attention trio 64,
        # pair 8): the layer buffers are row-prefix viewed per step.
        m = max((max(r) for r in self.rows_ok.values() if r), default=1)
        for lk in self.layers.values():
            attn = lk.attn
            q_size = attn.num_heads * HEAD_DIM
            lk.r_mid = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device=dev)
            lk.q_fp8 = torch.zeros(m, q_size, dtype=torch.float8_e4m3fn, device=dev)
            lk.q_bf16 = torch.zeros(m, q_size, dtype=torch.bfloat16, device=dev)
            lk.index_q = torch.zeros(
                m, attn.num_idx_heads * HEAD_DIM, dtype=torch.float8_e4m3fn, device=dev
            )
            lk.table = torch.full(
                (attn.num_kv_heads, m, 16), -1, dtype=torch.int32, device=dev
            )
            lk.out = torch.zeros(m, q_size, dtype=torch.bfloat16, device=dev)
            lk.xq = torch.zeros(m, q_size, dtype=torch.float8_e4m3fn, device=dev)
            lk.xsf = torch.zeros(128 * (q_size // 32), dtype=torch.uint8, device=dev)
            lk.sfp = torch.zeros(m, q_size // 32, dtype=torch.uint8, device=dev)
        self.row_ids = torch.arange(m, dtype=torch.int64, device=dev)

    # -- per-stage builds -----------------------------------------------------
    def _build_front(self) -> None:
        from .kernels.front import QkvFrontStage

        for lk in self.layers.values():
            layer, attn = lk.layer, lk.attn
            heads = attn.num_heads + 2 * attn.num_kv_heads + attn.num_idx_heads + 1
            # The stock MXFP8 linear may keep a [K, N] view (CuTe-DSL); the kernel reads
            # [N, K] row-major (rows q|k|v|index_q|index_k).
            qkv, sf = mxfp8_linear_nk(
                attn.qkv_proj,
                f"layer {lk.layer_id}: qkv_proj",
                shape=(heads * HEAD_DIM, HIDDEN),
            )
            rope = attn.rotary_emb
            lk.front_weights = dict(
                qkv=qkv,
                qkv_sf=sf,
                input_norm=_bf16(layer.input_layernorm.weight),
                q_norm=_bf16(attn.q_norm.weight),
                k_norm=_bf16(attn.k_norm.weight),
                iq_norm=_bf16(attn.index_q_norm.weight),
                ik_norm=_bf16(attn.index_k_norm.weight),
                # [max_pos, rot] = [cos | sin]; a BF16 copy if the cache is FP32 (the
                # stock kernel then reads FP32 values).
                cos_sin=_bf16(rope.cos_sin_cache),
            )
            kwargs = dict(
                transport=self.transport,
                nq=attn.num_heads,
                nkv=attn.num_kv_heads,
                niq=attn.num_idx_heads,
                rot=rope.rotary_dim,
                block_size=PAGE,
                kv_cache_dtype="fp8",
                indexer_kv_dtype="fp8",
                eps=layer.input_layernorm.variance_epsilon,
                head_eps=attn.q_norm.variance_epsilon,
                q_scale=float(getattr(attn, "_q_scale_float", 1.0)),
                # The step metadata is built once per step by begin_step.
                with_step_meta=False,
            )
            # One build per exchange bucket: the bucket-8 one (A operand in smem) serves
            # rows 1..8, the bucket-16 one rows 9..16.
            for b in self.front_buckets():
                lk.front_x[b] = QkvFrontStage(prepushed=False, max_rows=b, **kwargs)
            # Only a kernel MoE pushes (up to PAIR_MAX_ROWS rows), and only into the
            # next covered layer.
            if lk.after_covered and "moe" in self.enabled and 8 in lk.front_x:
                lk.front_pushed = QkvFrontStage(prepushed=True, max_rows=8, **kwargs)
            # Compile every front specialisation now (the first launch is inside a FULL
            # graph capture, where compiling is refused).
            for front in (*lk.front_x.values(), lk.front_pushed):
                if front is not None:
                    front.precompile()

    def _build_index_score(self) -> None:
        from .kernels.index_score import StockIndexScoreStage

        attn = next(iter(self.layers.values())).attn
        self.index_score = StockIndexScoreStage(
            num_index_heads=attn.indexer.impl.num_index_heads,
            max_decode_query_len=self.config.query_len,
            indexer_kv_dtype="fp8",
            max_rows=self.max_rows_of("index_score"),
        ).build()

    def _build_page_topk(self) -> None:
        from .kernels.page_topk import MAX_COLS, PageTopkStage

        impls = {
            (lk.attn.indexer.impl.init_blocks, lk.attn.indexer.impl.local_blocks)
            for lk in self.layers.values()
        }
        if len(impls) != 1:
            raise ValueError(
                f"page_topk: layers differ in (init, local) blocks {impls}"
            )
        ((init_blocks, local_blocks),) = impls
        self.page_topk = PageTopkStage(
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            max_rows=self.max_rows_of("page_topk"),
        ).build()
        dev = self.row_ids.device  # type: ignore[union-attr]
        self.identity_rows = (
            torch.arange(MAX_COLS, dtype=torch.int32, device=dev)
            .unsqueeze(0)
            .repeat(self.max_rows_of("page_topk"), 1)
        )

    def _build_sparse_decode(self) -> None:
        from .kernels.sparse_decode import ClusterSparseDecodeStage

        for lk in self.layers.values():
            attn = lk.attn
            lk.sparse = ClusterSparseDecodeStage(
                num_kv_heads=attn.num_kv_heads,
                sm_scale=attn.impl.scale,
                q_scale=float(getattr(attn, "_q_scale_float", 1.0)),
                k_scale=float(getattr(attn, "_k_scale_float", 1.0)),
                v_scale=float(getattr(attn, "_v_scale_float", 1.0)),
                kv_cache_dtype="fp8",
                max_rows=self.max_rows_of("sparse_decode"),
            ).build()

    def _build_allreduce_norm(self) -> None:
        from .kernels.allreduce_norm import PostRouteStage

        for lk in self.layers.values():
            moe = lk.block_sparse_moe
            for rows in self.rows_ok["allreduce_norm"]:
                post = PostRouteStage(
                    transport=self.transport,  # type: ignore[arg-type]
                    rx=lk.rx[rows],
                    post_norm_weight=_bf16(lk.layer.post_attention_layernorm.weight),
                    router_bias=moe.e_score_correction_bias.detach(),
                    rows=rows,
                    eps=lk.layer.post_attention_layernorm.variance_epsilon,
                )
                post.build()
                lk.post[rows] = post

    def _build_o_proj(self) -> None:
        from .kernels.o_proj import OProjRouterStage
        from .kernels.o_proj import weights as hw

        # One fold per layer: the 128x4 index of the MXFP8 o_proj input scales (for a
        # stock attention output, quantized here).
        n = torch.arange(128, device=self.row_ids.device).unsqueeze(1)  # type: ignore[union-attr]
        kb = torch.arange(hw.KO // 32, device=self.row_ids.device).unsqueeze(0)  # type: ignore[union-attr]
        self.sf_index = hw.sf128x4_index(n, kb, hw.KO // 32)
        for lk in self.layers.values():
            attn, moe = lk.attn, lk.block_sparse_moe
            o_w, o_s = mxfp8_linear_nk(
                attn.o_proj, f"layer {lk.layer_id}: o_proj", shape=(hw.H, hw.KO)
            )
            post_norm = _bf16(lk.layer.post_attention_layernorm.weight)
            router = moe.gate.weight.detach()
            sf = hw.unswizzle_128x4(o_s.reshape(-1), hw.H, hw.KO // 32)
            blob_a, blob_r, _, _ = hw.blobs(
                router, post_norm, o_w, sf, self.rank, self.tp
            )
            blobs = (blob_a.to(o_w.device), blob_r.to(o_w.device))
            for rows in self.rows_ok["o_proj"]:
                stage = OProjRouterStage(
                    transport=self.transport,  # type: ignore[arg-type]
                    rx=lk.rx[rows],
                    o_weight=o_w,
                    o_weight_scale=o_s,
                    router_weight=router,
                    post_norm_weight=post_norm,
                    residual_mid=lk.r_mid[:rows],  # type: ignore[index]
                    xq=lk.xq[:rows],  # type: ignore[index]
                    xsf=lk.xsf,
                    rows=rows,
                    blobs=blobs,
                )
                stage.build()
                lk.o_proj[rows] = stage

    def _build_moe(self) -> None:
        from .kernels.moe import FusedMoeConfig, FusedMoeStage, from_vllm_moe

        for lk in self.layers.values():
            weights = from_vllm_moe(lk.block_sparse_moe)
            # Per-layer tensors the engine derives from the weights alike for every row
            # count (the 75 MB pre-spliced FC1 scale image, the pre-tiled shared
            # expert): kept once per layer.
            shared_by: dict[tuple, Any] = {}
            for rows in self.rows_ok["moe"]:
                post = lk.post[rows]
                stage = FusedMoeStage(
                    FusedMoeConfig(rows=rows, push=lk.before_covered),
                    weights,
                    transport=self.transport,
                    handoff=post.out,
                    co_resident=(post.grid_use(),),
                )
                self._dedupe_moe_layer(stage, shared_by)
                stage.warmup()
                lk.moe[rows] = stage
                lk.ffn_out[rows] = torch.empty(
                    rows, HIDDEN, dtype=torch.bfloat16, device=post.out.normed.device
                )

    @staticmethod
    def _dedupe_moe_layer(stage, shared_by: dict[tuple, Any]) -> None:
        layer = stage.layer
        img = layer.sfa1_img
        if img is not None:
            key = ("sfa1", tuple(img.shape), img.dtype)
            kept = shared_by.setdefault(key, img)
            layer.sfa1_img = kept
        sh = layer.shared_weights
        if sh is not None:
            key = (
                "shared",
                stage.n_shared,
                tuple((tuple(t.shape), t.dtype) for t in sh),
            )
            layer.shared_weights = shared_by.setdefault(key, sh)

    def _build_front_wide(self) -> None:
        from .kernels import wide

        rows = self.rows_ok["front_wide"]
        dev = self.row_ids.device  # type: ignore[union-attr]
        lks = list(self.layers.values())
        n = None
        for i, lk in enumerate(lks):
            attn = lk.attn
            if getattr(attn.qkv_proj, "bias", None) is not None:
                raise wide.WideError("qkv_proj has a bias")
            heads = attn.num_heads + 2 * attn.num_kv_heads + attn.num_idx_heads + 1
            qkv, sf = mxfp8_linear_nk(
                attn.qkv_proj,
                f"layer {lk.layer_id}: qkv_proj",
                shape=(heads * HEAD_DIM, HIDDEN),
            )
            if i == 0:
                n = qkv.shape[0]
                xq = torch.zeros(
                    wide.MAX_ROWS, HIDDEN, dtype=torch.float8_e4m3fn, device=dev
                )
                xsf = torch.zeros(128 * (HIDDEN // 32), dtype=torch.uint8, device=dev)
                out = torch.zeros(wide.MAX_ROWS, n, dtype=torch.bfloat16, device=dev)
            if qkv.shape[0] != n:
                raise wide.WideError("layers differ in their QKV width")
            lk.wide_front = wide.WideFront(qkv, sf, xq, xsf, out)
            lk.wide_front.build(rows, compile_gemv=i == 0)
            if i == 0:
                for r in rows:
                    lk.wide_front(
                        torch.zeros(r, HIDDEN, dtype=torch.bfloat16, device=dev)
                    )
        torch.accelerator.synchronize()

    def _build_o_proj_wide(self) -> None:
        from .kernels import wide
        from .kernels.o_proj import weights as hw

        rows = self.rows_ok["o_proj_wide"]
        dev = self.row_ids.device  # type: ignore[union-attr]
        if self.sf_index is None:
            # The 128x4 index of the MXFP8 input scales (``quantize_attn_out`` of a
            # stock attention output).
            n = torch.arange(128, device=dev).unsqueeze(1)
            kb = torch.arange(hw.KO // 32, device=dev).unsqueeze(0)
            self.sf_index = hw.sf128x4_index(n, kb, hw.KO // 32)
        self.wide_y = torch.zeros(
            wide.MAX_ROWS, HIDDEN, dtype=torch.bfloat16, device=dev
        )
        for i, lk in enumerate(self.layers.values()):
            o_w, o_s = mxfp8_linear_nk(
                lk.attn.o_proj, f"layer {lk.layer_id}: o_proj", shape=(hw.H, hw.KO)
            )
            lk.wide_o_proj = wide.WideOProj(o_w, o_s, self.wide_y)
            if i == 0:
                # One compile per MMA width (the key does not depend on the weight),
                # then one eager launch each (module load).
                lk.wide_o_proj.build(rows)
                for r in rows:
                    xq = torch.zeros(r, hw.KO, dtype=torch.float8_e4m3fn, device=dev)
                    xsf = torch.full(
                        (128 * (hw.KO // 32),), 127, dtype=torch.uint8, device=dev
                    )
                    lk.wide_o_proj(xq, xsf)
        torch.accelerator.synchronize()

    # -- per step -------------------------------------------------------------
    def begin_step(self, step: DecodeStep, positions: torch.Tensor) -> None:
        """The step's sparse lengths (once per step, shared by every layer)."""
        if self._step is step:
            return
        self._step = step
        self._page_rows = {}
        self.lens = None
        if "sparse_decode" in self.ok:
            from .kernels.sparse_decode import sparse_lens

            rows = step.rows
            # Row t attends to positions[t] + 1 tokens (causal within its request's
            # draft tokens); padded rows read a valid page.
            lens = sparse_lens(positions[:rows] + 1)
            heads = next(iter(self.layers.values())).attn.num_kv_heads
            self.lens = lens.view(1, rows).expand(heads, rows).contiguous()

    def page_rows(self, attn, rows: int) -> torch.Tensor:
        """int32 [rows, max_pages]: row t's request's block-table row (logical
        KV page -> physical page), from the main attention metadata; once per
        block table and step."""
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)
        decode = attn_metadata[attn.layer_name].decode  # type: ignore[attr-defined]
        block_table = decode.block_table
        key = block_table.data_ptr()
        cached = self._page_rows.get(key)
        if cached is not None and cached.shape[0] == rows:
            return cached
        query_len = _indexer_metadata(attn).decode.decode_query_len
        assert self.row_ids is not None
        req = (self.row_ids[:rows] // query_len).clamp_(max=block_table.shape[0] - 1)
        page_rows = block_table.index_select(0, req).to(torch.int32)
        self._page_rows[key] = page_rows
        return page_rows

    def quantize_attn_out(self, lk: LayerKernels, attn_out: torch.Tensor) -> None:
        """MXFP8 of a BF16 attention output into the o_proj's bound input
        buffers (a stock sparse decode; the kernel one writes them itself).
        The reference quantizer: UE8M0 = roundup(amax / 448), E4M3 RN."""
        from .kernels.o_proj import weights as hw

        rows = attn_out.shape[0]
        q, code = hw.mxfp8_quant_rows(attn_out)
        assert lk.xq is not None and lk.xsf is not None and self.sf_index is not None
        lk.xq[:rows].copy_(q)
        lk.xsf.index_put_((self.sf_index[:rows].reshape(-1),), code.reshape(-1))

    def launch_o_proj_post(
        self,
        lk: LayerKernels,
        attn_out: torch.Tensor,
        residual: torch.Tensor,
    ):
        """The o_proj + post pair of `lk` on `attn_out` and the residual
        entering the post-attention boundary; returns the post's
        `MoeHandoff`."""
        rows = residual.shape[0]
        assert lk.r_mid is not None and lk.out is not None
        r_mid = lk.r_mid[:rows]
        if residual.data_ptr() != r_mid.data_ptr():
            # A stock front: the router CTAs and the post read the bound buffer.
            r_mid.copy_(residual)
        if attn_out.data_ptr() != lk.out.data_ptr():
            # A stock sparse decode: no MXFP8 copy was written.
            self.quantize_attn_out(lk, attn_out)
        lk.o_proj[rows]()
        return lk.post[rows](r_mid)


class _KernelStage:
    """What the adapters share: the registry envelope and the build."""

    kernel = ""  # registry stage name
    max_rows: int | None = KERNEL_MAX_ROWS
    cache_dtypes: frozenset[CacheDtypes] | None = FP8_CACHES

    def __init__(self, ctx: KernelContext) -> None:
        self.ctx = ctx

    def supports(self, rows: int) -> bool:
        return self.ctx.supports(self.kernel, rows)

    def warm_up(self, rows) -> None:
        self.ctx.build()

    def begin_step(self, step: DecodeStep, positions: torch.Tensor) -> None:
        self.ctx.begin_step(step, positions)


def _after_deferring_moe(backbone: MiniMaxM3Model, layer_id: int, rows: int) -> bool:
    """Whether the stock layer before `layer_id` is a MoE that leaves
    `rows` rows unfinalized (host state only)."""
    if layer_id <= backbone.start_layer:
        return False
    prev = backbone.layers[layer_id - 1]
    return bool(
        getattr(prev, "is_moe_layer", False)
        and prev.block_sparse_moe.defers_finalize(rows)
    )


class KernelFront(_KernelStage, FrontStage):
    """`front`: one launch per sparse layer for the input boundary (AR land
    of the previous FFN's partials, residual add, input norm, MXFP8) + QKV /
    index GEMV + norms + RoPE + FP8 KV / index-K insert."""

    kernel = "front"

    def __init__(self, ctx: KernelContext, stock: StockFront) -> None:
        super().__init__(ctx)
        self.stock = stock

    def run(self, attn, hidden: LayerInput, positions: torch.Tensor) -> FrontOutput:
        # Only layers whose input it does not fuse (dense or uncovered).
        return self.stock.run(attn, hidden, positions)

    def fuses_input(self, layer: MiniMaxM3DecoderLayer) -> bool:
        lk = self.ctx.by_layer(layer)
        return lk is not None and bool(lk.front_x)

    def polls_push(self, layer: MiniMaxM3DecoderLayer) -> bool:
        lk = self.ctx.by_layer(layer)
        return lk is not None and lk.front_pushed is not None

    def fuses_finalize(self, layer: MiniMaxM3DecoderLayer) -> bool:
        return self.fuses_input(layer)

    def run_fused(
        self,
        layer: MiniMaxM3DecoderLayer,
        pending: PendingFFN,
        residual: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[FrontOutput, torch.Tensor]:
        from .kernels._exchange import bucket
        from .kernels.front import FinalizeIn

        lk = self.ctx.by_layer(layer)
        assert lk is not None and lk.front_x
        attn = lk.attn
        rows = positions.shape[0]
        finalize = x2 = None
        # The stock input boundary after a MoE that defers its finalize rounds the
        # residual once (finalize + all-reduce + norm); FlashInfer's fused all-reduce +
        # norm (after a dense MLP or a finalized MoE) twice.
        single_round = _after_deferring_moe(self.ctx.backbone, lk.layer_id, rows)
        if isinstance(pending, MoEOutput) and pending.routed_scaling_factor != 1.0:
            pending = _finalize_moe_output(pending, rows)
        if isinstance(pending, PushedFFN):
            front, x = lk.front_pushed, None
            assert front is not None, "a push into a front that does not poll it"
            assert rows <= front.max_rows, "a push of more rows than the front polls"
        elif isinstance(pending, MoEOutput):
            # A deferring stock MoE: the front runs its top-k reduction (and the shared
            # add) while landing the partial.
            front, x = lk.front_x.get(bucket(rows)), None
            r = pending.routed
            if front is None or not isinstance(r, UnfinalizedMoEOutput):
                raise RuntimeError(
                    f"kernel front: cannot finalize a {type(r).__name__} at {rows} rows"
                )
            perm = r.expanded_idx_to_permuted_idx
            topk = perm.numel() // rows
            finalize = FinalizeIn(
                fc2=r.gemm2_permuted,
                perm=perm.reshape(-1).contiguous(),
                scales=r.expert_weights.reshape(-1).contiguous(),
                topk=topk,
            )
            if pending.shared_output is not None:
                x2 = pending.shared_output.contiguous()
        elif isinstance(pending, torch.Tensor):
            front, x = lk.front_x.get(bucket(rows)), pending.contiguous()
            if front is None:
                raise RuntimeError(
                    f"kernel front: no build for {rows} rows (buckets "
                    f"{sorted(lk.front_x)})"
                )
            if x.dtype != torch.bfloat16 or tuple(x.shape) != (rows, HIDDEN):
                raise ValueError(
                    f"kernel front: partial {x.dtype} {tuple(x.shape)}, expected "
                    f"BF16 [{rows}, {HIDDEN}]"
                )
        else:
            raise TypeError(
                f"kernel front: the previous FFN's output is a {type(pending)}"
            )
        slot_mappings = get_forward_context().slot_mapping
        assert isinstance(slot_mappings, dict)
        index_cache = attn.indexer.index_cache
        assert lk.q_fp8 is not None and lk.q_bf16 is not None
        assert lk.index_q is not None and lk.r_mid is not None
        # The stock sparse decode reads E4M3 q on CUTLASS, else BF16.
        q_out = None if attn.impl.use_cutlass_decode else lk.q_bf16[:rows]
        w = lk.front_weights
        front(
            rows,
            residual=residual.contiguous(),
            norm_w=w["input_norm"],
            out_residual=lk.r_mid[:rows],
            w=w["qkv"],
            wsf=w["qkv_sf"],
            q_w=w["q_norm"],
            k_w=w["k_norm"],
            iq_w=w["iq_norm"],
            ik_w=w["ik_norm"],
            cos_sin=w["cos_sin"],
            positions=positions,
            slot_mapping=slot_mappings[attn.layer_name][:rows],
            index_slot_mapping=slot_mappings[index_cache.prefix][:rows],
            kv_cache=attn.kv_cache,
            index_cache=index_cache.kv_cache.view(-1, attn.idx_head_dim),
            q_fp8=lk.q_fp8[:rows],
            index_q=lk.index_q[:rows],
            x=x,
            x2=x2,
            finalize=finalize,
            q_out=q_out,
            single_round=single_round,
        )
        q = lk.q_fp8[:rows] if q_out is None else q_out
        return FrontOutput(q=q, index_q=lk.index_q[:rows]), lk.r_mid[:rows]


class KernelIndexScore(_KernelStage, IndexScoreStage):
    """`index_score`: upstream's CuTe DSL decode scorer through the kernel
    package's fp8 wrapper (the same op the stock stage picks at this tile)."""

    kernel = "index_score"

    def __init__(self, ctx: KernelContext, stock: StockIndexScore) -> None:
        super().__init__(ctx)
        self.stock = stock

    def run(self, attn, index_q: torch.Tensor) -> torch.Tensor:
        impl = attn.indexer.impl
        md = _indexer_metadata(attn)
        d = md.decode
        nd = md.num_decode_tokens
        scorer = self.ctx.index_score
        if (
            scorer is None
            or d is None
            or nd != md.num_actual_tokens
            or d.max_decode_query_len != scorer.max_decode_query_len
            or impl.num_index_heads != scorer.num_index_heads
            or index_q.dtype != torch.float8_e4m3fn
        ):
            return self.stock.run(attn, index_q)
        unified_scores = md.unified_scores
        assert unified_scores is not None
        scorer(
            index_q[:nd].view(-1, impl.num_index_heads, impl.index_head_dim),
            impl.index_cache.kv_cache,
            d.block_table,
            d.seq_lens,
            unified_scores[:nd].transpose(0, 1),
            decode_query_len=d.decode_query_len,
        )
        return unified_scores


class KernelPageTopKLogical(_KernelStage, PageTopKStage):
    """`page_topk` into the stock `topk_indices_buffer` (logical block
    ids, identity page rows): composes with any sparse decode."""

    kernel = "page_topk"

    def __init__(self, ctx: KernelContext, stock: StockPageTopK) -> None:
        super().__init__(ctx)
        self.stock = stock

    def select_pages(
        self, attn, scores: torch.Tensor, table: torch.Tensor, physical: bool
    ) -> None:
        """Top-16 pages of `scores` into `table` [Hidx, rows, >= 16]:
        physical page ids (`page_rows`) or logical block ids."""
        md = _indexer_metadata(attn)
        rows = table.shape[1]
        assert self.ctx.page_topk is not None
        n_valid = md.topk_num_valid_pages
        assert n_valid is not None
        if physical:
            page_rows = self.ctx.page_rows(attn, rows)
            # Every row's valid pages are within its block-table row.
            cols = min(scores.shape[2], page_rows.shape[1])
            self.ctx.page_topk(
                scores[:rows, :, :cols], n_valid[:rows], page_rows, table
            )
        else:
            assert self.ctx.identity_rows is not None
            self.ctx.page_topk(
                scores[:rows], n_valid[:rows], self.ctx.identity_rows[:rows], table
            )

    def run(self, attn, scores: torch.Tensor) -> torch.Tensor:
        md = _indexer_metadata(attn)
        rows = md.num_actual_tokens
        buf = attn.indexer.impl.topk_indices_buffer
        assert buf is not None
        self.select_pages(attn, scores, buf[:rows].transpose(0, 1), physical=False)
        return buf[:rows]


class KernelPageTopK(KernelPageTopKLogical):
    """`page_topk` writing physical page ids into the layer's own table,
    which only `KernelSparseDecode` reads (it skips the logical -> physical
    gather)."""

    def compatible(self, step: StepStages) -> bool:
        return isinstance(step.sparse_decode, KernelSparseDecode)

    def run(self, attn, scores: torch.Tensor) -> torch.Tensor:
        lk = self.ctx.by_attn(attn)
        if lk is None or lk.sparse is None:
            return super().run(attn, scores)
        assert lk.table is not None
        rows = _indexer_metadata(attn).num_actual_tokens
        table = lk.table[:, :rows]
        self.select_pages(attn, scores, table, physical=True)
        return table


class KernelSparseDecode(_KernelStage, SparseDecodeStage):
    """`sparse_decode`: the 16-CTA cluster FP8 decode over the selected
    pages, BF16 output plus the MXFP8 copy the kernel o_proj reads."""

    kernel = "sparse_decode"

    def __init__(self, ctx: KernelContext, stock: StockSparseDecode) -> None:
        super().__init__(ctx)
        self.stock = stock

    def run(self, attn, q: torch.Tensor, topk: torch.Tensor) -> torch.Tensor:
        lk = self.ctx.by_attn(attn)
        if lk is None or lk.sparse is None:
            return self.stock.run(attn, q, topk)
        rows = q.shape[0]
        assert lk.table is not None and lk.out is not None
        # q from the front (two or more launches back) may be read before the kernel's
        # grid wait; a q converted here is the previous launch's output.
        q_prewait = q.dtype == torch.float8_e4m3fn
        if q.dtype != torch.float8_e4m3fn:
            # A front that wrote BF16 q (the stock front for a Triton stock decode):
            # E4M3 under the layer's q scale, as the stock front writes it for CUTLASS.
            q_scale = float(getattr(attn, "_q_scale_float", 1.0))
            q = (q.float() / q_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        tables = self.tables(attn, lk, topk, rows)
        if self.ctx.lens is None or self.ctx.lens.shape[1] != rows:
            raise RuntimeError("kernel sparse decode: no step lengths for these rows")
        from .kernels.sparse_decode import split_kv

        k, v = split_kv(attn.kv_cache)
        lk.sparse(
            q.view(rows, attn.num_heads, HEAD_DIM),
            k,
            v,
            tables,
            self.ctx.lens,
            lk.out[:rows],
            lk.xq[:rows],  # type: ignore[index]
            lk.xsf,
            lk.sfp[:rows],  # type: ignore[index]
            q_prewait=q_prewait,
        )
        return lk.out[:rows]

    def tables(
        self, attn, lk: LayerKernels, topk: torch.Tensor, rows: int
    ) -> torch.Tensor:
        """[Hkv, rows, 16] physical pages: `KernelPageTopK`'s table as is,
        or the stock top-k buffer's logical blocks through `page_rows`."""
        assert lk.table is not None
        if topk.data_ptr() == lk.table.data_ptr():
            if topk.shape[1] != rows:
                raise RuntimeError(
                    f"kernel sparse decode: top-k table of {topk.shape[1]} rows "
                    f"for {rows} query rows"
                )
            return topk
        buf = attn.topk_indices_buffer
        assert buf is not None and topk.data_ptr() == buf.data_ptr()
        ids = topk[:rows].transpose(0, 1)  # [Hkv, rows, 16] logical, -1 = none
        page_rows = self.ctx.page_rows(attn, rows)
        heads = ids.shape[0]
        phys = torch.gather(
            page_rows.unsqueeze(0).expand(heads, -1, -1), 2, ids.clamp(min=0).long()
        )
        return torch.where(ids >= 0, phys, torch.full_like(phys, -1)).contiguous()


class KernelOProj(_KernelStage, OProjStage):
    """`o_proj` (MXFP8 GEMV + router CTAs), launched by its partner
    `KernelAllReduceNorm` (they pair one call to one call)."""

    kernel = "o_proj"

    def __init__(self, ctx: KernelContext, stock: StockOProj) -> None:
        super().__init__(ctx)
        self.stock = stock

    def compatible(self, step: StepStages) -> bool:
        return isinstance(step.allreduce_norm, KernelAllReduceNorm)

    def run(self, attn, attn_out: torch.Tensor):
        lk = self.ctx.by_attn(attn)
        if lk is None or attn_out.shape[0] not in lk.o_proj:
            return self.stock.run(attn, attn_out)
        return OProjPending(lk, attn_out)


class KernelAllReduceNorm(_KernelStage, AllReduceNormStage):
    """`allreduce_norm.post_attention` of covered layers: the o_proj + post
    pair (AR land + residual + norm + NVFP4 FC1 input + top-4 routes). The
    input boundary is fused into `KernelFront` (or runs stock), and the
    final norm runs stock."""

    kernel = "allreduce_norm"

    def __init__(self, ctx: KernelContext, stock: StockAllReduceNorm) -> None:
        super().__init__(ctx)
        self.stock = stock

    def compatible(self, step: StepStages) -> bool:
        return isinstance(step.o_proj, KernelOProj)

    def input(self, layer, pending: PendingFFN, residual) -> BoundaryOutput:
        return self.stock.input(layer, pending, residual)

    def post_attention(self, layer, partial, residual: torch.Tensor) -> BoundaryOutput:
        if not isinstance(partial, OProjPending):
            return self.stock.post_attention(layer, partial, residual)
        handoff = self.ctx.launch_o_proj_post(partial.lk, partial.attn_out, residual)
        return BoundaryOutput(
            hidden=handoff.normed, residual=handoff.residual_out, router_logits=None
        )

    def final(self, model, pending: PendingFFN, residual) -> BoundaryOutput:
        return self.stock.final(model, pending, residual)


class KernelMoE(_KernelStage, MoEStage):
    """`moe`: the fused NVFP4 routed + MXFP8 shared-expert engine on the
    post's hand-off; pushes into the next layer's kernel front."""

    kernel = "moe"

    def __init__(self, ctx: KernelContext, stock: StockMoE) -> None:
        super().__init__(ctx)
        self.stock = stock

    def compatible(self, step: StepStages) -> bool:
        return isinstance(step.allreduce_norm, KernelAllReduceNorm)

    def can_push(self, moe: MiniMaxM3MoE, rows: int) -> bool:
        lk = self.ctx.by_moe(moe)
        return (
            lk is not None
            and lk.before_covered
            and rows in lk.moe
            and lk.moe[rows].push_ws is not None
        )

    def run(
        self,
        moe,
        hidden: torch.Tensor,
        router_logits: torch.Tensor | None,
        defer_finalize: bool,
        push: bool = False,
    ) -> PendingFFN:
        lk = self.ctx.by_moe(moe)
        rows = hidden.shape[0]
        engine = None if lk is None else lk.moe.get(rows)
        if engine is None:
            return self.stock.run(moe, hidden, router_logits, defer_finalize)
        if hidden.data_ptr() != engine.handoff.normed.data_ptr():
            # Its routes and NVFP4 input come from the paired post only.
            raise RuntimeError(
                f"kernel MoE of layer {lk.layer_id}: its input is not the kernel "  # type: ignore[union-attr]
                "post's hand-off (a stock allreduce_norm ran before it)"
            )
        out = lk.ffn_out[rows]  # type: ignore[union-attr]
        engine(out=out, push=push)
        return PushedFFN(out) if push else out


class KernelFrontWide(_KernelStage, FrontStage):
    """`front` above the kernel front's rows: the stock input boundary, then
    the MXFP8 quantize + GEMV QKV / index projection and the stock q/k norm +
    RoPE + cache insert op."""

    kernel = "front_wide"

    def __init__(self, ctx: KernelContext, stock: StockFront) -> None:
        super().__init__(ctx)
        self.stock = stock

    def run(self, attn, hidden: LayerInput, positions: torch.Tensor) -> FrontOutput:
        lk = self.ctx.by_attn(attn)
        if lk is None or lk.wide_front is None:
            return self.stock.run(attn, hidden, positions)
        if isinstance(hidden, QuantizedActivation):
            # The stock input boundary after a deferring MoE (finalize + all-reduce +
            # norm) already wrote the MXFP8 rows the GEMV reads.
            q = hidden.data
            rows = hidden.orig_shape[0]
            if (
                hidden.quant_key != kMxfp8Dynamic
                or q.dtype != torch.float8_e4m3fn
                or tuple(q.shape) != (rows, HIDDEN)
                or not q.is_contiguous()
                or hidden.scale.numel() < 128 * (HIDDEN // 32)
            ):
                return self.stock.run(attn, hidden, positions)
            qkv = lk.wide_front.quantized(q, hidden.scale.view(torch.uint8).reshape(-1))
            return self.stock.run_qkv(attn, qkv, rows, hidden.orig_dtype, positions)
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.dtype != torch.bfloat16
            or not hidden.is_contiguous()
        ):
            return self.stock.run(attn, hidden, positions)
        qkv = lk.wide_front(hidden)
        return self.stock.run_qkv(attn, qkv, hidden.shape[0], hidden.dtype, positions)


class KernelOProjWide(_KernelStage, OProjStage):
    """`o_proj` above the pair's rows: the MXFP8 GEMV (rows = the MMA's N)
    on the kernel sparse decode's MXFP8 output; a BF16 partial, like stock."""

    kernel = "o_proj_wide"

    def __init__(self, ctx: KernelContext, stock: StockOProj) -> None:
        super().__init__(ctx)
        self.stock = stock

    def run(self, attn, attn_out: torch.Tensor):
        lk = self.ctx.by_attn(attn)
        if (
            lk is None
            or lk.wide_o_proj is None
            or lk.out is None
            or attn_out.data_ptr() != lk.out.data_ptr()
        ):
            # A stock sparse decode: no MXFP8 copy of its output.
            return self.stock.run(attn, attn_out)
        rows = attn_out.shape[0]
        assert lk.xq is not None and lk.xsf is not None
        return lk.wide_o_proj(lk.xq[:rows], lk.xsf)


def kernel_stage_lists(
    backbone: MiniMaxM3Model,
    config: DecodeConfig,
    stock: dict[str, Any],
) -> dict[str, list]:
    """The kernel implementations of every slot (to put ahead of the stock
    ones). `stock`: the stock stage objects by slot (the adapters delegate
    to them)."""
    enabled = frozenset(KERNEL_STAGES)
    ctx = KernelContext(backbone, config, enabled)
    lists: dict[str, list] = {}
    if "front" in enabled:
        lists["front"] = [KernelFront(ctx, stock["front"])]
    if "index_score" in enabled:
        lists["index_score"] = [KernelIndexScore(ctx, stock["index_score"])]
    if "page_topk" in enabled:
        page_topk: list = [KernelPageTopKLogical(ctx, stock["page_topk"])]
        if "sparse_decode" in enabled:
            page_topk.insert(0, KernelPageTopK(ctx, stock["page_topk"]))
        lists["page_topk"] = page_topk
    if "sparse_decode" in enabled:
        lists["sparse_decode"] = [KernelSparseDecode(ctx, stock["sparse_decode"])]
    # The pair runs together or not at all.
    if {"o_proj", "allreduce_norm"} <= enabled:
        lists["o_proj"] = [KernelOProj(ctx, stock["o_proj"])]
        lists["allreduce_norm"] = [KernelAllReduceNorm(ctx, stock["allreduce_norm"])]
        if "moe" in enabled:
            lists["moe"] = [KernelMoE(ctx, stock["moe"])]
    # The wide variants behind them (rows above the pair's / the front's).
    if "front_wide" in enabled:
        lists.setdefault("front", []).append(KernelFrontWide(ctx, stock["front"]))
    if "o_proj_wide" in enabled:
        lists.setdefault("o_proj", []).append(KernelOProjWide(ctx, stock["o_proj"]))
    return lists


__all__ = [
    "FP8_CACHES",
    "KERNEL_MAX_ROWS",
    "KERNEL_STAGES",
    "KernelAllReduceNorm",
    "KernelContext",
    "KernelFront",
    "KernelIndexScore",
    "KernelMoE",
    "KernelOProj",
    "KernelOProjWide",
    "KernelFrontWide",
    "KernelPageTopK",
    "KernelPageTopKLogical",
    "KernelSparseDecode",
    "LayerKernels",
    "OProjPending",
    "kernel_stage_lists",
    "rx_slot_key",
]
