# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step decode metadata of the die-local indexer.

Every decode step, each indexer cache group needs its requests' pages split
by locality domain and both domains' DeepGEMM schedules (`build.cu` describes
the outputs and the rows of speculative decoding). The metadata builder only
records the inputs (its persistent buffers); the forward's first localized
scan launches one kernel that builds every recorded group, so FULL CUDA graphs
capture it and nothing runs between graphs.
"""

import ctypes
import functools
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.indexer_localize.kernels import CudaKernel, get_kernel
from vllm.v1.attention.ops.indexer_localize.placement import (
    NUM_DOMAINS,
    DomainResources,
    get_domain_resources,
    page_domain_plan,
    page_domain_table,
    unsupported_config_reason,
)

logger = init_logger(__name__)

# Build kernel geometry (build.cu): 256 threads x 4 pages per split block.
_THREADS = 256
_BUILD_CHUNK = 1024
_RESET_ROWS = 8
_MAX_GROUPS = 4


# ---------------------------------------------------------------------------
# Build kernel launcher
# ---------------------------------------------------------------------------
# The fields of build.cu's `Group`: pointers, 64-bit and 32-bit integers.
_PTRS = ["bt", "lens", "dom", "indices", "lbt0", "lbt1", "inv0", "inv1", "jmap"]
_PTRS += ["llen0", "llen1", "sched0", "sched1", "flags", "state"]
_I64 = ["bt_stride", "out_stride", "jmap_stride", "dom_q0"]
_I32 = ["R", "width", "cap", "num_rows", "nch", "split_begin", "reset_begin"]
_I32 += ["reset_end", "dom_k"]


class _Group(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_uint64) for n in _PTRS]
        + [(n, ctypes.c_int64) for n in _I64]
        + [(n, ctypes.c_int32) for n in _I32]
    )


class _Params(ctypes.Structure):
    _fields_ = [("g", _Group * _MAX_GROUPS)] + [
        (n, ctypes.c_int32) for n in ("num_groups", "sms0", "sms1")
    ]


assert ctypes.sizeof(_Group) == 192 and ctypes.sizeof(_Params) == 784


@dataclass
class _GroupArgs:
    """One group's tensors for a launch: the inputs (block table, lengths,
    request indices, the page-to-domain table), the persistent outputs and the
    group's zero-initialized flags and epoch."""

    bt: torch.Tensor
    lens: torch.Tensor
    dom: torch.Tensor
    indices: torch.Tensor
    lbt: list[torch.Tensor]
    inv: list[torch.Tensor]
    jmap: torch.Tensor
    llen: list[torch.Tensor]
    sched: list[torch.Tensor]
    flags: torch.Tensor
    state: torch.Tensor
    R: int
    width: int
    cap: int
    num_rows: int
    # dom_k > 0: domain(b) = ((dom_q0 + b) // dom_k) % 2, no table reads.
    dom_q0: int = 0
    dom_k: int = 0


def _num_chunks(width: int) -> int:
    return max(1, cdiv(width, _BUILD_CHUNK))


def _pack(groups: list[_GroupArgs], sms: list[int]) -> tuple[_Params, int, int]:
    """The parameter block, grid size and dynamic shared memory of a launch."""
    p = _Params()
    blocks = 0
    for i, g in enumerate(groups):
        p.g[i].split_begin = blocks  # the schedule block; split blocks follow
        blocks += 1 + g.num_rows * _num_chunks(g.width)
    for i, g in enumerate(groups):
        G = p.g[i]
        G.reset_begin = blocks
        blocks += cdiv(max(0, g.cap - g.num_rows), _RESET_ROWS)
        G.reset_end = blocks
        tensors = [g.bt, g.lens, g.dom, g.indices, *g.lbt, *g.inv, g.jmap]
        tensors += [*g.llen, *g.sched, g.flags, g.state]
        for name, t in zip(_PTRS, tensors):
            setattr(G, name, t.data_ptr())
        G.bt_stride, G.out_stride = g.bt.stride(0), g.lbt[0].stride(0)
        G.jmap_stride = g.jmap.stride(0)
        G.R, G.width, G.cap, G.num_rows = g.R, g.width, g.cap, g.num_rows
        G.nch, G.dom_q0, G.dom_k = _num_chunks(g.width), g.dom_q0, g.dom_k
    p.num_groups = len(groups)
    p.sms0, p.sms1 = sms
    # The schedule block keeps 16 bytes per row in shared memory.
    return p, blocks, 16 * max(g.num_rows for g in groups)


class _BuildKernel:
    """Launches `build.cu` for up to `_MAX_GROUPS` groups at once."""

    def __init__(self, kernel: CudaKernel):
        self._kernel = kernel
        # Packed parameters per list of _GroupArgs, checked by identity: the
        # localizers keep one _GroupArgs per set of tensors.
        self._packed: dict[tuple, tuple] = {}

    def launch(self, groups: list[_GroupArgs], sms: list[int]) -> None:
        key = tuple(id(g) for g in groups) + tuple(sms)
        entry = self._packed.get(key)
        if entry is None or any(r() is not g for r, g in zip(entry[0], groups)):
            params, grid, smem = _pack(groups, sms)
            entry = ([weakref.ref(g) for g in groups], params, grid, smem)
            self._packed[key] = entry
            if len(self._packed) > 1024:
                self._packed = {
                    k: e
                    for k, e in self._packed.items()
                    if all(r() is not None for r in e[0])
                }
        _, params, grid, smem = entry
        self._kernel.launch(grid, _THREADS, smem, [params])


@functools.cache
def _build_kernel(device_index: int) -> _BuildKernel | None:
    kernel = get_kernel("build.cu", "indexer_localize_build", (), device_index)
    return None if kernel is None else _BuildKernel(kernel)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def topk_ctas_per_row(num_rows: int) -> int:
    """CTAs per row of the top-k the stock selector runs on `num_rows` decode
    rows (one CTA fills an SM): `cooperative_topk`'s cluster of 16, 8, 4 or 2
    CTAs up to 4, 8, 33 or 64 rows, then `persistent_topk`'s one CTA per row."""
    for max_rows, ctas in ((4, 16), (8, 8), (33, 4), (64, 2)):
        if num_rows <= max_rows:
            return ctas
    return 1


_CAPACITY: dict[int, dict[int, int] | None] = {}


def topk_cluster_capacity(res: DomainResources) -> dict[int, int] | None:
    """Per top-k cluster size (16, 8, 4, 2 or 1 CTAs), how many such clusters
    every locality domain's SMs hold at once: CUDA's occupancy query for the
    top-k's CTA shape (`topk_shape.cu`) on each domain's green-context stream,
    the smallest over the domains. Clusters cannot span GPCs, so this can be
    less than the SMs over the cluster size: 23 clusters of 4 CTAs on a VR200
    domain's 100 SMs. None when the query is unavailable. Once per device."""
    if res.device not in _CAPACITY:
        capacity = None
        kernel = get_kernel("topk_shape.cu", "topk_cta_shape", (), res.device)
        if kernel is not None:
            # Dynamic shared memory up to the opt-in limit: one CTA per SM, as
            # the top-k's own shared memory gives it.
            props = torch.cuda.get_device_properties(res.device)
            smem = props.shared_memory_per_block_optin
            try:
                capacity = {
                    cluster: min(
                        kernel.max_active_clusters(cluster, 1024, smem, s, res.device)
                        for s in res.streams
                    )
                    for cluster in (16, 8, 4, 2, 1)
                }
            except Exception as e:  # noqa: BLE001 - the caller falls back
                logger.warning(
                    "Die-local indexer: no cluster occupancy query (%s: %s)",
                    type(e).__name__,
                    e,
                )
        _CAPACITY[res.device] = capacity
    return _CAPACITY[res.device]


def use_die_split(num_rows: int, capacity: Mapping[int, int]) -> bool:
    """Whether a decode batch of `num_rows` rows (one per query token) takes
    the die-local indexer; otherwise the whole batch takes the stock one.

    Yes when each domain's top-k fits in one round of CTAs: at most as many
    rows as every domain's SMs hold clusters of the top-k's size at once
    (`capacity`, from `topk_cluster_capacity`). A second round costs about
    one CTA's fixed time, 7-9 us on VR200, more than the die split saves
    there. A FULL CUDA graph fixes the path per captured batch size, so the
    rule depends only on the row count, never on context lengths.
    """
    return num_rows <= capacity.get(topk_ctas_per_row(num_rows), 0)


def _row_ranges(rows: list[int]) -> str:
    """`[1, 2, 3, 7, 8]` -> `1-3, 7-8`."""
    runs: list[list[int]] = []
    for n in rows:
        if runs and runs[-1][1] == n - 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs)


# ---------------------------------------------------------------------------
# Localizer
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class _PendingBuild:
    """A group's build, recorded by its metadata builder for the forward's
    first localized scan."""

    args: _GroupArgs
    done: bool = False


# Builds recorded since the last launch, per device and localizer.
_PENDING: dict[int, dict[int, _PendingBuild]] = {}


@dataclass
class LocalizedDecodeMetadata:
    """Per-step, per-domain decode scan inputs (views of persistent buffers),
    shared by every indexer layer of one cache group."""

    page_rows: int
    capacity: int
    block_tables: list[torch.Tensor]  # [T, W] int32 per domain
    seq_lens: list[torch.Tensor]  # [T, 1] int32 per domain
    schedules: list[torch.Tensor]  # DeepGEMM schedule per domain
    # Per row, of its request's pages (the same for every row of a request):
    join_map: torch.Tensor  # [T, W] int32: (local_col << 1) | domain
    inv_maps: list[torch.Tensor]  # [T, W] int32 per domain: local -> page
    page_stride: int  # bytes between pages of the indexer cache view
    resources: DomainResources = field(repr=False)
    pending: _PendingBuild | None = field(default=None, repr=False)


def ensure_built(loc: LocalizedDecodeMetadata) -> None:
    """Launch every build recorded since the last launch, on this stream.

    Called by each localized scan before its fork; only the forward's first
    call launches (inside the CUDA graph while one is captured).
    """
    pending = loc.pending
    if pending is None or pending.done:
        return
    device = loc.resources.device
    todo = [p for p in _PENDING.get(device, {}).values() if not p.done]
    if pending not in todo:
        todo.append(pending)
    kernel = _build_kernel(device)
    assert kernel is not None
    for i in range(0, len(todo), _MAX_GROUPS):
        kernel.launch([p.args for p in todo[i : i + _MAX_GROUPS]], loc.resources.sms)
    for p in todo:
        p.done = True


class IndexerDecodeLocalizer:
    """Builds `LocalizedDecodeMetadata` for one indexer cache group each
    decode step. Owned by the group's indexer metadata builder."""

    def __init__(
        self,
        layer_names: list[str],
        static_forward_context: dict[str, Any],
        device: torch.device,
        max_rows: int,
        width: int,
        page_rows: int,
    ):
        self.layer_names = list(layer_names)
        self.ctx = static_forward_context
        self.device = device
        self.width = width
        self.page_rows = page_rows
        self.res = get_domain_resources(device)
        # Buffers for the largest batch the policy sends to the die split.
        accepted = []
        self._capacity: dict[int, int] = {}
        if self.res is not None:
            self._capacity = topk_cluster_capacity(self.res) or {}
            accepted = [
                n for n in range(1, max_rows + 1) if use_die_split(n, self._capacity)
            ]
            logger.info_once(
                "Die-local sparse indexer: decode batches of %s rows take the die "
                "split; other batches the stock indexer.",
                _row_ranges(accepted) or "no",
            )
        self.max_rows = max(accepted, default=0)
        self.disabled = (
            self.res is None
            or _build_kernel(self.res.device) is None
            or self.max_rows == 0
        )
        self._key: Any = None
        self._dom_table: torch.Tensor | None = None
        self._dom_plan = (0, 0)
        self.page_stride = 0
        self._bufs: dict[str, Any] | None = None
        # The build kernel's copy of the page-to-domain table, and its source.
        self._dom: torch.Tensor | None = None
        self._dom_src: torch.Tensor | None = None
        self._args: dict[tuple, tuple] = {}

    def _disable(self, reason: str) -> None:
        self.disabled = True
        logger.warning(
            "Die-local sparse indexer disabled for %s.. (%s); using the stock "
            "indexer for it.",
            self.layer_names[0],
            reason,
        )

    def _layer_cache(self, name: str) -> torch.Tensor | None:
        kv = getattr(self.ctx.get(name), "kv_cache", None)
        if not isinstance(kv, torch.Tensor) or kv.numel() == 0:
            return None
        if kv.dim() == 4:
            kv = kv.squeeze(1) if kv.shape[1] == 1 else kv.squeeze(-2)
        return kv

    def _resolve(self) -> bool:
        """Page-to-domain table of the bound indexer caches, recomputed when
        they change (the builder first runs on the profiling-time cache)."""
        kv0 = self._layer_cache(self.layer_names[0]) if self.layer_names else None
        if kv0 is None:
            return False
        key = (kv0.data_ptr(), tuple(kv0.shape), tuple(kv0.stride()))
        if key == self._key:
            return self._dom_table is not None
        self._key = key
        self._dom_table = None
        tables = []
        for name in self.layer_names:
            kv = self._layer_cache(name)
            # None: not in a localized allocation (yet).
            table = None if kv is None else page_domain_table(kv)
            if kv is None or table is None:
                return False
            if kv.shape[1] != self.page_rows:
                self._disable(f"{kv.shape[1]} page rows, expected {self.page_rows}")
                return False
            tables.append((table, kv.stride(0) * kv.element_size()))
        table0, stride0 = tables[0]
        if any(t.shape != table0.shape or s != stride0 for t, s in tables[1:]):
            self._disable("the group's layers have different page geometries")
            return False
        self._dom_table = table0
        self._dom_plan = page_domain_plan(kv0)
        self.page_stride = stride0
        assert self.res is not None
        logger.info(
            "Die-local sparse indexer active for %d layers (%s..): %d pages of "
            "%d rows, %s SMs per domain.",
            len(self.layer_names),
            self.layer_names[0],
            table0.numel(),
            self.page_rows,
            self.res.sms,
        )
        return True

    def _buffers(self) -> dict[str, Any]:
        """The group's persistent outputs, named as `_GroupArgs` fields."""
        if self._bufs is None:
            assert self.res is not None
            rows, width, device = self.max_rows, self.width, self.device

            def zeros(*shape: int, dtype: torch.dtype = torch.int32):
                return torch.zeros(shape, dtype=dtype, device=device)

            self._bufs = dict(
                lbt=[zeros(rows, width) for _ in range(NUM_DOMAINS)],
                llen=[zeros(rows, 1) for _ in range(NUM_DOMAINS)],
                sched=[zeros(s + 1, 2) for s in self.res.sms],
                jmap=zeros(rows, width),
                inv=[zeros(rows, width) for _ in range(NUM_DOMAINS)],
                flags=zeros(rows * _num_chunks(width), dtype=torch.int64),
                state=zeros(4),
            )
        return self._bufs

    def _kernel_dom(self) -> torch.Tensor:
        """The page-to-domain table the build kernel reads, updated in place:
        captured builds keep its address and size."""
        table = self._dom_table
        assert table is not None
        if self._dom_src is not table:
            if self._dom is None:
                self._dom = torch.empty(
                    max(1, table.numel()), dtype=torch.int8, device=table.device
                )
            elif self._dom.numel() < table.numel():
                raise RuntimeError(
                    "Die-local indexer: the indexer cache grew from "
                    f"{self._dom.numel()} to {table.numel()} pages after its "
                    "page-to-domain table was first used."
                )
            self._dom[: table.numel()].copy_(table)
            self._dom_src = table
        return self._dom

    def build(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        num_rows: int,
    ) -> LocalizedDecodeMetadata | None:
        """Record this step's inputs; None runs the stock indexer instead.

        The inputs are the builder's persistent varlen buffers (an int32 block
        table `width` pages wide, and int32 lengths and request indices, one
        per row): the build kernel reads them when the forward's first
        localized scan launches it.
        """
        if (
            self.disabled
            or not use_die_split(num_rows, self._capacity)
            or not self._resolve()
            or not 0 < num_rows <= self.max_rows
        ):
            return None
        assert self.res is not None
        lens = seq_lens.reshape(-1)
        bufs = self._buffers()
        dom = self._kernel_dom()
        q0, k = self._dom_plan
        ptrs = tuple(t.data_ptr() for t in (block_table, lens, indices, dom))
        key = (num_rows, block_table.stride(0), q0, k, *ptrs)
        entry = self._args.get(key)
        if entry is None:
            args = _GroupArgs(
                bt=block_table,
                lens=lens,
                dom=dom,
                indices=indices[:num_rows],
                **bufs,
                R=self.page_rows,
                width=self.width,
                cap=self.max_rows,
                num_rows=num_rows,
                dom_q0=q0,
                dom_k=k,
            )
            meta = LocalizedDecodeMetadata(
                page_rows=self.page_rows,
                capacity=self.max_rows,
                block_tables=[b[:num_rows] for b in bufs["lbt"]],
                seq_lens=[x[:num_rows] for x in bufs["llen"]],
                schedules=list(bufs["sched"]),
                join_map=bufs["jmap"][:num_rows],
                inv_maps=[x[:num_rows] for x in bufs["inv"]],
                page_stride=self.page_stride,
                resources=self.res,
            )
            entry = self._args[key] = (args, meta)
        args, meta = entry
        pending = _PendingBuild(args)
        _PENDING.setdefault(self.res.device, {})[id(self)] = pending
        return replace(meta, pending=pending)


def maybe_make_decode_localizer(
    builder: Any, block_table_width: int
) -> IndexerDecodeLocalizer | None:
    """The localizer of an indexer metadata builder, or None (stock).

    Called from the builder's constructor when `VLLM_INDEXER_LOCALIZE` is set.
    """
    vllm_config = builder.vllm_config
    reason = unsupported_config_reason(vllm_config)
    if reason is not None:
        logger.warning_once(
            "VLLM_INDEXER_LOCALIZE does not support %s; using the stock sparse "
            "indexer.",
            reason,
        )
        return None
    scheduler_config = vllm_config.scheduler_config
    spec = builder.kv_cache_spec
    # One decode row per query token: 1 + the speculative tokens per request.
    rows_per_request = 1 + builder.num_speculative_tokens
    loc = IndexerDecodeLocalizer(
        builder.layer_names,
        vllm_config.compilation_config.static_forward_context,
        builder.device,
        min(
            scheduler_config.max_num_seqs * rows_per_request,
            scheduler_config.max_num_batched_tokens,
        ),
        block_table_width,
        int(getattr(spec, "num_states", spec.block_size)),
    )
    return None if loc.disabled else loc
