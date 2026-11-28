# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap dense MLA chunked-context DCP KV all-gathers behind attention.

While the main stream runs `reorg_kvcache`, `kv_b_proj`, attention, and layer
projections (including layer 0's new-token attention for chunk 0), `side` and a
dedicated zero-SM `copy_stream` prefetch the next chunk (or next layer's chunk 0)
into an alternating ping-pong region over peer-mapped symmetric memory:

- **Wire formats**: `unquantized` (BF16/FP16 straight into the gathered region),
  `fp8_ds_mla` (656 B uint8 -> BF16), and generic `fp8` (`fp8`/`fp8_e4m3`/
  `fp8_e5m2` -> BF16/FP16), saving ~43–50% NVLink traffic on quantized caches.
- **Zero-SM conflict-free ring pull & 2-stream pipeline**: `HEAD_READY` fires
  right after packing `own_wire`; `copy_stream` pulls peers in staggered
  `pull_order` while `side` overlaps local self-copy/own-slot dequantization and
  pipelines per-peer dequantizations when `toks >= PIPELINE_MIN_ROWS`.
- **Race-free historical reads**: local cache reads stop at each request's exact
  historical row count on this rank so `update_kv_cache` on layer `l + 1` is
  never raced even for unaligned context lengths under breakable CUDA graphs.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.ops import kv_ce_rows
from vllm.v1.attention.ops.cp_common import (
    direct_cp_enabled,
    symm_mem,
    symm_mem_available,
)
from vllm.v1.attention.ops.cuda_direct import (
    CopyList,
    DirectKernel,
    Int32Table,
    OnStream,
)
from vllm.v1.attention.ops.kv_ce_overlap import (
    BARRIER_TIMEOUT_MS,
    CHANNEL_HEAD_DONE,
    KV_ENTRY_BYTES,
    PIPELINE_MIN_ROWS,
    PLAN_SLOTS,
    STAGE_BLOCK,
    DCPCopyEngineBase,
    LayerOrder,
    head_chunks,
    plan_hash,
)
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

_CACHE_KIND_CODE = {
    "unquantized": 1,
    "fp8_ds_mla": 2,
    "fp8": 3,
}
_PLAN_DOMAIN_TAG = 0x4D4C4150  # "MLAP"


def classify_kv_cache_kind(
    kv_cache_dtype: str, out_dtype: torch.dtype, row_dim: int
) -> str | None:
    """Return the wire mode for dense MLA DCP prefetch, or None if unsupported."""
    if out_dtype not in (torch.bfloat16, torch.float16):
        return None
    if kv_cache_dtype == "fp8_ds_mla":
        return "fp8_ds_mla" if out_dtype == torch.bfloat16 and row_dim == 576 else None
    if not is_quantized_kv_cache(kv_cache_dtype):
        return "unquantized"
    if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        return None
    if row_dim not in (320, 512, 576):
        return None
    return "fp8"


def chunk_local_row_spans(
    padded_local_seq_lens: list[int],
    local_context_lens_allranks: list[list[int]],
    local_starts: list[int],
    rank: int,
) -> tuple[bool, list[tuple[int, int, int]]]:
    """Compute strictly historical local row spans for `rank` within one chunk.

    Returns `(all_aligned, spans)` where each span is
    `(req_idx, dst_offset, actual_len)`. `dst_offset` matches the padded slot
    layout expected by `reorg_kvcache`, while `actual_len <= padded_len` never
    extends into the current step's new-token slots.
    """
    spans: list[tuple[int, int, int]] = []
    dst_offset = 0
    all_aligned = True
    for req_idx, (padded, allranks, local_start) in enumerate(
        zip(padded_local_seq_lens, local_context_lens_allranks, local_starts)
    ):
        actual = min(max(0, allranks[rank] - local_start), padded)
        if actual != padded:
            all_aligned = False
        spans.append((req_idx, dst_offset, actual))
        dst_offset += padded
    return all_aligned, spans


@dataclass(frozen=True)
class ChunkedDCPPlan:
    """Rank-invariant description of one dense MLA prefill step's DCP gathers."""

    num_layers: int
    world_size: int
    cache_kind: str
    chunk_toks: tuple[int, ...]

    def digest(self) -> list[int]:
        return [
            _PLAN_DOMAIN_TAG,
            self.num_layers,
            self.world_size,
            _CACHE_KIND_CODE[self.cache_kind],
            len(self.chunk_toks),
            *self.chunk_toks,
        ]


def build_chunked_dcp_plan(
    *,
    num_layers: int,
    world_size: int,
    cache_kind: str,
    chunk_toks: list[int],
    max_local_tokens: int,
) -> ChunkedDCPPlan | None:
    """Validate and construct a step plan; returns None when serial is required."""
    if num_layers <= 0 or world_size <= 1 or not chunk_toks:
        return None
    if any(toks <= 0 or toks > max_local_tokens for toks in chunk_toks):
        return None
    return ChunkedDCPPlan(
        num_layers=num_layers,
        world_size=world_size,
        cache_kind=cache_kind,
        chunk_toks=tuple(chunk_toks),
    )


@dataclass(slots=True)
class _PendingItem:
    layer_name: str
    layer_pos: int
    chunk_idx: int
    global_idx: int
    toks: int


@dataclass(slots=True)
class _ChunkCopyPlan:
    peer_chunks: list[CopyList]
    own_copy: CopyList | None
    peer_copies: list[CopyList]
    pipelined: bool


@dataclass(slots=True)
class _StepState:
    plan: ChunkedDCPPlan
    chunks: list[Any]
    block_tables: list[torch.Tensor]
    regions: tuple[int, int]
    slot: int
    kept: set[int]
    checked: bool = False
    copy_cache: dict[tuple[int, int], _ChunkCopyPlan] | None = None

    def __post_init__(self) -> None:
        if self.copy_cache is None:
            self.copy_cache = {}


class ChunkedDCPKVPrefetch(DCPCopyEngineBase):
    """Overlapped zero-SM copy-engine DCP KV gather for dense MLA chunked context."""

    def __init__(
        self,
        group: GroupCoordinator,
        device: torch.device,
        max_gathered_tokens: int,
        row_dim: int,
        out_dtype: torch.dtype,
        kv_cache_dtype: str,
    ) -> None:
        self._init_base(group, device, row_dim)
        self.max_gathered_tokens = max_gathered_tokens
        self.max_local_tokens = max_gathered_tokens // self.world_size
        self.out_dtype = out_dtype
        self.kv_cache_dtype = kv_cache_dtype

        kind = classify_kv_cache_kind(kv_cache_dtype, out_dtype, row_dim)
        self.cache_kind: str = kind or ""
        self.order = LayerOrder()
        # Separate LayerOrder per entry layer so alternating target/draft
        # forwards (e.g. MTP speculative decoding) never unfreeze each other.
        self.orders: dict[str, LayerOrder] = {}
        self.plan: ChunkedDCPPlan | None = None
        self.step: _StepState | None = None
        self.pending: _PendingItem | None = None
        self.position: int | None = None
        self.forward: weakref.ref[Any] | None = None
        self.layers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        self.wire_dtype: torch.dtype = out_dtype
        self.wire_cols = row_dim
        self.wire_row_bytes = row_dim * out_dtype.itemsize
        self.wire_rows = self.max_local_tokens
        self.wire_stage: torch.Tensor | None = None
        self.wire_cache: torch.Tensor | None = None
        self.wire_table: torch.Tensor | None = None
        self.wire_part_table: list[torch.Tensor] = []
        self.own_wire_cache: torch.Tensor | None = None
        self.own_wire_table: torch.Tensor | None = None
        self.stage_ptr = 0
        self.dummy_t2s: torch.Tensor | None = None

        self.ready = bool(kind) and self._allocate()

    def _allocate(self) -> bool:
        world = self.world_size
        try:
            self.gathered = torch.empty(
                (2 * self.max_gathered_tokens, self.row_dim),
                dtype=self.out_dtype,
                device=self.device,
            )
            self.gathered_ptr = self.gathered.data_ptr()
            self.row_bytes = self.row_dim * self.out_dtype.itemsize

            if self.cache_kind == "unquantized":
                self.wire_dtype, self.wire_cols = self.out_dtype, self.row_dim
                self.wire_row_bytes, self.wire_rows = (
                    self.row_bytes,
                    self.max_local_tokens,
                )
            elif self.cache_kind == "fp8_ds_mla":
                self.wire_dtype, self.wire_cols = torch.uint8, KV_ENTRY_BYTES
                self.wire_row_bytes = KV_ENTRY_BYTES
                self.wire_rows = -(-self.max_local_tokens // STAGE_BLOCK) * STAGE_BLOCK
            else:  # "fp8"
                self.wire_dtype = current_platform.fp8_dtype()
                self.wire_cols = self.row_dim
                self.wire_row_bytes = self.row_dim * self.wire_dtype.itemsize
                self.wire_rows = -(-self.max_local_tokens // STAGE_BLOCK) * STAGE_BLOCK

            self.own_wire = symm_mem.empty(
                (self.wire_rows, self.wire_cols),
                dtype=self.wire_dtype,
                device=self.device,
            )
            self.own_wire_ptr = self.own_wire.data_ptr()

            if self.cache_kind != "unquantized":
                blocks = self.wire_rows // STAGE_BLOCK
                self.own_wire_cache = self.own_wire.view(
                    blocks, STAGE_BLOCK, self.wire_cols
                )
                self.own_wire_table = torch.arange(
                    blocks, dtype=torch.int32, device=self.device
                ).unsqueeze(0)
                (
                    self.wire_stage,
                    self.stage_ptr,
                    self.wire_cache,
                    self.wire_table,
                    self.wire_part_table,
                ) = self._init_paged_staging(
                    self.wire_rows, self.wire_cols, self.wire_dtype
                )
                self.dummy_t2s = torch.empty(0, dtype=torch.int32, device=self.device)

            self._init_common_device_state()
            self.slot_starts = torch.zeros(
                world + 1, dtype=torch.int32, device=self.device
            )
            self.slot_writer = Int32Table(self.driver, self.slot_starts)
            words = self.plan_words.data_ptr()
            empty_i32 = self.slot_starts[:0]
            self.plan_writers = [
                Int32Table(
                    self.driver,
                    empty_i32,
                    before=(words + 8 * slot, words + 8 * slot + 4),
                )
                for slot in range(PLAN_SLOTS)
            ]
            self.check_kernel = self._compile_check_kernel()
            allocated = 1
        except (RuntimeError, ValueError, OSError, AttributeError) as error:
            logger.warning(
                "Dense MLA DCP copy-engine prefetch is unavailable: %s", error
            )
            allocated = 0

        all_allocated = self._agree([allocated])
        if not allocated or not all_allocated:
            self.gathered = None  # type: ignore[assignment]
            self.own_wire = None  # type: ignore[assignment]
            self.wire_stage = None
            return False

        self._connect_symm_buffers()
        self._load_kernels()
        return True

    def _compile_check_kernel(self) -> DirectKernel:
        check = kv_ce_rows.plan_check_kernel(self.world_size)
        if check is None:
            raise RuntimeError("plan_check Triton kernel cannot be launched directly")
        return check

    def _load_kernels(self) -> None:
        self._bind_plan_check(self.check_kernel)
        block = torch.zeros(
            (1, STAGE_BLOCK, self.wire_cols),
            dtype=self.wire_dtype,
            device=self.device,
        )
        table = torch.zeros((1, 1), dtype=torch.int32, device=self.device)
        warm_toks = min(STAGE_BLOCK, self.wire_rows)
        warm_cu = torch.tensor([0, warm_toks], dtype=torch.int32, device=self.device)
        warm_scale = torch.ones(1, dtype=torch.float32, device=self.device)
        torch.accelerator.synchronize()
        with OnStream(self.side):
            ops.cp_gather_cache(
                block,
                self.own_wire[:warm_toks],
                table,
                self.wire_bounds,
                1,
                self.wire_zero,
            )
            if self.cache_kind == "fp8_ds_mla":
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    block,
                    self.gathered[:warm_toks],
                    table,
                    self.wire_zero,
                    1,
                )
            elif self.cache_kind == "fp8":
                assert self.dummy_t2s is not None
                ops.gather_and_maybe_dequant_cache(
                    src_cache=block,
                    dst=self.gathered[:warm_toks],
                    block_table=table,
                    cu_seq_lens=warm_cu,
                    token_to_seq=self.dummy_t2s,
                    num_tokens=warm_toks,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=warm_scale,
                )
        torch.accelerator.synchronize()

    def _region(self, parity: int, toks: int) -> torch.Tensor:
        rows = self.world_size * toks
        start = parity * self.max_gathered_tokens
        return self.gathered[start : start + rows]

    def _wire_serves(self, kv_cache: torch.Tensor) -> bool:
        if kv_cache.dim() != 3:
            return False
        if self.cache_kind == "unquantized":
            return (
                kv_cache.dtype == self.out_dtype
                and kv_cache.shape[2] == self.row_dim
                and kv_cache.stride(1) == self.row_dim
            )
        if self.cache_kind == "fp8_ds_mla":
            return (
                kv_cache.dtype == torch.uint8
                and kv_cache.shape[2] == KV_ENTRY_BYTES
                and kv_cache.stride(1) == KV_ENTRY_BYTES
            )
        return (
            kv_cache.dtype == self.wire_dtype
            and kv_cache.shape[2] == self.row_dim
            and kv_cache.stride(1) == self.row_dim
        )

    def begin_layer(
        self,
        layer_name: str,
        kv_cache: torch.Tensor,
        k_scale: torch.Tensor,
        forward: object,
        attn_metadata_of: Callable[[str], Any],
        scheduled: bool = True,
    ) -> None:
        """Register layer state and plan the step when layer 0 begins."""
        self.layers[layer_name] = (kv_cache, k_scale)
        seen = self.forward() if self.forward is not None else None
        new_forward = seen is not forward
        if new_forward:
            self._abandon()
            self.forward = weakref.ref(forward)
        if not scheduled:
            self._abandon()
            self.position = None
            return
        if new_forward:
            order = self.orders.get(layer_name)
            if order is None:
                if len(self.orders) < 8:
                    order = LayerOrder()
                    self.orders[layer_name] = order
                else:
                    self._abandon()
                    self.position = None
                    return
            self.order = order
        self.position = self.order.advance(layer_name, new_forward)
        if self.position is None:
            self._abandon()
        elif self.position == 0:
            self._derive_plan(attn_metadata_of)

    def _derive_plan(self, attn_metadata_of: Callable[[str], Any]) -> None:
        names = self.order.names
        if not names:
            return
        first_meta = attn_metadata_of(names[0])
        if (
            first_meta is None
            or first_meta.prefill is None
            or first_meta.prefill.chunked_context is None
        ):
            return
        chunks = first_meta.prefill.chunked_context.chunks
        if not chunks:
            return
        chunk_toks = [int(c.num_local_context_tokens) for c in chunks]
        plan = build_chunked_dcp_plan(
            num_layers=len(names),
            world_size=self.world_size,
            cache_kind=self.cache_kind,
            chunk_toks=chunk_toks,
            max_local_tokens=self.max_local_tokens,
        )
        if plan is None:
            return

        # Verify every layer in the learned order has compatible cache and
        # identical chunked-context plan.
        block_tables: list[torch.Tensor] = []
        first_cc_id = id(first_meta.prefill.chunked_context)
        for other in names:
            entry = self.layers.get(other)
            if entry is None or not self._wire_serves(entry[0]):
                return
            other_meta = attn_metadata_of(other)
            if (
                other_meta is None
                or other_meta.prefill is None
                or other_meta.prefill.chunked_context is None
            ):
                return
            other_cc = other_meta.prefill.chunked_context
            if id(other_cc) != first_cc_id:
                other_toks = [int(c.num_local_context_tokens) for c in other_cc.chunks]
                if other_toks != chunk_toks:
                    return
            block_tables.append(other_meta.prefill.block_table)

        self.plan = plan
        self.plan_count += 1
        slot = self.plan_count % PLAN_SLOTS
        region_bytes = self.max_gathered_tokens * self.row_bytes
        self.step = _StepState(
            plan=plan,
            chunks=chunks,
            block_tables=block_tables,
            regions=(self.gathered_ptr, self.gathered_ptr + region_bytes),
            slot=slot,
            kept=set(),
        )
        self.plan_writers[slot].write(
            current_stream().cuda_stream, -2, list(plan_hash(plan.digest()))
        )

    def _chunk_copies(self, step: _StepState, parity: int, toks: int) -> _ChunkCopyPlan:
        assert step.copy_cache is not None
        key = (parity, toks)
        cached = step.copy_cache.get(key)
        if cached is not None:
            return cached
        if self.cache_kind == "unquantized":
            region_ptr = step.regions[parity]
            nbytes = toks * self.row_bytes
            # Pull remote peers in staggered ring order on `copy_stream` while
            # `self.side` copies this rank's own shard locally in HBM.
            copies = [
                (
                    region_ptr + peer * nbytes,
                    self.remote_ptrs[peer],
                    nbytes,
                )
                for peer in self.pull_order
            ]
            own_copy = CopyList(
                [(region_ptr + self.rank * nbytes, self.own_wire_ptr, nbytes)]
            )
            built = _ChunkCopyPlan(
                peer_chunks=[CopyList(copies)] if copies else [],
                own_copy=own_copy,
                peer_copies=[],
                pipelined=False,
            )
        else:
            nbytes = toks * self.wire_row_bytes
            per_peer = [
                (
                    self.stage_ptr
                    + self.wire_index[peer] * self.wire_rows * self.wire_row_bytes,
                    self.remote_ptrs[peer],
                    nbytes,
                )
                for peer in self.pull_order
            ]
            pipelined = toks >= PIPELINE_MIN_ROWS and len(self.pull_order) > 1
            built = _ChunkCopyPlan(
                peer_chunks=head_chunks(per_peer, self.wire_row_bytes),
                own_copy=None,
                peer_copies=[CopyList([entry]) for entry in per_peer],
                pipelined=pipelined,
            )
        step.copy_cache[key] = built
        return built

    def _pack_local_rows(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        chunk: Any,
        toks: int,
    ) -> None:
        all_aligned, spans = chunk_local_row_spans(
            chunk.padded_local_seq_lens,
            chunk.local_context_lens_allranks,
            chunk.local_starts,
            self.rank,
        )
        if all_aligned:
            ops.cp_gather_cache(
                src_cache=kv_cache,
                dst=self.own_wire[:toks],
                block_table=block_table,
                cu_seq_lens=chunk.padded_local_cu_seq_lens,
                batch_size=chunk.num_requests,
                seq_starts=chunk.starts,
            )
            return
        for req_idx, dst_offset, actual in spans:
            if actual > 0:
                ops.cp_gather_cache(
                    src_cache=kv_cache,
                    dst=self.own_wire[dst_offset : dst_offset + actual],
                    block_table=block_table[req_idx : req_idx + 1],
                    cu_seq_lens=self.wire_bounds,
                    batch_size=1,
                    seq_starts=chunk.starts[req_idx : req_idx + 1],
                )

    def _dequant_slice(
        self,
        src_cache: torch.Tensor,
        dst: torch.Tensor,
        table: torch.Tensor,
        count: int,
        toks: int,
        k_scale: torch.Tensor,
    ) -> None:
        if self.cache_kind == "fp8_ds_mla":
            ws = self.wire_zero if count == 1 else self.slot_starts[:count]
            ops.cp_gather_and_upconvert_fp8_kv_cache(src_cache, dst, table, ws, count)
        else:
            assert self.dummy_t2s is not None
            ops.gather_and_maybe_dequant_cache(
                src_cache=src_cache,
                dst=dst,
                block_table=table,
                cu_seq_lens=self.slot_starts[: count + 1],
                token_to_seq=self.dummy_t2s,
                num_tokens=count * toks,
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
            )

    def _dequant_quantized(
        self, region: torch.Tensor, toks: int, k_scale: torch.Tensor
    ) -> None:
        assert self.wire_cache is not None and self.wire_table is not None
        for first, end in ((0, self.rank), (self.rank + 1, self.world_size)):
            count = end - first
            if count > 0:
                idx = self.wire_index[first]
                self._dequant_slice(
                    self.wire_cache,
                    region[first * toks : end * toks],
                    self.wire_table[idx : idx + count],
                    count,
                    toks,
                    k_scale,
                )

    def _dequant_peer(
        self, region: torch.Tensor, peer: int, toks: int, k_scale: torch.Tensor
    ) -> None:
        assert self.wire_cache is not None
        self._dequant_slice(
            self.wire_cache,
            region[peer * toks : (peer + 1) * toks],
            self.wire_part_table[self.wire_index[peer]],
            1,
            toks,
            k_scale,
        )

    def _dequant_own(
        self, region: torch.Tensor, toks: int, k_scale: torch.Tensor
    ) -> None:
        assert self.own_wire_cache is not None and self.own_wire_table is not None
        self._dequant_slice(
            self.own_wire_cache,
            region[self.rank * toks : (self.rank + 1) * toks],
            self.own_wire_table,
            1,
            toks,
            k_scale,
        )

    def _launch_item(self, layer_pos: int, chunk_idx: int) -> None:
        step = self.step
        assert step is not None
        name = self.order.names[layer_pos]
        kv_cache, k_scale = self.layers[name]
        chunk = step.chunks[chunk_idx]
        toks = int(chunk.num_local_context_tokens)
        block_table = step.block_tables[layer_pos][chunk.request_slice]
        num_chunks = len(step.chunks)
        global_idx = layer_pos * num_chunks + chunk_idx
        parity = global_idx % 2

        main = current_stream()
        driver = self.driver
        driver.record(self.fence, main.cuda_stream)
        driver.wait_event(self.side_handle, self.fence)

        # Keep metadata tensors alive until the auxiliary stream finishes.
        for tensor in (block_table, chunk.padded_local_cu_seq_lens, chunk.starts):
            tid = id(tensor)
            if tid not in step.kept:
                step.kept.add(tid)
                tensor.record_stream(self.side)

        region = self._region(parity, toks)
        copy_plan = self._chunk_copies(step, parity, toks)

        torch.cuda.set_stream(self.side)
        try:
            if self.head_done_owed:
                self.handle.barrier(
                    channel=CHANNEL_HEAD_DONE, timeout_ms=BARRIER_TIMEOUT_MS
                )
                self.head_done_owed = False
            self._pack_local_rows(kv_cache, block_table, chunk, toks)
            # Peers only read `own_wire`: signal HEAD_READY immediately so
            # zero-SM copy-engine transfers start while `self.side` handles
            # local self-copy or own-slot dequantization.
            self._head_ready()
            side_h = self.side_handle
            copy_h = self.copy_handle
            driver.record(self.copy_ready, side_h)
            driver.wait_event(copy_h, self.copy_ready)
            if self.cache_kind == "unquantized":
                for copy_list in copy_plan.peer_chunks:
                    driver.memcpy_list(copy_list, copy_h)
                if self.peer_events:
                    driver.record(self.peer_events[0], copy_h)
                if copy_plan.own_copy is not None:
                    driver.memcpy_list(copy_plan.own_copy, side_h)
                if self.peer_events:
                    driver.wait_event(side_h, self.peer_events[0])
            else:
                if copy_plan.pipelined:
                    for peer_idx, copy_list in enumerate(copy_plan.peer_copies):
                        driver.memcpy_list(copy_list, copy_h)
                        driver.record(self.peer_events[peer_idx], copy_h)
                else:
                    for copy_list in copy_plan.peer_chunks:
                        driver.memcpy_list(copy_list, copy_h)
                    if self.peer_events:
                        driver.record(self.peer_events[0], copy_h)
                self.slot_writer.write(
                    side_h, 0, [i * toks for i in range(self.world_size + 1)]
                )
                self._dequant_own(region, toks, k_scale)
                if copy_plan.pipelined:
                    for peer_idx, peer in enumerate(self.pull_order):
                        driver.wait_event(side_h, self.peer_events[peer_idx])
                        self._dequant_peer(region, peer, toks, k_scale)
                else:
                    if self.peer_events:
                        driver.wait_event(side_h, self.peer_events[0])
                    self._dequant_quantized(region, toks, k_scale)
        finally:
            torch.cuda.set_stream(main)

        self.head_done_owed = True
        self.pending = _PendingItem(name, layer_pos, chunk_idx, global_idx, toks)

    def start_first(self) -> None:
        """Kick off `(layer 0, chunk 0)` ahead of layer 0's new-token attention."""
        if self.step is None or self.position != 0 or self.pending is not None:
            return
        self._launch_item(0, 0)

    def join(
        self, layer_name: str, chunk_idx: int, scheduled: bool = True
    ) -> torch.Tensor | None:
        """Wait for `(layer_name, chunk_idx)` and return its gathered tensor, or
        return None when this step takes the serial path."""
        pending = self.pending
        if (
            pending is None
            or pending.layer_name != layer_name
            or pending.chunk_idx != chunk_idx
            or not scheduled
        ):
            self._pay_head_done("main", scheduled)
            return None
        self.pending = None
        self._wait(current_stream(), self.side)
        return self._region(pending.global_idx % 2, pending.toks)

    def start_next(self, layer_name: str, chunk_idx: int) -> None:
        """Launch prefetch of the next chunk in this layer, or chunk 0 of the
        next layer, right after joining `(layer_name, chunk_idx)`."""
        step, pos = self.step, self.position
        if step is None or pos is None or self.order.names[pos] != layer_name:
            return
        next_chunk = chunk_idx + 1
        if next_chunk < len(step.chunks):
            self._launch_item(pos, next_chunk)
            return
        next_pos = pos + 1
        if next_pos < len(self.order.names):
            self._launch_item(next_pos, 0)
            return
        # Last chunk of the last layer.
        self.step = None


@functools.cache
def get_chunked_dcp_kv_prefetch(
    group: GroupCoordinator,
    device: torch.device,
    max_gathered_tokens: int,
    row_dim: int,
    out_dtype: torch.dtype,
    kv_cache_dtype: str,
) -> ChunkedDCPKVPrefetch | None:
    """Shared per-device dense MLA DCP prefetch state, or None if unavailable."""
    if group.world_size <= 1 or max_gathered_tokens <= 0 or row_dim <= 0:
        return None
    if max_gathered_tokens % group.world_size != 0:
        return None
    if classify_kv_cache_kind(kv_cache_dtype, out_dtype, row_dim) is None:
        return None
    if not symm_mem_available or not direct_cp_enabled(group, out_dtype, None):
        return None
    state = ChunkedDCPKVPrefetch(
        group,
        device,
        max_gathered_tokens,
        row_dim,
        out_dtype,
        kv_cache_dtype,
    )
    if state.ready:
        logger.info_once(
            "Using copy-engine DCP KV prefetch for dense MLA (%s, max_gathered=%d).",
            state.cache_kind,
            max_gathered_tokens,
        )
        return state
    return None
