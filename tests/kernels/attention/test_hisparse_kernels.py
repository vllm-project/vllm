# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact reference model and differential tests for the HiSparse kernels.

The four HiSparse ops (``hisparse_resolve_residency``,
``hisparse_invalidate_written_slots``, ``hisparse_gather_plan``,
``hisparse_gather_compact``) only move bytes and indices -- they are dtype
agnostic and contain no floating-point math. That makes an exact CPU oracle
practical, and exactness is what we need: a wrong wavefront mask does not
crash, it hands attention a plausible-looking wrong row.

Two notes that shape what is worth asserting here:

* The resolver's open-addressing table is internal. Linear probing with no
  deletions finds an inserted key from its home slot regardless of the order
  concurrent inserts won their slots, and ``hash_size == 2 * top_k`` keeps the
  table under half full so probes always terminate. So for the documented
  precondition (global ids unique within a row) the table is equivalent to a
  plain dict, and no choice of hash function is observable through the op.

* ``copy_row_warp``/``zero_row_warp`` have 16-byte, 4-byte and scalar paths,
  but every op validates ``row_bytes % 16 == 0`` and requires contiguous rows,
  so only the 16-byte path is reachable through the public ops. The narrower
  paths still must be ported, they just cannot be covered from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

POISON = np.int32(-999)

DEVICE = current_platform.device_type


def _hisparse_ops_available() -> bool:
    import vllm  # noqa: F401  (registers the stable-ABI ops)

    ops = getattr(torch.ops, "_C_cache_ops", None)
    if ops is None:
        return False
    return all(
        hasattr(ops, name)
        for name in (
            "hisparse_resolve_residency",
            "hisparse_invalidate_written_slots",
            "hisparse_gather_plan",
            "hisparse_gather_compact",
        )
    )


_OPS_MISSING_REASON = (
    "HiSparse ops are not compiled into this build "
    "(csrc/libtorch_stable/hisparse_kernels.cu is gated on CUDA). "
    "Set VLLM_HISPARSE_REQUIRE_OPS=1 to turn this skip into a failure."
)


def _requires_ops() -> None:
    if _hisparse_ops_available():
        return
    if os.environ.get("VLLM_HISPARSE_REQUIRE_OPS") == "1":
        pytest.fail(_OPS_MISSING_REASON)
    pytest.skip(_OPS_MISSING_REASON)


requires_hisparse_ops = pytest.fixture(_requires_ops)


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------


@dataclass
class ResolveState:
    """Long-lived device state the resolver reads and rewrites each step."""

    device_global_indices: np.ndarray  # int32 [num_states, region_stride]
    lru_slots: np.ndarray  # int16 [num_states, hot_size]


@dataclass
class ResolveOutputs:
    """Per-step outputs. Entries the kernel does not write keep their poison."""

    hot_indices: np.ndarray
    attention_indices: np.ndarray | None = None
    miss_mask: np.ndarray | None = None
    resolved_global_indices: np.ndarray | None = None
    valid_counts: np.ndarray | None = None
    swap_host_physical_rows: np.ndarray | None = None
    swap_device_physical_rows: np.ndarray | None = None
    swap_counts: np.ndarray | None = None


@dataclass
class ResolveConfig:
    host_rows: int
    hot_block_size: int
    attention_block_stride: int
    region_stride: int
    source_block_size: int = 0
    resident_block_size: int = 0
    resident_null_block: int = 0
    hot_block_table: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), np.int32)
    )
    request_ids: np.ndarray | None = None
    request_state_indices: np.ndarray | None = None
    source_block_table: np.ndarray | None = None
    resident_block_table: np.ndarray | None = None


def _store_hot_index(
    out: ResolveOutputs,
    row: int,
    col: int,
    physical_row: int,
    hot_block_size: int,
    attention_block_stride: int,
) -> None:
    out.hot_indices[row, col] = physical_row
    if out.attention_indices is not None:
        out.attention_indices[row, col] = (
            -1
            if physical_row < 0
            else (physical_row // hot_block_size) * attention_block_stride
            + physical_row % hot_block_size
        )


def _physical_hot_row(
    hot_block_table: np.ndarray, row: int, hot_block_size: int, slot: int
) -> int:
    block = int(hot_block_table[row, slot // hot_block_size])
    return block * hot_block_size + slot % hot_block_size


def ref_resolve_residency(
    global_indices: np.ndarray,
    state: ResolveState,
    out: ResolveOutputs,
    cfg: ResolveConfig,
) -> None:
    """Exact CPU model of ``hisparse_resolve_residency``.

    Mirrors the kernel's five phases. Mutates ``state`` and ``out`` in place so
    that "the kernel did not write here" is itself an assertable outcome.
    """
    num_rows, top_k = global_indices.shape
    hot_size = state.lru_slots.shape[1]
    region_stride = cfg.region_stride

    for batch_row in range(num_rows):
        request_row = (
            int(cfg.request_ids[batch_row])
            if cfg.request_ids is not None
            else batch_row
        )
        if cfg.request_state_indices is not None:
            in_range = 0 <= request_row < cfg.request_state_indices.shape[0]
            state_row = int(cfg.request_state_indices[request_row]) if in_range else -1
        else:
            state_row = request_row

        # CUDA-graph padding rows publish -1 and touch no long-lived state.
        if state_row < 0:
            for i in range(top_k):
                _store_hot_index(
                    out,
                    batch_row,
                    i,
                    -1,
                    cfg.hot_block_size,
                    cfg.attention_block_stride,
                )
                if out.resolved_global_indices is not None:
                    out.resolved_global_indices[batch_row, i] = -1
            if out.valid_counts is not None:
                out.valid_counts[batch_row] = 0
            if out.swap_counts is not None:
                out.swap_counts[batch_row] = 0
            continue

        row_dgi = state.device_global_indices[state_row]
        row_lru = state.lru_slots[state_row]

        # --- Phase 1: translate + resolve against the resident cache ---------
        pending: dict[int, int] = {}  # global id -> top-k column, still unresolved
        resolved_g = [-1] * top_k  # translated host row per column
        resolved_count = 0
        valid_count = 0
        done = [False] * top_k

        for i in range(top_k):
            token_index = int(global_indices[batch_row, i])
            g = token_index
            resident_row = -1

            if cfg.source_block_table is not None:
                req = int(cfg.request_ids[batch_row])
                src_block = (
                    token_index // cfg.source_block_size if token_index >= 0 else -1
                )
                if (
                    0 <= req < cfg.source_block_table.shape[0]
                    and 0 <= src_block < cfg.source_block_table.shape[1]
                ):
                    physical_block = int(cfg.source_block_table[req, src_block])
                    # NOTE: the kernel tests `> 0`, so source block 0 is a null
                    # block, not a valid page.
                    g = (
                        physical_block * cfg.source_block_size
                        + token_index % cfg.source_block_size
                        if physical_block > 0
                        else -1
                    )
                else:
                    g = -1

                if cfg.resident_block_table is not None:
                    res_block = (
                        token_index // cfg.resident_block_size
                        if token_index >= 0
                        else -1
                    )
                    if (
                        0 <= req < cfg.resident_block_table.shape[0]
                        and 0 <= res_block < cfg.resident_block_table.shape[1]
                    ):
                        physical_block = int(cfg.resident_block_table[req, res_block])
                        if (
                            physical_block != cfg.resident_null_block
                            and physical_block >= 0
                        ):
                            resident_row = (
                                physical_block * cfg.resident_block_size
                                + token_index % cfg.resident_block_size
                            )

            if g >= cfg.host_rows:
                g = -1

            resolved_g[i] = g
            if out.resolved_global_indices is not None:
                out.resolved_global_indices[batch_row, i] = g
            if resident_row >= 0 or g >= 0:
                valid_count += 1
            if out.miss_mask is not None:
                out.miss_mask[batch_row, i] = 0

            if resident_row >= 0:
                _store_hot_index(
                    out,
                    batch_row,
                    i,
                    resident_row,
                    cfg.hot_block_size,
                    cfg.attention_block_stride,
                )
                done[i] = True
                resolved_count += 1
            elif g < 0:
                _store_hot_index(
                    out,
                    batch_row,
                    i,
                    -1,
                    cfg.hot_block_size,
                    cfg.attention_block_stride,
                )
                done[i] = True
                resolved_count += 1
            else:
                pending[g] = i

        if out.valid_counts is not None:
            out.valid_counts[batch_row] = valid_count

        # Fully resident rows never consult or rewrite the hot LRU.
        if resolved_count == top_k:
            if out.swap_counts is not None:
                out.swap_counts[batch_row] = 0
            continue

        # --- Phase 2: scan hot slots in LRU order, classify and compact ------
        # Compaction is a parallel prefix sum over fixed-width chunks, but the
        # resulting order is simply "increasing LRU position" and so is
        # independent of the chunk (wavefront) width. That independence is
        # exactly what makes this a valid oracle for a wave64 port.
        hits: list[int] = []  # slots, oldest-first
        evictables: list[int] = []  # slots, oldest-first
        for pos in range(hot_size):
            slot = int(row_lru[pos])
            # Corruption tripwire: an out-of-range slot degrades to a re-miss
            # rather than an unbounded device_global_indices read.
            cached_g = int(row_dgi[slot]) if 0 <= slot < region_stride else -1
            # Lookup, not removal: the kernel never deletes from its hash
            # table, so if two hot slots somehow hold the same global id both
            # count as hits (and store the same column twice, benignly).
            col = pending.get(cached_g) if cached_g >= 0 else None
            if col is not None:
                done[col] = True
                hits.append(slot)
                _store_hot_index(
                    out,
                    batch_row,
                    col,
                    _physical_hot_row(
                        cfg.hot_block_table, request_row, cfg.hot_block_size, slot
                    ),
                    cfg.hot_block_size,
                    cfg.attention_block_stride,
                )
            else:
                evictables.append(slot)

        # --- Phase 3: compact misses, assign eviction slots oldest-first -----
        misses = [i for i in range(top_k) if not done[i]]
        for m, i in enumerate(misses):
            g = resolved_g[i]
            evict_slot = evictables[m]
            compact = m
            if not (0 <= evict_slot < region_stride):
                # Same tripwire as phase 2: resolve invalid, re-miss later.
                _store_hot_index(
                    out,
                    batch_row,
                    i,
                    -1,
                    cfg.hot_block_size,
                    cfg.attention_block_stride,
                )
                if out.swap_host_physical_rows is not None:
                    out.swap_host_physical_rows[batch_row, compact] = g
                    out.swap_device_physical_rows[batch_row, compact] = -1
                continue
            physical_row = _physical_hot_row(
                cfg.hot_block_table, request_row, cfg.hot_block_size, evict_slot
            )
            _store_hot_index(
                out,
                batch_row,
                i,
                physical_row,
                cfg.hot_block_size,
                cfg.attention_block_stride,
            )
            if out.swap_host_physical_rows is not None:
                out.swap_host_physical_rows[batch_row, compact] = g
                out.swap_device_physical_rows[batch_row, compact] = physical_row
            if out.miss_mask is not None:
                out.miss_mask[batch_row, i] = 1
            row_dgi[evict_slot] = g

        total_misses = len(misses)
        if out.swap_counts is not None:
            out.swap_counts[batch_row] = total_misses

        # --- Phase 4: rewrite the LRU order ---------------------------------
        # stale evictables (oldest first) | freshly filled misses | hits at MRU
        new_lru = evictables[total_misses:] + evictables[:total_misses] + hits
        assert len(new_lru) == hot_size
        row_lru[:] = np.asarray(new_lru, dtype=np.int16)


def ref_gather_plan(
    host_cache: np.ndarray,
    hot_cache: np.ndarray,
    global_indices: np.ndarray,
    hot_indices: np.ndarray,
    miss_mask: np.ndarray,
    request_state_indices: np.ndarray | None,
    attention_indices: np.ndarray | None,
    attention_block_stride: int,
    hot_block_size: int,
    hot_block_stride: int,
) -> None:
    """Exact model of ``hisparse_gather_plan``. Byte arrays are flat uint8."""
    num_rows, top_k = global_indices.shape
    host_rows, row_bytes = host_cache.shape
    hot_rows = (hot_cache.size // hot_block_stride) * hot_block_size

    for row in range(num_rows):
        is_padding = (
            request_state_indices is not None and int(request_state_indices[row]) < 0
        )
        for col in range(top_k):
            dst = int(hot_indices[row, col])
            if attention_indices is not None:
                attention_indices[row, col] = (
                    -1
                    if is_padding or dst < 0
                    else (dst // hot_block_size) * attention_block_stride
                    + dst % hot_block_size
                )
            if is_padding or int(miss_mask[row, col]) == 0:
                continue
            g = int(global_indices[row, col])
            if g < 0 or dst < 0 or dst >= hot_rows:
                continue
            off = (dst // hot_block_size) * hot_block_stride + (
                dst % hot_block_size
            ) * row_bytes
            if g < host_rows:
                hot_cache[off : off + row_bytes] = host_cache[g]
            else:
                # No source row: zero rather than serve stale bytes.
                hot_cache[off : off + row_bytes] = 0


def ref_gather_compact(
    host_cache: np.ndarray,
    hot_cache: np.ndarray,
    miss_global_indices: np.ndarray,
    miss_hot_indices: np.ndarray,
    miss_counts: np.ndarray,
    hot_block_size: int,
    hot_block_stride: int,
) -> None:
    num_rows, top_k = miss_global_indices.shape
    host_rows, row_bytes = host_cache.shape
    hot_rows = (hot_cache.size // hot_block_stride) * hot_block_size

    for row in range(num_rows):
        miss_count = min(max(int(miss_counts[row]), 0), top_k)
        for col in range(miss_count):
            g = int(miss_global_indices[row, col])
            dst = int(miss_hot_indices[row, col])
            if g < 0 or dst < 0 or dst >= hot_rows:
                continue
            off = (dst // hot_block_size) * hot_block_stride + (
                dst % hot_block_size
            ) * row_bytes
            if g < host_rows:
                hot_cache[off : off + row_bytes] = host_cache[g]
            else:
                hot_cache[off : off + row_bytes] = 0


def ref_invalidate_written_slots(
    device_global_indices: np.ndarray,
    request_state_indices: np.ndarray,
    req_id_per_token: np.ndarray,
    written_slots: np.ndarray,
) -> None:
    num_state_rows, region_stride = device_global_indices.shape
    num_request_ids = request_state_indices.shape[0]
    for token_idx in range(written_slots.shape[0]):
        req_idx = int(req_id_per_token[token_idx])
        if not 0 <= req_idx < num_request_ids:
            continue
        state_idx = int(request_state_indices[req_idx])
        written_slot = int(written_slots[token_idx])
        if not (0 <= state_idx < num_state_rows) or written_slot < 0:
            continue
        row = device_global_indices[state_idx]
        row[row == written_slot] = -1


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

FP8_DS_MLA_ROW_BYTES = 656


@dataclass
class ResolveCase:
    """A fully materialized resolver invocation, host-side."""

    global_indices: np.ndarray
    cfg: ResolveConfig
    state: ResolveState
    hot_num_blocks: int
    row_bytes: int = FP8_DS_MLA_ROW_BYTES

    @property
    def num_rows(self) -> int:
        return self.global_indices.shape[0]

    @property
    def top_k(self) -> int:
        return self.global_indices.shape[1]

    @property
    def hot_size(self) -> int:
        return self.state.lru_slots.shape[1]


def _blank_outputs(num_rows: int, top_k: int) -> ResolveOutputs:
    """All-poison outputs, so 'the kernel never wrote here' is assertable."""

    def poisoned(shape):
        return np.full(shape, POISON, dtype=np.int32)

    return ResolveOutputs(
        hot_indices=poisoned((num_rows, top_k)),
        attention_indices=poisoned((num_rows, top_k)),
        miss_mask=poisoned((num_rows, top_k)),
        resolved_global_indices=poisoned((num_rows, top_k)),
        valid_counts=poisoned(num_rows),
        swap_host_physical_rows=poisoned((num_rows, top_k)),
        swap_device_physical_rows=poisoned((num_rows, top_k)),
        swap_counts=poisoned(num_rows),
    )


def make_case(
    *,
    top_k: int,
    hot_size: int,
    num_rows: int = 2,
    num_states: int | None = None,
    hot_block_size: int = 64,
    source_block_size: int = 16,
    host_rows: int = 4096,
    seed: int = 0,
    padding_rows: tuple[int, ...] = (),
    resident_fraction: float = 0.0,
    null_fraction: float = 0.0,
) -> ResolveCase:
    """Build a self-consistent resolver case.

    ``resident_fraction`` routes that share of columns through the resident
    block table (they resolve in phase 1 and never touch the hot buffer);
    ``null_fraction`` maps that share onto source block 0, which the kernel
    treats as a null page.
    """
    rng = np.random.default_rng(seed)
    num_states = num_states if num_states is not None else num_rows
    region_stride = hot_size

    # Each batch row is one request; padding rows get state index -1.
    request_ids = np.arange(num_rows, dtype=np.int32)
    request_state_indices = np.arange(num_rows, dtype=np.int32) % max(num_states, 1)
    for r in padding_rows:
        request_state_indices[r] = -1

    max_token = top_k * 4
    num_source_blocks = max_token // source_block_size + 2
    # Block 0 is the null page; hand out distinct positive pages elsewhere so
    # translated ids stay unique within a row (the kernel's precondition).
    source_block_table = np.zeros((num_rows, num_source_blocks), dtype=np.int32)
    next_page = 1
    for r in range(num_rows):
        for b in range(num_source_blocks):
            source_block_table[r, b] = next_page
            next_page += 1

    if null_fraction > 0:
        for r in range(num_rows):
            for b in range(num_source_blocks):
                if rng.random() < null_fraction:
                    source_block_table[r, b] = 0

    resident_block_table = None
    resident_block_size = 0
    resident_null_block = 0
    if resident_fraction > 0:
        resident_block_size = hot_block_size
        resident_null_block = 0
        num_res_blocks = max_token // resident_block_size + 2
        resident_block_table = np.zeros((num_rows, num_res_blocks), dtype=np.int32)
        for r in range(num_rows):
            for b in range(num_res_blocks):
                resident_block_table[r, b] = (
                    (r * num_res_blocks + b + 1)
                    if rng.random() < resident_fraction
                    else resident_null_block
                )

    # Unique token positions per row.
    global_indices = np.stack(
        [
            rng.choice(max_token, size=top_k, replace=False).astype(np.int32)
            for _ in range(num_rows)
        ]
    )

    hot_rows_needed = region_stride
    hot_blocks_per_row = (hot_rows_needed + hot_block_size - 1) // hot_block_size
    hot_block_table = np.arange(num_rows * hot_blocks_per_row, dtype=np.int32).reshape(
        num_rows, hot_blocks_per_row
    )
    hot_num_blocks = num_rows * hot_blocks_per_row

    cfg = ResolveConfig(
        host_rows=host_rows,
        hot_block_size=hot_block_size,
        attention_block_stride=hot_block_size,
        region_stride=region_stride,
        source_block_size=source_block_size,
        resident_block_size=resident_block_size,
        resident_null_block=resident_null_block,
        hot_block_table=hot_block_table,
        request_ids=request_ids,
        request_state_indices=request_state_indices,
        source_block_table=source_block_table,
        resident_block_table=resident_block_table,
    )
    state = ResolveState(
        device_global_indices=np.full((num_states, region_stride), -1, dtype=np.int32),
        lru_slots=np.tile(np.arange(hot_size, dtype=np.int16), (num_states, 1)),
    )
    return ResolveCase(
        global_indices=global_indices,
        cfg=cfg,
        state=state,
        hot_num_blocks=hot_num_blocks,
    )


def _pinned(t: torch.Tensor) -> torch.Tensor:
    return t.pin_memory()


def run_device_resolve(case: ResolveCase, state: ResolveState) -> ResolveOutputs:
    """Invoke the real op, returning outputs in the reference model's shape."""
    cfg = case.cfg
    num_rows, top_k = case.num_rows, case.top_k
    row_width = case.row_bytes  # uint8 rows

    host_cache = _pinned(torch.zeros(cfg.host_rows, row_width, dtype=torch.uint8))
    hot_cache = torch.zeros(
        case.hot_num_blocks,
        cfg.hot_block_size,
        row_width,
        dtype=torch.uint8,
        device=DEVICE,
    )

    def dev(a, dtype=torch.int32):
        return torch.as_tensor(a.copy()).to(dtype).to(DEVICE)

    d_hot_indices = dev(np.full((num_rows, top_k), POISON, np.int32))
    d_attention = dev(np.full((num_rows, top_k), POISON, np.int32))
    d_miss = dev(np.full((num_rows, top_k), POISON, np.int32))
    d_resolved = dev(np.full((num_rows, top_k), POISON, np.int32))
    d_valid = dev(np.full(num_rows, POISON, np.int32))
    d_swap_host = dev(np.full((num_rows, top_k), POISON, np.int32))
    d_swap_dev = dev(np.full((num_rows, top_k), POISON, np.int32))
    d_swap_counts = dev(np.full(num_rows, POISON, np.int32))

    d_dgi = dev(state.device_global_indices)
    d_lru = dev(state.lru_slots, torch.int16)

    torch.ops._C_cache_ops.hisparse_resolve_residency(
        host_cache,
        hot_cache,
        dev(cfg.hot_block_table),
        dev(case.global_indices),
        d_hot_indices,
        d_dgi,
        d_lru,
        dev(cfg.request_state_indices),
        cfg.region_stride,
        d_miss,
        d_attention,
        cfg.attention_block_stride,
        dev(cfg.request_ids),
        dev(cfg.source_block_table),
        cfg.source_block_size,
        d_resolved,
        d_valid,
        d_swap_host,
        d_swap_dev,
        d_swap_counts,
        dev(cfg.resident_block_table) if cfg.resident_block_table is not None else None,
        cfg.resident_block_size,
        cfg.resident_null_block,
    )
    torch.cuda.synchronize()

    # Write the mutated long-lived state back so the caller can keep stepping.
    state.device_global_indices[:] = d_dgi.cpu().numpy()
    state.lru_slots[:] = d_lru.cpu().numpy()

    return ResolveOutputs(
        hot_indices=d_hot_indices.cpu().numpy(),
        attention_indices=d_attention.cpu().numpy(),
        miss_mask=d_miss.cpu().numpy(),
        resolved_global_indices=d_resolved.cpu().numpy(),
        valid_counts=d_valid.cpu().numpy(),
        swap_host_physical_rows=d_swap_host.cpu().numpy(),
        swap_device_physical_rows=d_swap_dev.cpu().numpy(),
        swap_counts=d_swap_counts.cpu().numpy(),
    )


def assert_outputs_equal(got: ResolveOutputs, want: ResolveOutputs, ctx: str) -> None:
    for name in (
        "hot_indices",
        "attention_indices",
        "miss_mask",
        "resolved_global_indices",
        "valid_counts",
        "swap_counts",
    ):
        np.testing.assert_array_equal(
            getattr(got, name), getattr(want, name), err_msg=f"{ctx}: {name}"
        )
    # Compact swap rows are only defined below each row's swap_count.
    for row, count in enumerate(want.swap_counts):
        if count == POISON or count < 0:
            continue
        np.testing.assert_array_equal(
            got.swap_host_physical_rows[row, :count],
            want.swap_host_physical_rows[row, :count],
            err_msg=f"{ctx}: swap_host_physical_rows[{row}]",
        )
        np.testing.assert_array_equal(
            got.swap_device_physical_rows[row, :count],
            want.swap_device_physical_rows[row, :count],
            err_msg=f"{ctx}: swap_device_physical_rows[{row}]",
        )


def assert_state_equal(got: ResolveState, want: ResolveState, ctx: str) -> None:
    # Exact LRU *order*, not set membership. Verified against a simulated
    # wave64 compaction bug (compaction done 32-wide on a 64-wide wavefront):
    # the LRU *sets* stay identical and only the order is wrong, so a
    # set-membership check never fires. Comparing state catches it on step 0;
    # comparing outputs alone does not catch it until step 1, which is why the
    # multi-step tests exist.
    np.testing.assert_array_equal(
        got.lru_slots, want.lru_slots, err_msg=f"{ctx}: lru_slots order"
    )
    np.testing.assert_array_equal(
        got.device_global_indices,
        want.device_global_indices,
        err_msg=f"{ctx}: device_global_indices",
    )


def _ref_step(case: ResolveCase, state: ResolveState) -> ResolveOutputs:
    out = _blank_outputs(case.num_rows, case.top_k)
    ref_resolve_residency(case.global_indices, state, out, case.cfg)
    return out


# ---------------------------------------------------------------------------
# Reference-model self-consistency (runs everywhere, no GPU, no compiled ops)
#
# These guard the oracle itself. If the oracle is wrong, every differential
# test below is worthless, so these must hold independently of the kernel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top_k", [1, 31, 32, 33, 63, 64, 65, 128])
def test_ref_lru_is_a_permutation(top_k: int) -> None:
    """Phase 4 must rewrite the LRU as a permutation of the same slots."""
    case = make_case(top_k=top_k, hot_size=2 * top_k, seed=top_k)
    before = case.state.lru_slots.copy()
    _ref_step(case, case.state)
    for r in range(before.shape[0]):
        assert sorted(case.state.lru_slots[r]) == sorted(before[r])


@pytest.mark.parametrize("top_k", [32, 64, 65])
def test_ref_hits_are_mru_after_step(top_k: int) -> None:
    """A second identical step must hit everything and evict nothing."""
    case = make_case(top_k=top_k, hot_size=2 * top_k, seed=7)
    first = _ref_step(case, case.state)
    assert first.swap_counts.sum() > 0, "cold start should miss"

    dgi_before = case.state.device_global_indices.copy()
    second = _ref_step(case, case.state)
    assert (second.swap_counts == 0).all(), "warm replay must not swap"
    np.testing.assert_array_equal(second.hot_indices, first.hot_indices)
    # Ownership is unchanged, so no eviction occurred.
    np.testing.assert_array_equal(case.state.device_global_indices, dgi_before)


def test_ref_padding_rows_publish_minus_one_and_touch_no_state() -> None:
    case = make_case(top_k=32, hot_size=64, num_rows=3, padding_rows=(1,), seed=3)
    lru_before = case.state.lru_slots.copy()
    dgi_before = case.state.device_global_indices.copy()
    out = _ref_step(case, case.state)

    assert (out.hot_indices[1] == -1).all()
    assert (out.attention_indices[1] == -1).all()
    assert (out.resolved_global_indices[1] == -1).all()
    assert out.valid_counts[1] == 0
    assert out.swap_counts[1] == 0
    # miss_mask for a padding row is never written by the kernel.
    assert (out.miss_mask[1] == POISON).all()
    # Row 1's state row is -1, so no long-lived row belongs to it. Rows 0 and 2
    # own states 0 and 2; state 1 must be untouched.
    np.testing.assert_array_equal(case.state.lru_slots[1], lru_before[1])
    np.testing.assert_array_equal(case.state.device_global_indices[1], dgi_before[1])


def test_ref_fully_resident_row_skips_the_hot_buffer() -> None:
    """The phase-1 early exit must leave the LRU and ownership untouched."""
    case = make_case(top_k=32, hot_size=64, num_rows=1, resident_fraction=1.0, seed=11)
    lru_before = case.state.lru_slots.copy()
    dgi_before = case.state.device_global_indices.copy()
    out = _ref_step(case, case.state)

    assert out.swap_counts[0] == 0
    assert (out.hot_indices[0] >= 0).all()
    np.testing.assert_array_equal(case.state.lru_slots, lru_before)
    np.testing.assert_array_equal(case.state.device_global_indices, dgi_before)


def test_ref_null_source_pages_resolve_to_minus_one() -> None:
    case = make_case(top_k=64, hot_size=128, num_rows=1, null_fraction=1.0, seed=5)
    out = _ref_step(case, case.state)
    assert (out.resolved_global_indices[0] == -1).all()
    assert (out.hot_indices[0] == -1).all()
    assert (out.attention_indices[0] == -1).all()
    assert out.valid_counts[0] == 0
    assert out.swap_counts[0] == 0


def test_ref_out_of_range_lru_slot_degrades_to_remiss() -> None:
    """The corruption tripwire must not become a hot-cache write."""
    case = make_case(top_k=8, hot_size=8, num_rows=1, seed=2)
    # Corrupt one slot out of [0, region_stride).
    case.state.lru_slots[0, 0] = np.int16(case.cfg.region_stride + 5)
    out = _ref_step(case, case.state)

    # The bad slot is oldest, so it is handed to the first miss, which must
    # resolve invalid rather than address the hot cache.
    assert (out.hot_indices[0] == -1).sum() >= 1
    bad = np.where(out.hot_indices[0] == -1)[0]
    for col in bad:
        assert out.miss_mask[0, col] == 0, "an invalid entry must not be a miss"
    # Ownership was never recorded for the bad slot.
    assert case.state.device_global_indices[0].max() < case.cfg.host_rows


def test_ref_eviction_is_oldest_first() -> None:
    """Misses take eviction slots in LRU (oldest-first) order."""
    case = make_case(top_k=4, hot_size=8, num_rows=1, seed=17)
    _ref_step(case, case.state)  # populate 4 of 8 slots

    # Fresh indices force 4 new misses; they must land on the 4 slots that
    # were never touched (still oldest), not on the just-loaded ones.
    warm_lru = case.state.lru_slots[0].copy()
    hits_at_mru = set(warm_lru[-4:].tolist())
    # Fresh in-range tokens (out-of-range ones would resolve to -1 in phase 1
    # and never reach the eviction path at all).
    used_tokens = set(case.global_indices[0].tolist())
    fresh = [t for t in range(case.top_k * 4) if t not in used_tokens][:4]
    case.global_indices = np.asarray([fresh], dtype=np.int32)
    out = _ref_step(case, case.state)

    assert out.swap_counts[0] == 4
    evicted = set(warm_lru[:4].tolist())
    used = set()
    for col in range(case.top_k):
        phys = int(out.hot_indices[0, col])
        used.add(phys % case.cfg.hot_block_size)
    assert used == evicted, "must evict the 4 oldest, not the 4 MRU"
    assert not (used & hits_at_mru)


# ---------------------------------------------------------------------------
# Differential tests: compiled op vs reference model
#
# These are the tests that matter for the AMD port. They require the kernels
# to be compiled; on a build without them they skip with a clear reason (or
# fail, under VLLM_HISPARSE_REQUIRE_OPS=1).
# ---------------------------------------------------------------------------


# Straddles both the 32-lane (NVIDIA warp) and 64-lane (CDNA wavefront)
# boundaries, which is where compaction/ballot bugs hide.
LANE_BOUNDARY_TOP_K = [1, 31, 32, 33, 63, 64, 65, 128]


@pytest.mark.parametrize("top_k", LANE_BOUNDARY_TOP_K)
@pytest.mark.parametrize("hot_multiple", [1, 2, 5])
def test_resolve_matches_reference(
    requires_hisparse_ops, top_k: int, hot_multiple: int
) -> None:
    """Cold-start resolve must match the oracle exactly at every lane width.

    ``hot_multiple`` covers the minimum legal hot buffer (== top_k), a decode
    sizing, and an MTP-like oversubscription.
    """
    case = make_case(
        top_k=top_k, hot_size=top_k * hot_multiple, seed=top_k * 10 + hot_multiple
    )
    ref_state = ResolveState(
        case.state.device_global_indices.copy(), case.state.lru_slots.copy()
    )
    want = _ref_step(case, ref_state)
    got = run_device_resolve(case, case.state)

    ctx = f"top_k={top_k} hot={top_k * hot_multiple}"
    assert_outputs_equal(got, want, ctx)
    assert_state_equal(case.state, ref_state, ctx)


@pytest.mark.parametrize("top_k", [32, 64, 65])
@pytest.mark.parametrize(
    "resident_fraction,null_fraction",
    [(0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.3, 0.3)],
    ids=["all-miss", "all-resident", "mixed", "half-null", "mixed-null"],
)
def test_resolve_residency_mix_matches_reference(
    requires_hisparse_ops, top_k: int, resident_fraction: float, null_fraction: float
) -> None:
    """Residency mixes, including the all-resident phase-1 early exit."""
    case = make_case(
        top_k=top_k,
        hot_size=2 * top_k,
        resident_fraction=resident_fraction,
        null_fraction=null_fraction,
        seed=int(top_k + 100 * resident_fraction + 17 * null_fraction),
    )
    ref_state = ResolveState(
        case.state.device_global_indices.copy(), case.state.lru_slots.copy()
    )
    want = _ref_step(case, ref_state)
    got = run_device_resolve(case, case.state)

    ctx = f"top_k={top_k} res={resident_fraction} null={null_fraction}"
    assert_outputs_equal(got, want, ctx)
    assert_state_equal(case.state, ref_state, ctx)


@pytest.mark.parametrize("top_k", [32, 64])
def test_resolve_padding_rows_match_reference(
    requires_hisparse_ops, top_k: int
) -> None:
    """CUDA-graph padding rows must publish -1 and leave state alone."""
    case = make_case(
        top_k=top_k, hot_size=2 * top_k, num_rows=4, padding_rows=(1, 3), seed=99
    )
    ref_state = ResolveState(
        case.state.device_global_indices.copy(), case.state.lru_slots.copy()
    )
    want = _ref_step(case, ref_state)
    got = run_device_resolve(case, case.state)

    assert_outputs_equal(got, want, f"top_k={top_k} padded")
    assert_state_equal(case.state, ref_state, f"top_k={top_k} padded")
    for row in (1, 3):
        assert (got.hot_indices[row] == -1).all()
        assert (got.attention_indices[row] == -1).all()


@pytest.mark.parametrize("top_k", [31, 32, 33, 64, 65])
def test_resolve_multi_step_matches_reference(
    requires_hisparse_ops, top_k: int
) -> None:
    """Twenty steps of drifting working sets, compared at every step.

    Single-step tests cannot catch LRU-order bugs by construction: the order
    only becomes observable through which slot a *later* step evicts. The
    token window drifts so each step mixes hits, misses and evictions.
    """
    num_steps = 20
    case = make_case(top_k=top_k, hot_size=2 * top_k, num_rows=2, seed=top_k)
    ref_state = ResolveState(
        case.state.device_global_indices.copy(), case.state.lru_slots.copy()
    )
    rng = np.random.default_rng(top_k)
    max_token = top_k * 4

    saw_hit = False
    saw_miss = False
    for step in range(num_steps):
        # Drift the window: keep roughly half of last step's tokens.
        case.global_indices = np.stack(
            [
                rng.choice(max_token, size=top_k, replace=False).astype(np.int32)
                for _ in range(case.num_rows)
            ]
        )
        want = _ref_step(case, ref_state)
        got = run_device_resolve(case, case.state)

        ctx = f"top_k={top_k} step={step}"
        assert_outputs_equal(got, want, ctx)
        assert_state_equal(case.state, ref_state, ctx)

        total_misses = int(want.swap_counts.sum())
        saw_miss |= total_misses > 0
        saw_hit |= total_misses < case.num_rows * top_k

    assert saw_miss, "test did not exercise the miss path"
    assert saw_hit, "test did not exercise the hit path"


@pytest.mark.parametrize("top_k", [32, 64])
def test_resolve_warm_replay_is_all_hits(requires_hisparse_ops, top_k: int) -> None:
    """Re-resolving the same top-K must hit everything and swap nothing."""
    case = make_case(top_k=top_k, hot_size=2 * top_k, num_rows=2, seed=4)
    first = run_device_resolve(case, case.state)
    assert first.swap_counts.sum() > 0

    dgi_before = case.state.device_global_indices.copy()
    second = run_device_resolve(case, case.state)
    assert (second.swap_counts == 0).all()
    np.testing.assert_array_equal(second.hot_indices, first.hot_indices)
    np.testing.assert_array_equal(case.state.device_global_indices, dgi_before)


def test_resolve_out_of_range_lru_slot_degrades_to_remiss(
    requires_hisparse_ops,
) -> None:
    """The corruption tripwire must survive the port.

    An out-of-range LRU slot must resolve as invalid rather than becoming an
    unbounded read or a hot-cache write. This is exactly the kind of defensive
    path a rewrite silently drops.
    """
    case = make_case(top_k=8, hot_size=8, num_rows=1, seed=2)
    case.state.lru_slots[0, 0] = np.int16(case.cfg.region_stride + 5)
    ref_state = ResolveState(
        case.state.device_global_indices.copy(), case.state.lru_slots.copy()
    )
    want = _ref_step(case, ref_state)
    got = run_device_resolve(case, case.state)

    assert_outputs_equal(got, want, "corrupt-lru")
    assert_state_equal(case.state, ref_state, "corrupt-lru")
    assert (got.hot_indices[0] == -1).any()


@pytest.mark.parametrize("top_k", [64, 2048])
def test_resolve_production_shapes(requires_hisparse_ops, top_k: int) -> None:
    """GLM 5.3 decode (hot=2*top_k) and MTP-3 (hot=5*top_k) sizings.

    At top_k=2048 these are the real shapes, and the ones whose shared-memory
    footprint must stay under CDNA's hard 64 KB LDS ceiling.
    """
    for hot_multiple in (2, 5):
        case = make_case(
            top_k=top_k,
            hot_size=top_k * hot_multiple,
            num_rows=1,
            host_rows=max(4096, top_k * 8),
            seed=top_k + hot_multiple,
        )
        ref_state = ResolveState(
            case.state.device_global_indices.copy(), case.state.lru_slots.copy()
        )
        want = _ref_step(case, ref_state)
        got = run_device_resolve(case, case.state)
        ctx = f"top_k={top_k} hot={top_k * hot_multiple}"
        assert_outputs_equal(got, want, ctx)
        assert_state_equal(case.state, ref_state, ctx)


# ---------------------------------------------------------------------------
# Data movement: gather_plan / gather_compact / invalidate_written_slots
# ---------------------------------------------------------------------------


def _byte_pattern(num_rows: int, row_bytes: int, seed: int) -> np.ndarray:
    """Distinct, non-repeating bytes per row so a misaddressed copy shows up."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(num_rows, row_bytes), dtype=np.uint8)


def _distinct_hot_indices(
    rng: np.random.Generator, num_rows: int, top_k: int, hot_rows: int
) -> np.ndarray:
    """Destination slots that are globally unique, padded with -1.

    The gather kernels parallelise over plan entries with no ordering between
    them, so two entries naming the same destination slot race and the winner
    is undefined. That is not a defect: the resolver only ever emits distinct
    slots (each miss is assigned its own evicted slot). Drawing destinations
    i.i.d. would collide and make the reference's sequential order the only
    "right" answer, which the kernel never promised. Unique slots keep the
    comparison byte-exact and still cover the -1 skip path.
    """
    total = num_rows * top_k
    num_live = min(hot_rows, total)
    flat = np.full(total, -1, dtype=np.int32)
    flat[:num_live] = rng.permutation(hot_rows)[:num_live]
    rng.shuffle(flat)
    return flat.reshape(num_rows, top_k)


@pytest.mark.parametrize("top_k", LANE_BOUNDARY_TOP_K)
@pytest.mark.parametrize("row_bytes", [16, FP8_DS_MLA_ROW_BYTES])
def test_gather_plan_matches_reference(
    requires_hisparse_ops, top_k: int, row_bytes: int
) -> None:
    """Byte-exact gather, including the zero-fill path for absent sources.

    ``row_bytes`` covers the minimum legal row and the real fp8_ds_mla row
    (656 B = 512 quantized NoPE + 16 scales + 128 RoPE). Both are multiples of
    16, which every op validates, so only the 16-byte vector path is reachable.
    """
    num_rows, host_rows = 2, 64
    hot_block_size = 8
    # Enough slots that every plan entry can get a distinct destination; see
    # _distinct_hot_indices for why destinations must not collide.
    hot_num_blocks = max(4, (num_rows * top_k + hot_block_size - 1) // hot_block_size)
    hot_rows = hot_block_size * hot_num_blocks
    rng = np.random.default_rng(top_k + row_bytes)

    host = _byte_pattern(host_rows, row_bytes, seed=top_k)
    hot = _byte_pattern(hot_rows, row_bytes, seed=top_k + 1).reshape(-1)

    # Mix of real rows, absent sources (g >= host_rows -> zero fill) and -1.
    global_indices = rng.integers(
        -1, host_rows + 8, size=(num_rows, top_k), dtype=np.int32
    )
    hot_indices = _distinct_hot_indices(rng, num_rows, top_k, hot_rows)
    miss_mask = rng.integers(0, 2, size=(num_rows, top_k), dtype=np.int32)
    request_state_indices = np.array([0, -1][:num_rows], dtype=np.int32)

    hot_block_stride = hot_block_size * row_bytes
    want_hot = hot.copy()
    want_attn = np.full((num_rows, top_k), POISON, dtype=np.int32)
    ref_gather_plan(
        host,
        want_hot,
        global_indices,
        hot_indices,
        miss_mask,
        request_state_indices,
        want_attn,
        hot_block_size,
        hot_block_size,
        hot_block_stride,
    )

    d_host = _pinned(torch.from_numpy(host.copy()))
    d_hot = torch.from_numpy(
        hot.copy().reshape(hot_num_blocks, hot_block_size, row_bytes)
    ).to(DEVICE)
    d_attn = torch.from_numpy(np.full((num_rows, top_k), POISON, dtype=np.int32)).to(
        DEVICE
    )

    torch.ops._C_cache_ops.hisparse_gather_plan(
        d_host,
        d_hot,
        torch.from_numpy(global_indices).to(DEVICE),
        torch.from_numpy(hot_indices).to(DEVICE),
        torch.from_numpy(miss_mask).to(DEVICE),
        torch.from_numpy(request_state_indices).to(DEVICE),
        d_attn,
        hot_block_size,
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(
        d_hot.cpu().numpy().reshape(-1),
        want_hot,
        err_msg=f"gather_plan bytes (top_k={top_k} row_bytes={row_bytes})",
    )
    np.testing.assert_array_equal(
        d_attn.cpu().numpy(), want_attn, err_msg="gather_plan attention_indices"
    )


@pytest.mark.parametrize("top_k", [31, 32, 64, 65])
def test_gather_compact_matches_reference(requires_hisparse_ops, top_k: int) -> None:
    """Compact gather copies exactly miss_count rows and no more."""
    num_rows, host_rows = 3, 64
    row_bytes, hot_block_size = FP8_DS_MLA_ROW_BYTES, 8
    # Enough slots that every plan entry can get a distinct destination; see
    # _distinct_hot_indices for why destinations must not collide.
    hot_num_blocks = max(4, (num_rows * top_k + hot_block_size - 1) // hot_block_size)
    hot_rows = hot_block_size * hot_num_blocks
    rng = np.random.default_rng(top_k)

    host = _byte_pattern(host_rows, row_bytes, seed=top_k + 2)
    hot = _byte_pattern(hot_rows, row_bytes, seed=top_k + 3).reshape(-1)

    miss_global = rng.integers(
        -1, host_rows + 4, size=(num_rows, top_k), dtype=np.int32
    )
    miss_hot = _distinct_hot_indices(rng, num_rows, top_k, hot_rows)
    # Include 0, a mid count, and an over-large count that must clamp to top_k.
    miss_counts = np.array([0, top_k // 2, top_k + 5][:num_rows], dtype=np.int32)

    hot_block_stride = hot_block_size * row_bytes
    want_hot = hot.copy()
    ref_gather_compact(
        host,
        want_hot,
        miss_global,
        miss_hot,
        miss_counts,
        hot_block_size,
        hot_block_stride,
    )

    d_hot = torch.from_numpy(
        hot.copy().reshape(hot_num_blocks, hot_block_size, row_bytes)
    ).to(DEVICE)
    torch.ops._C_cache_ops.hisparse_gather_compact(
        _pinned(torch.from_numpy(host.copy())),
        d_hot,
        torch.from_numpy(miss_global).to(DEVICE),
        torch.from_numpy(miss_hot).to(DEVICE),
        torch.from_numpy(miss_counts).to(DEVICE),
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(
        d_hot.cpu().numpy().reshape(-1),
        want_hot,
        err_msg=f"gather_compact bytes (top_k={top_k})",
    )


@pytest.mark.parametrize("region_stride", [31, 32, 64, 65, 256])
def test_invalidate_written_slots_matches_reference(
    requires_hisparse_ops, region_stride: int
) -> None:
    """Every copy of a rewritten slot must be invalidated, in every row."""
    num_states, num_reqs, num_tokens = 4, 4, 16
    rng = np.random.default_rng(region_stride)

    dgi = rng.integers(-1, 40, size=(num_states, region_stride), dtype=np.int32)
    # Plant duplicates: one slot id appearing several times in a row must be
    # cleared everywhere, not just at its first occurrence.
    dgi[0, : min(5, region_stride)] = 7
    request_state_indices = np.array([0, 1, -1, 9], dtype=np.int32)
    req_id_per_token = rng.integers(-2, num_reqs + 2, size=num_tokens, dtype=np.int32)
    written_slots = rng.integers(-1, 40, size=num_tokens, dtype=np.int64)
    written_slots[0] = 7

    want = dgi.copy()
    ref_invalidate_written_slots(
        want, request_state_indices, req_id_per_token, written_slots
    )

    d_dgi = torch.from_numpy(dgi.copy()).to(DEVICE)
    torch.ops._C_cache_ops.hisparse_invalidate_written_slots(
        d_dgi,
        torch.from_numpy(request_state_indices).to(DEVICE),
        torch.from_numpy(req_id_per_token).to(DEVICE),
        torch.from_numpy(written_slots).to(DEVICE),
    )
    torch.cuda.synchronize()

    np.testing.assert_array_equal(
        d_dgi.cpu().numpy(),
        want,
        err_msg=f"invalidate_written_slots (region_stride={region_stride})",
    )
