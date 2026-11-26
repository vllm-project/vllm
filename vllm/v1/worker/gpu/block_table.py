# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.grant_table import NULL_BLOCK_ID, NULL_GEN, normalize_gen_updates
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter,
    StagedWriteTensor,
    UvaBackedTensor,
    UvaBufferPool,
    _load_ptr,
)

logger = init_logger(__name__)


def _as_i32(gens: Sequence[int] | np.ndarray) -> np.ndarray:
    """u32 gens (Python ints or numpy) as their int32 bit pattern."""
    return np.asarray(gens, dtype=np.int64).astype(np.uint32).view(np.int32)


class PercyGenCheck:
    """Percy W2: the device generation table and the step's write windows.

    `table[block]` mirrors the GrantTable gen of every block (int32 bit
    pattern; the null block holds NULL_GEN, which no grant carries). Each
    step, `window_meta[group, req_state_idx]` = (first column, length,
    offset into `window_gens`) describes the columns a scheduled request may
    write and the gens they must hold; length -1 leaves a request unchecked
    and -2 refuses all its writes. All device buffers keep fixed addresses:
    the slot-mapping kernel is captured in CUDA graphs (draft decode).

    Args:
        num_kv_blocks: Blocks in the main KV pool.
        block_sizes: Tokens per block of each KV cache group.
        max_num_reqs: Request-state slots (rows of the block tables).
        max_num_batched_tokens: Tokens per step, which bounds the windows.
        device: Device of the tables.
        max_window_extra_tokens: Positions a window may span beyond a
            request's scheduled tokens: async spec-decode slack before them
            plus drafter lookahead after them.

    """

    UNCHECKED = -1
    REFUSED = -2

    def __init__(
        self,
        num_kv_blocks: int,
        block_sizes: Sequence[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        device: torch.device,
        max_window_extra_tokens: int = 0,
    ) -> None:
        self.device = device
        self.num_groups = len(block_sizes)
        self.table = torch.ones(num_kv_blocks, dtype=torch.int32, device=device)
        self.table[NULL_BLOCK_ID] = int(_as_i32([NULL_GEN])[0])
        self.meta_np = np.zeros((self.num_groups, max_num_reqs, 3), dtype=np.int32)
        self.meta_np[..., 1] = self.UNCHECKED
        self.meta = torch.from_numpy(self.meta_np).to(device)
        # A request's window covers positions [computed - slack, computed +
        # scheduled + lookahead): at most cdiv(scheduled + extra, bs) + 1
        # columns, and cdiv(s + x, bs) <= s / bs + cdiv(x, bs) + 1. Summed
        # over requests whose scheduled tokens total at most
        # max_num_batched_tokens, a group needs at most
        # cdiv(max_num_batched_tokens, bs) + max_num_reqs * (cdiv(x, bs) + 2).
        capacity = sum(
            cdiv(max_num_batched_tokens, bs)
            + max_num_reqs * (cdiv(max_window_extra_tokens, bs) + 2)
            for bs in block_sizes
        )
        self.gens_np = np.zeros(max(capacity, 1), dtype=np.int32)
        self.gens = torch.zeros(self.gens_np.shape[0], dtype=torch.int32, device=device)
        self._num_gens = 0
        self._staged = device.type != "cpu"
        if self._staged:
            self._meta_pool = UvaBufferPool(self.meta_np.size, torch.int32)
            self._gens_pool = UvaBufferPool(self.gens_np.shape[0], torch.int32)
            self._ids_pool = UvaBufferPool(1024, torch.int64)
            self._vals_pool = UvaBufferPool(1024, torch.int32)

    def _to_device(
        self, pool: "UvaBufferPool | None", x: np.ndarray, out: torch.Tensor
    ) -> torch.Tensor:
        if pool is None:
            return out.copy_(torch.from_numpy(x))
        # Staged host buffer (round-robin, retired before reuse), then an
        # in-stream copy into the fixed device buffer.
        return pool.copy_to_gpu(x, out=out)

    def apply_gen_updates(
        self, gen_updates: tuple[list[int], list[int]] | None
    ) -> None:
        """Copy the step's gen bumps into the device table (G1, I10).

        Ordering: this runs in `execute_model` on the compute stream, after
        the step's row updates and before any slot mapping of the step is
        launched; every reader of `table` (the slot-mapping kernel, also when
        speculators launch it later in the step or replay it inside a CUDA
        graph) runs on that same stream. So the update is ordered after every
        earlier step's reads (a running step never sees it) and before every
        read of this step. The H2D source is cloned into device memory first:
        no kernel reads pinned host memory.
        """
        if not gen_updates or not gen_updates[0]:
            return
        ids, gens, dropped = normalize_gen_updates(
            gen_updates[0], gen_updates[1], self.table.shape[0]
        )
        if dropped:
            logger.error(
                "Percy: dropped %d gen updates for the null block or out-of-"
                "range block ids",
                dropped,
            )
        if not ids.size:
            return
        vals = gens.view(np.int32)
        n = ids.shape[0]
        if self._staged:
            ids_d = self._ids_pool.copy_to_gpu(ids)
            vals_d = self._vals_pool.copy_to_gpu(vals)
        else:
            ids_d = torch.from_numpy(ids)
            vals_d = torch.from_numpy(vals.copy())
        self.table.index_copy_(0, ids_d[:n], vals_d[:n])

    @property
    def table_bytes_per_block(self) -> int:
        return self.table.element_size()

    def mismatched_blocks(self, gen_mirror: np.ndarray) -> np.ndarray:
        """Debug/test only (synchronizes): block ids whose device gen differs
        from a CPU mirror of the same gens (e.g. `RowVerifier.gen_mirror`)."""
        device_gens = self.table.cpu().numpy().view(np.uint32)
        return np.flatnonzero(device_gens != np.asarray(gen_mirror, dtype=np.uint32))

    def begin_windows(self) -> None:
        """Start a step: every request is unchecked until given a window."""
        self.meta_np[..., 1] = self.UNCHECKED
        self._num_gens = 0

    def set_windows(
        self,
        req_state_idx: int,
        windows: "Sequence[Sequence[int] | None]",
        refused: bool = False,
    ) -> None:
        """Record a scheduled request's write windows, one per group: a
        window is (first column, gen, gen, ...); None leaves the group
        unchecked; an empty `windows` leaves the request unchecked."""
        meta = self.meta_np[:, req_state_idx]
        if refused:
            meta[:, 1] = self.REFUSED
            return
        if not windows:
            return
        if len(windows) != self.num_groups:
            logger.warning_once(
                "Percy: write windows for %d groups, expected %d; unchecked",
                len(windows),
                self.num_groups,
            )
            return
        for group, window in enumerate(windows):
            if window is None:
                continue
            n = len(window) - 1
            start = self._num_gens
            if start + n > self.gens_np.shape[0]:
                # The capacity is a proven bound (see __init__), and growing
                # would move a graph-captured buffer: a broken bound is a bug.
                raise RuntimeError(
                    f"Percy write windows need {start + n} gens, above the "
                    f"bounded capacity {self.gens_np.shape[0]} (request slot "
                    f"{req_state_idx}, group {group}, window of {n} columns)"
                )
            self.gens_np[start : start + n] = _as_i32(window[1:])
            meta[group] = (window[0], n, start)
            self._num_gens = start + n

    def commit_windows(self) -> None:
        """Copy the step's windows to the device buffers."""
        self._to_device(
            self._meta_pool if self._staged else None,
            self.meta_np.reshape(-1),
            self.meta.view(-1),
        )
        n = self._num_gens
        if n:
            self._to_device(
                self._gens_pool if self._staged else None,
                self.gens_np[:n],
                self.gens[:n],
            )


def make_slot_mappings(
    num_groups: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    """Allocate a [num_groups, num_tokens] int64 slot-mapping buffer.

    Each group's row is handed to attention as its own tensor, and some
    kernels assume an aligned base address (DeepSeek-V4's CuTe DSL compressor
    requires 64 bytes). Pad the row stride so every row starts 128-byte
    aligned whatever num_tokens is.
    """
    buffer = torch.zeros(
        num_groups, round_up(num_tokens, 16), dtype=torch.int64, device=device
    )
    return buffer[:, :num_tokens]


class BlockTables:
    # Set by the elastic EP warmup so KV writes land in the null block.
    redirect_writes_to_null_block = False

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
        slot_mapping_enabled: list[bool] | None = None,
        dcp_sharded: list[bool] | None = None,
        sliding_window_reach: list[int] | None = None,
        num_kv_blocks: int | None = None,
        max_window_extra_tokens: int = 0,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        # Percy W2: the device gen table and write windows. The slot-mapping
        # check addresses block-table columns by token position, which only
        # holds without context parallelism.
        self.gen_check: PercyGenCheck | None = None
        if num_kv_blocks is not None and block_sizes:
            self.gen_check = PercyGenCheck(
                num_kv_blocks,
                block_sizes,
                max_num_reqs,
                max_num_batched_tokens,
                device,
                max_window_extra_tokens,
            )
            if cp_size > 1:
                logger.warning_once(
                    "Percy: the slot_mapping generation check is disabled under "
                    "context parallelism (cp_size=%d)",
                    cp_size,
                )
        self.gen_check_enabled = self.gen_check is not None and cp_size == 1

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        if slot_mapping_enabled is None:
            slot_mapping_enabled = [True] * self.num_kv_cache_groups
        assert len(slot_mapping_enabled) == self.num_kv_cache_groups
        self._slot_mapping_enabled = slot_mapping_enabled
        if dcp_sharded is None:
            dcp_sharded = [True] * self.num_kv_cache_groups
        assert len(dcp_sharded) == self.num_kv_cache_groups
        self.dcp_sharded = torch.tensor(dcp_sharded, dtype=torch.bool, device=device)
        # Per group, how many tokens below a request's first scheduled position
        # attention can still read; 0 disables it for the group. Under context
        # parallelism a block-table column does not map to a contiguous token
        # range, so the masking in gather_block_tables is disabled.
        if sliding_window_reach is None or cp_size > 1:
            sliding_window_reach = [0] * self.num_kv_cache_groups
        assert len(sliding_window_reach) == self.num_kv_cache_groups
        self._sliding_window_reach = sliding_window_reach
        self.has_sliding_window_groups = any(r > 0 for r in sliding_window_reach)

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for max_num_blocks in max_num_blocks_per_group:
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)

        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs), dtype=torch.int32
        )
        self.fused_writer: FusedStagedWriter | None = None
        if self.num_kv_cache_groups > 1:
            # Only the multi-group path uses the fused writer.
            self.fused_writer = FusedStagedWriter(
                self.device, self.num_kv_cache_groups * self.max_num_reqs
            )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]
        self._num_dirty_rows = 0

        self.slot_mappings = make_slot_mappings(
            self.num_kv_cache_groups, self.max_num_batched_tokens, self.device
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.slot_mapping_enabled = torch.tensor(
            self._slot_mapping_enabled, dtype=torch.bool, device=self.device
        )
        self.sliding_window_reach = torch.tensor(
            self._sliding_window_reach, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            self.write_row(req_index, i, start, new_block_ids[i])

    def write_row(
        self, req_index: int, group: int, start: int, block_ids: list[int]
    ) -> None:
        """Stage `block_ids` into the row from column `start`, which becomes
        the row's end. Percy W1 merges a step's revokes and appends for a row
        into this one write: staged writes run in parallel, one per row."""
        if self.redirect_writes_to_null_block:
            block_ids = [0] * len(block_ids)
        end = start + len(block_ids)
        row_capacity = self.block_tables[group].gpu.shape[1]
        if end > row_capacity:
            raise RuntimeError(
                f"Block table write for request {req_index}, group {group} exceeds "
                f"row capacity ({end} > {row_capacity})"
            )
        self.block_tables[group].stage_write(req_index, start, block_ids)
        self.num_blocks.np[group, req_index] = end

    def apply_staged_writes(self) -> None:
        if self.num_kv_cache_groups == 0:
            return
        if self.num_kv_cache_groups == 1:
            # Single group: write directly, skipping the per-write group lookup.
            self.block_tables[0].apply_write()
        elif self.num_kv_cache_groups > 1:
            # Multiple groups: apply all block tables with one fused kernel.
            assert self.fused_writer is not None
            self.fused_writer.apply(
                self.block_tables, self.block_table_ptrs, self.block_table_strides
            )
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Gather the batch's block-table rows into the forward-pass tables.

        When num_computed_tokens ([max_num_reqs], indexed like the block tables)
        is given, sliding-window groups get every entry wholly below the window
        of the request's first scheduled token replaced by the null block. Those
        blocks may already have been freed by the sliding-window manager and
        reused, while attention kernels can still load them as part of a KV
        tile that straddles the window start: masked-out NaN in such a tile
        propagates into the output.
        """
        if self.num_kv_cache_groups == 0:
            return ()
        mask_sliding_window = (
            self.has_sliding_window_groups and num_computed_tokens is not None
        )
        if not mask_sliding_window:
            # Unused by the kernel; any int32 tensor satisfies the signature.
            num_computed_tokens = self.sliding_window_reach
        num_reqs = idx_mapping.shape[0]
        num_rows = num_reqs_padded
        if out is None:
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs
            num_rows = max(num_rows, self._num_dirty_rows)
            self._num_dirty_rows = num_reqs
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups
        # Also clear rows left by a larger batch in the shared buffer.
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_rows)](
            idx_mapping,
            self.block_table_ptrs,
            out_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            num_reqs,
            num_computed_tokens,
            self.sliding_window_reach,
            self.block_sizes_tensor,
            MASK_SLIDING_WINDOW=mask_sliding_window,
            BLOCK_SIZE=1024,  # type: ignore
        )
        return tuple(bt[:num_reqs_padded] for bt in out)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        #
        # Zero the rows so dummy runs write mamba state to the reserved null
        # block rather than through the previous real step's (stale) block
        # ids, which may point at blocks since freed and reallocated.
        num_rows = max(num_reqs, self._num_dirty_rows)
        for block_table in self.input_block_tables:
            block_table[:num_rows].zero_()
        # Profiling may populate the returned rows with valid block IDs.
        self._num_dirty_rows = num_reqs
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def clear_input_block_tables(self) -> None:
        for block_table in self.input_block_tables:
            block_table.zero_()
        self._num_dirty_rows = 0

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_kv_cache_groups == 0:
            return (self.slot_mappings if out is None else out)[:, :num_tokens_padded]
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        slot_mappings = self.slot_mappings if out is None else out
        gen_check = self.gen_check
        if self.gen_check_enabled:
            assert gen_check is not None
            gen_args = (
                gen_check.table,
                gen_check.meta,
                gen_check.meta.stride(0),
                gen_check.gens,
            )
        else:
            # Unused by the kernel; any int32 tensor satisfies the signature.
            unused = self.sliding_window_reach
            gen_args = (unused, unused, 0, unused)
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            slot_mappings.shape[1],
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mapping_enabled,
            self.dcp_sharded,
            slot_mappings,
            slot_mappings.stride(0),
            self.cp_rank,
            *gen_args,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
            GEN_CHECK=self.gen_check_enabled,
        )
        return slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    num_computed_tokens_ptr,  # [max_num_reqs]
    sliding_window_reach_ptr,  # [num_kv_cache_groups]
    block_sizes_ptr,  # [num_kv_cache_groups]
    MASK_SLIDING_WINDOW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    max_num_blocks = stride  # stride equals max_num_blocks for this group.
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    # Leading entries no query of this step can attend: null them (see
    # BlockTables.gather_block_tables).
    num_null = 0
    if MASK_SLIDING_WINDOW:
        reach = tl.load(sliding_window_reach_ptr + group_id)
        first_pos = tl.load(num_computed_tokens_ptr + req_idx)
        block_size = tl.load(block_sizes_ptr + group_id)
        num_null = tl.where(
            reach > 0, tl.maximum(first_pos - reach, 0) // block_size, 0
        )

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(
            src_row_ptr + offset,
            mask=(offset >= num_null) & (offset < num_blocks),
            other=0,
        )
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mapping_enabled,  # [num_kv_cache_groups]
    dcp_sharded,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    gen_table,  # [num_kv_blocks] int32 (u32 bit pattern)
    window_meta,  # [num_kv_cache_groups, max_num_reqs, 3] int32
    window_meta_group_stride,
    window_gens,  # [capacity] int32
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    GEN_CHECK: tl.constexpr = False,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    kv_block_size = tl.load(block_sizes + group_id)
    mapping_enabled = tl.load(slot_mapping_enabled + group_id)
    if CP_SIZE != 1:
        sharded = tl.load(dcp_sharded + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    # idx_mapping == -1 marks a dummy (or CUDA-graph padding) request that owns
    # no blocks: never read its block-table row and emit PAD for its tokens.
    is_real_req = req_state_idx >= 0
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    if GEN_CHECK:
        # Percy W2: this request's write window in this group, as
        # (first column, length, offset into window_gens). Length -1 means
        # unchecked; -2 a refused request (every token PADs).
        meta = (
            window_meta
            + group_id * window_meta_group_stride
            + tl.maximum(req_state_idx, 0) * 3
        )
        win_first = tl.load(meta, mask=is_real_req, other=0)
        win_len = tl.load(meta + 1, mask=is_real_req, other=-1)
        win_off = tl.load(meta + 2, mask=is_real_req, other=0)
        checked = is_real_req & (win_len != -1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            local_positions = positions
            is_local = True
        else:
            # Context parallelism is used.
            virtual_block_size = kv_block_size * CP_SIZE
            virtual_block_indices = positions // virtual_block_size
            virtual_block_offsets = positions % virtual_block_size
            is_local = virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = virtual_block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            local_positions = virtual_block_indices * kv_block_size + local_offsets
            local_positions = tl.where(sharded, local_positions, positions)
            is_local = ~sharded | is_local

        block_indices = tl.where(mapping_enabled, local_positions // kv_block_size, 0)
        block_offsets = local_positions % kv_block_size
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_local & is_real_req,
            other=0,
        )
        slot_ids = block_numbers * kv_block_size + block_offsets
        if CP_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        if GEN_CHECK:
            # Percy W2 (I6): a written block must hold the gen the scheduler
            # granted for its column; the null block's gen is never granted.
            in_token = offset < end_idx
            w = block_indices - win_first
            in_window = (w >= 0) & (w < win_len)
            # Broadcast the per-request scalar with `where`, not `&`, so the
            # masks stay boolean vectors (also under the Triton interpreter).
            load_mask = tl.where(checked, in_token & in_window, False)
            expected = tl.load(window_gens + win_off + w, mask=load_mask, other=0)
            actual = tl.load(gen_table + block_numbers, mask=load_mask, other=1)
            # -1 is NULL_GEN's int32 bit pattern: no window may claim it, so
            # a write aimed at the null block always fails.
            ok = in_window & (expected == actual) & (expected != -1)
            slot_ids = tl.where(tl.where(checked, ~ok, False), PAD_ID, slot_ids)

        slot_ids = tl.where(mapping_enabled & is_real_req, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
