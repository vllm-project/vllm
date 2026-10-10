# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched sliding-window metadata for DeepSeek-V4.1.

The KC cache layout puts every sliding-window cache in its own KV cache group
(a group may hold only one 64-token window page and still pack the pool block),
and each group builds its own attention metadata once per step. Almost none of
that metadata is per group: the window bounds of a token are a function of
positions, sequence lengths and the replay start, which every group shares. The
only per-group input is the group's paged block table.

So one kernel can serve every group: pass the block tables as a pointer array
and let the second grid axis pick the group. That turns 2 launches plus ~40 us
of Python per group into 2 launches per step plus a view per group, which is
what makes the one-group-per-layer packing affordable.

``SWAWindowMetadata`` owns the buffers for one engine and is refreshed once per
step by the model state (which has every group's block table) and the first
sliding-window builder to run (which has the batch's shared inputs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from vllm.v1.attention.backend import CommonAttentionMetadata

# One token per program, looping over the index width in chunks of this many.
_TRITON_BLOCK_SIZE = 128


def split_batch_counts(
    cm: CommonAttentionMetadata, decode_threshold: int
) -> tuple[int, int, int, int]:
    """(num_decodes, num_prefiles, num_decode_tokens, num_prefill_tokens)."""
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    return split_decodes_and_prefills(cm, decode_threshold=decode_threshold)


@triton.jit
def _batched_swa_indices_and_lens_kernel(
    table_ptrs,
    indices_ptr,
    indices_width,
    lens_ptr,
    rows,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_stride,
    block_size,
    replay_start_ptr,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    """Window slots and lengths for one token of one group.

    ``table_ptrs`` addresses the group's block table (``[num_reqs, max_blocks]``
    int32, ``block_table_stride`` columns apart), so the groups need no
    contiguous layout. ``indices`` is ``[num_groups, rows, indices_width]`` and
    ``lens`` ``[num_groups, rows]``; row ``pid`` of group ``g`` lands at
    ``g * rows + pid``.
    """
    group = tl.program_id(1)
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    row = group * rows + pid
    table = tl.load(table_ptrs + group).to(tl.pointer_type(tl.int32))

    if not tl.load(is_valid_token_ptr + token_idx):
        tl.store(lens_ptr + row, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, indices_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                indices_ptr + row * indices_width + offset,
                -1,
                mask=offset < indices_width,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_len = tl.load(query_start_loc_ptr + req_idx + 1) - query_start
    pos = tl.load(seq_lens_ptr + req_idx) - query_len + token_idx - query_start

    start_pos = tl.maximum(pos - (window_size - 1), 0)
    # SWA bounded replay: no window KV exists below the request's replay start.
    start_pos = tl.maximum(start_pos, tl.load(replay_start_ptr + req_idx))
    end_pos = pos + 1
    swa_len = end_pos - start_pos
    tl.store(lens_ptr + row, swa_len)

    for i in range(0, indices_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        pos_offset = start_pos + offset
        block_numbers = tl.load(
            table + req_idx * block_table_stride + pos_offset // block_size,
            mask=pos_offset < end_pos,
        )
        slot_ids = block_numbers * block_size + pos_offset % block_size
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            indices_ptr + row * indices_width + offset,
            slot_ids,
            mask=offset < indices_width,
        )


class SWAWindowMetadata:
    """Per-step sliding-window metadata for every group, built once.

    Buffers are laid out ``[num_groups, rows, width]`` so each group gets a view
    instead of its own tensors.
    """

    def __init__(
        self,
        device: torch.device,
        num_groups: int,
        decode_rows: int,
        prefill_rows: int,
        window_size: int,
        index_width: int,
    ) -> None:
        self.device = device
        self.num_groups = num_groups
        self.window_size = window_size
        self.index_width = index_width
        self.decode_rows = decode_rows
        self.prefill_rows = prefill_rows
        # [groups, rows, 1, width] keeps the rows the attention kernels address
        # (their window rows carry a singleton head axis) and stays dense.
        self.decode_indices = torch.zeros(
            num_groups, decode_rows, 1, index_width, dtype=torch.int32, device=device
        )
        self.decode_lens = torch.zeros(
            num_groups, decode_rows, dtype=torch.int32, device=device
        )
        self.prefill_indices = (
            torch.zeros(
                num_groups,
                prefill_rows,
                1,
                index_width,
                dtype=torch.int32,
                device=device,
            )
            if prefill_rows
            else None
        )
        self.prefill_lens = (
            torch.zeros(num_groups, prefill_rows, dtype=torch.int32, device=device)
            if prefill_rows
            else None
        )
        # Set by the model state each step, consumed by the first builder.
        self.pending = False
        self.ready = False
        self.group_of_layer: dict[str, int] = {}
        self._token_scratch_buffer = torch.empty(
            max(prefill_rows, decode_rows), dtype=torch.int32, device=device
        )
        # Replay-start stand-in for batches nothing replays: keeps a stable
        # address for graphs that read the metadata in place.
        self._no_replay = torch.zeros(
            max(decode_rows, 1), dtype=torch.int32, device=device
        )
        # Filled in place every step: graphs read the metadata tensors where
        # they stand, so their addresses have to survive.
        self._is_valid_token = torch.zeros(
            max(prefill_rows, decode_rows), dtype=torch.bool, device=device
        )
        self._table_ptrs: torch.Tensor | None = None
        self._table_stride = 0
        self._group_tables: list[torch.Tensor] = []
        # Shared step inputs, filled by the first builder to run.
        self.query_start_loc: torch.Tensor | None = None
        self.query_start_loc_cpu: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.seq_lens_cpu: torch.Tensor | None = None
        self.token_to_req_indices: torch.Tensor | None = None
        self.is_valid_token: torch.Tensor | None = None
        self.replay_start: torch.Tensor | None = None
        self.num_decodes = 0
        self.num_prefiles = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.max_decode_query_len = 0
        self.prefill_metadata: dict[str, object] = {}

    # ---- refreshing -----------------------------------------------------
    def observe(
        self,
        layer_groups: dict[str, tuple[int, torch.Tensor]],
        replay_start: torch.Tensor | None = None,
    ) -> bool:
        """Point at this step's group block tables; invalidate the built rows.

        ``layer_groups`` maps a layer name to (group slot, block table) for every
        sliding-window group, in the order the groups are visited.
        """
        self.replay_start = (
            replay_start if replay_start is not None else self._no_replay
        )
        tables = []
        self.group_of_layer = {}
        for layer_name, (slot, table) in layer_groups.items():
            self.group_of_layer[layer_name] = slot
            tables.append(table)
        if len(tables) != self.num_groups:
            return False
        self._group_tables = tables
        # The tables are persistent buffers, so the pointer array only has to be
        # rebuilt when an address or a stride actually moves.
        stride = tables[0].stride(0)
        if stride != tables[0].shape[1]:
            return False
        addresses = [t.data_ptr() for t in tables]
        if (
            self._table_ptrs is None
            or self._table_stride != stride
            or self._addresses != addresses
        ):
            self._table_ptrs = torch.tensor(
                addresses, dtype=torch.int64, device=self.device
            )
            self._addresses = addresses
            self._table_stride = stride
        self.pending = True
        self.ready = False
        return True

    # ---- building -------------------------------------------------------
    def prepare(
        self,
        cm: CommonAttentionMetadata,
        block_size: int,
        decode_threshold: int,
        prefill_metadata: dict[str, object],
    ) -> None:
        """Build every group's window rows from the batch's shared inputs."""
        (
            num_decodes,
            num_prefiles,
            num_decode_tokens,
            num_prefill_tokens,
        ) = split_batch_counts(cm, decode_threshold)

        slot_mapping = cm.slot_mapping
        is_valid_token = self._is_valid_token[: slot_mapping.shape[0]]
        torch.ge(slot_mapping, 0, out=is_valid_token)
        token_to_req_indices = cm.token_to_req_indices(
            self._token_scratch_buffer[: _token_scratch_size(cm)]
        )
        replay_start = (
            self.replay_start if self.replay_start is not None else self._no_replay
        )
        self.replay_start = replay_start

        self.seq_lens = cm.seq_lens
        self.query_start_loc = cm.query_start_loc
        self.query_start_loc_cpu = cm.query_start_loc_cpu
        self.seq_lens_cpu = cm.seq_lens_cpu_upper_bound
        self.token_to_req_indices = token_to_req_indices
        self.is_valid_token = is_valid_token
        self.num_decodes = num_decodes
        self.num_prefiles = num_prefiles
        self.num_decode_tokens = num_decode_tokens
        self.num_prefill_tokens = num_prefill_tokens
        self.max_decode_query_len = min(cm.max_query_len, decode_threshold)
        self.prefill_metadata = prefill_metadata

        assert self._table_ptrs is not None
        # The rows are one stacked buffer per group, so a batch wider than the
        # buffers would spill into the next group instead of failing.
        assert num_decode_tokens <= self.decode_rows, (
            f"{num_decode_tokens} decode tokens exceed the {self.decode_rows} rows "
            "reserved for them"
        )
        assert num_prefill_tokens <= self.prefill_rows, (
            f"{num_prefill_tokens} prefill tokens exceed the {self.prefill_rows} rows "
            "reserved for them"
        )
        if num_decode_tokens:
            self._launch(
                self.decode_indices,
                self.decode_lens,
                token_offset=0,
                num_tokens=num_decode_tokens,
                block_size=block_size,
                rows=self.decode_rows,
            )
        if num_prefill_tokens:
            self._launch(
                self.prefill_indices,
                self.prefill_lens,
                token_offset=num_decode_tokens,
                num_tokens=num_prefill_tokens,
                block_size=block_size,
                rows=self.prefill_rows,
            )
        self.pending = False
        self.ready = True

    def _launch(
        self,
        indices: torch.Tensor | None,
        lens: torch.Tensor | None,
        *,
        token_offset: int,
        num_tokens: int,
        block_size: int,
        rows: int,
    ) -> None:
        assert indices is not None and lens is not None
        _batched_swa_indices_and_lens_kernel[(num_tokens, self.num_groups)](
            self._table_ptrs,
            indices,
            self.index_width,
            lens,
            rows,
            self.window_size,
            self.query_start_loc,
            self.seq_lens,
            self.token_to_req_indices,
            self.is_valid_token,
            self._table_stride,
            block_size,
            self.replay_start,
            token_offset,
            TRITON_BLOCK_SIZE=_TRITON_BLOCK_SIZE,
        )

    # ---- consuming ------------------------------------------------------
    def rows_for_slot(self, group: int) -> dict[str, object]:
        """This group's window rows plus the batch inputs shared by all groups."""
        return {
            "decode_swa_indices": self.decode_indices[group, : self.num_decode_tokens],
            "decode_swa_lens": self.decode_lens[group, : self.num_decode_tokens],
            "prefill_swa_indices": (
                self.prefill_indices[group, : self.num_prefill_tokens]
                if self.num_prefill_tokens
                else None
            ),
            "prefill_swa_lens": (
                self.prefill_lens[group, : self.num_prefill_tokens]
                if self.num_prefill_tokens
                else None
            ),
            "seq_lens": self.seq_lens,
            "query_start_loc": self.query_start_loc,
            "query_start_loc_cpu": self.query_start_loc_cpu,
            "token_to_req_indices": self.token_to_req_indices,
            "is_valid_token": self.is_valid_token,
            "replay_start": self.replay_start,
            "num_decodes": self.num_decodes,
            "num_prefills": self.num_prefiles,
            "num_decode_tokens": self.num_decode_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "max_decode_query_len": self.max_decode_query_len,
            "decode_swa_width": self.index_width,
            **self.prefill_metadata,
        }


def _token_scratch_size(cm: CommonAttentionMetadata) -> int:
    """Rows the per-token request mapping needs for this batch."""
    return max(cm.num_actual_tokens, int(cm.query_start_loc_cpu[-1]))
