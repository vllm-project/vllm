# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch equivalents of the align-mode mamba state migration.

The kernels here address state tensors and block tables through arrays of
device addresses, so they are replaced at the method level and the context is
made to keep the tensors themselves, which ``_populate_metadata`` otherwise
reduces to ``data_ptr()`` values.

Per-step work stays vectorized: a boundary crossing is rare, so the masks below
usually select nothing and the Python loop never runs.
"""

from typing import Any

import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncsByType,
    is_conv_state_dim_first,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    _get_mamba_spec_for_layer,
)

_ORIGINAL_POPULATE_METADATA = MambaSpecDecodeGPUContext._populate_metadata


def _full_rows(block_table: torch.Tensor) -> torch.Tensor:
    """Restore the rows a batch-order slice hides.

    The caller passes slices of the persistent block tables, sized to the batch
    that happened to be running at capture time. The kernels reach later rows
    anyway, since they address the buffer by pointer and stride; indexing the
    slice would instead raise once a second request needs a copy.
    """
    row_stride = block_table.stride(0)
    if row_stride == 0:
        return block_table
    elements = block_table.untyped_storage().nbytes() // block_table.element_size()
    rows = elements // row_stride
    if rows <= block_table.size(0):
        return block_table
    return block_table.as_strided((rows, block_table.size(1)), block_table.stride())


def populate_metadata(
    self,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: MambaStateCopyFuncsByType,
    block_tables: list[torch.Tensor],
) -> None:
    """Keep the state tensors and block tables the address metadata describes.

    The original still runs, so every other reader of that metadata — the conv
    widths and group indices below included — sees what it expects.
    """
    _ORIGINAL_POPULATE_METADATA(
        self, kv_cache_config, forward_context, mamba_state_copy_funcs, block_tables
    )

    # Same walk order as the original, so index i here is its state i.
    states: list[torch.Tensor] = []
    for mamba_group_id in self.mamba_group_ids:
        kv_cache_group = kv_cache_config.kv_cache_groups[mamba_group_id]
        for layer_name in kv_cache_group.layer_names:
            mamba_spec = _get_mamba_spec_for_layer(kv_cache_group, layer_name)
            state_copy_funcs = mamba_state_copy_funcs[mamba_spec.mamba_type]
            kv_caches: list[torch.Tensor] = forward_context[layer_name].kv_cache
            states.extend(kv_caches[: len(state_copy_funcs)])

    assert len(states) == self.num_states, (
        f"collected {len(states)} state tensors, metadata describes {self.num_states}"
    )
    self._cpu_states = states
    self._cpu_block_tables = [_full_rows(bt) for bt in block_tables]
    # Read once: the copy below is per (request, state) and these never move.
    self._cpu_conv_widths = [int(width) for width in self.state_conv_widths]
    self._cpu_group_indices = [int(group) for group in self.state_group_indices]
    self._cpu_conv_dim_first = is_conv_state_dim_first()


def _copy_state_block(
    self,
    state_idx: int,
    bt_row_idx: int,
    src_col: int,
    dst_col: int,
    token_bias: int,
) -> None:
    """One (layer, state-type) block copy, mirroring _copy_mamba_state_block."""
    state = self._cpu_states[state_idx]
    block_table_row = self._cpu_block_tables[self._cpu_group_indices[state_idx]][
        bt_row_idx
    ]
    conv_width = self._cpu_conv_widths[state_idx]
    dst_block_id = int(block_table_row[dst_col])

    if conv_width == 0:
        # Temporal state: token_bias picks the accepted speculative column.
        src_block_id = int(block_table_row[src_col + token_bias])
        if src_block_id != dst_block_id:
            state[dst_block_id] = state[src_block_id]
        return

    # Conv state: slide the window down by token_bias tokens. state_len is
    # axis 2 in the DS layout and axis 1 in the SD one.
    src_block_id = int(block_table_row[src_col])
    kept = conv_width - token_bias
    if self._cpu_conv_dim_first:
        source = state[src_block_id, :, token_bias:conv_width]
        if src_block_id == dst_block_id:
            source = source.clone()
        state[dst_block_id, :, :kept] = source
    else:
        source = state[src_block_id, token_bias:conv_width]
        if src_block_id == dst_block_id:
            source = source.clone()
        state[dst_block_id, :kept] = source


def run_fused_precopy(
    self,
    num_reqs: int,
    state_idx_gpu: torch.Tensor,
    src_col_gpu: torch.Tensor,
    token_bias_gpu: torch.Tensor,
    idx_mapping: torch.Tensor | None,
) -> None:
    if num_reqs == 0 or not self.is_initialized:
        return

    if idx_mapping is not None:
        req_indices = idx_mapping[:num_reqs].long()
        live = req_indices >= 0
    else:
        req_indices = torch.arange(num_reqs)
        live = torch.ones(num_reqs, dtype=torch.bool)

    safe = req_indices.clamp_min(0)
    src_cols = src_col_gpu[safe]
    dst_cols = state_idx_gpu[safe]
    # A fresh state (-1) or one still writing the same block has nothing to
    # migrate; the forward locates it in-block through num_accepted.
    batch_rows = (live & (src_cols >= 0) & (src_cols != dst_cols)).nonzero()

    for (batch_idx,) in batch_rows.tolist():
        req_idx = int(req_indices[batch_idx])
        bt_row_idx = batch_idx if idx_mapping is not None else req_idx
        token_bias = int(token_bias_gpu[req_idx])
        for state_idx in range(self.num_states):
            _copy_state_block(
                self,
                state_idx,
                bt_row_idx,
                int(src_cols[batch_idx]),
                int(dst_cols[batch_idx]),
                token_bias,
            )


def run_fused_postprocess_align(
    self,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    state_idx_gpu: torch.Tensor,
    new_num_computed_tokens_gpu: torch.Tensor,
    idx_mapping: torch.Tensor,
) -> None:
    if num_reqs == 0 or not self.is_initialized:
        return

    req_indices = idx_mapping[:num_reqs].long()
    live = req_indices >= 0
    safe = req_indices.clamp_min(0)

    # The kernel reads a snapshot because its programs race the in-place reset
    # below; here every gather completes first, so the reset cannot be seen.
    num_accepted = num_accepted_tokens_gpu[safe]
    src_blocks = state_idx_gpu[safe]
    new_num_computed = new_num_computed_tokens_gpu[safe]

    running_state_tokens = new_num_computed - num_accepted + 1
    aligned = (new_num_computed // self.block_size) * self.block_size
    needs_copy = live & (aligned >= running_state_tokens)
    if not bool(needs_copy.any()):
        return

    token_biases = aligned - running_state_tokens
    dst_blocks = aligned // self.block_size - 1

    # Acceptance landed inside the running block: the count resets, and only a
    # nonzero bias still has anything to move.
    at_running_block = needs_copy & (src_blocks == dst_blocks)
    num_accepted_tokens_gpu[req_indices[at_running_block]] = 1

    to_copy = needs_copy & ~(at_running_block & (token_biases == 0))
    for (batch_idx,) in to_copy.nonzero().tolist():
        for state_idx in range(self.num_states):
            _copy_state_block(
                self,
                state_idx,
                batch_idx,
                int(src_blocks[batch_idx]),
                int(dst_blocks[batch_idx]),
                int(token_biases[batch_idx]),
            )


def preprocess_mamba_align(
    grid: tuple[int, ...],
    idx_mapping: torch.Tensor,
    state_idx: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    src_col: torch.Tensor,
    src_off: torch.Tensor,
    num_reqs: int,
    MAMBA_BLOCK_SIZE: int = 0,
    **kwargs: Any,
) -> None:
    """Publish the pre-copy columns, then advance state_idx for this step."""
    rows = idx_mapping[:num_reqs]
    # The kernel reads these slots unconditionally; filtered rows would index
    # backwards from the end of the buffer here, so drop them instead.
    keep = rows >= 0
    req_indices = rows[keep].long()
    if req_indices.numel() == 0:
        return

    previous_state_idx = state_idx[req_indices]
    src_col[req_indices] = previous_state_idx
    src_off[req_indices] = (num_accepted_tokens[req_indices] - 1).clamp_min(0)

    query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
    computed_after = num_computed_tokens[req_indices] + query_lens[keep]
    new_state_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
    state_idx[req_indices] = new_state_idx.to(state_idx.dtype)

    # Only a crossed boundary resets the bias: the migrated state now sits at
    # the start of the new block.
    should_reset = (previous_state_idx >= 0) & (previous_state_idx != new_state_idx)
    num_accepted_tokens[req_indices[should_reset]] = 1
