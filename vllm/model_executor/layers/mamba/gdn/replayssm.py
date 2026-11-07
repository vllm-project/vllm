# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for FlashInfer's FP16-state GDN ReplaySSM decode kernel."""

from collections.abc import Callable
from functools import cache

import torch

from vllm.model_executor.layers.mamba.mamba_utils import GDN_REPLAYSSM_LOGICAL_WINDOW
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# CuTe loads every kernel input with 16-byte vector accesses.
_CUTE_ALIGNMENT_BYTES = 16


@cache
def load_gdn_replayssm_kernel() -> Callable[..., None]:
    """Import FlashInfer's SM100 UMMA ReplaySSM decode kernel."""
    try:
        from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
            gated_delta_rule_mtp_ucache_flush_umma,
        )
    except ImportError as e:
        raise ImportError(
            "FlashInfer GDN ReplaySSM requires "
            "gated_delta_rule_mtp_ucache_flush_umma (FlashInfer >= 0.7.3)"
        ) from e
    return gated_delta_rule_mtp_ucache_flush_umma


@cache
def load_gdn_replayssm_materializer() -> Callable[..., None]:
    """Import FlashInfer's prefix materializer and check its FP16 state dtype."""
    try:
        from flashinfer.gdn_kernels.gdn_prefix_materialize import (
            gdn_prefix_materialize,
            state_ty_torch,
        )
    except ImportError as e:
        raise ImportError(
            "FlashInfer GDN ReplaySSM prefix caching requires "
            "flashinfer.gdn_kernels.gdn_prefix_materialize (FlashInfer >= 0.7.3)"
        ) from e
    if state_ty_torch() != torch.float16:
        raise ValueError(
            f"FlashInfer GDN prefix materialization uses {state_ty_torch()} "
            "state because FlashInfer was imported before vLLM configured it. "
            "Set GDN_UCACHE_STATE_DTYPE=fp16 before starting the process."
        )
    return gdn_prefix_materialize


def pack_replayssm_rows(
    values: torch.Tensor,
    query_start_loc: torch.Tensor,
    executed_width: int,
    offsets: torch.Tensor | None = None,
    padding_value: float = 0.0,
    uniform: bool = False,
    allow_strided_rows: bool = False,
) -> torch.Tensor:
    """Pack variable-length token rows into ``[num_rows, executed_width, ...]``.

    Row ``i`` holds the tokens in ``query_start_loc[i]:query_start_loc[i + 1]``
    followed by ``padding_value``. With ``uniform``, every row has exactly
    ``executed_width`` tokens, so the result is a view of ``values`` when it
    is contiguous and aligned for CuTe, and a contiguous copy otherwise.
    ``allow_strided_rows`` also preserves aligned, non-overlapping 2D rows.
    """
    num_rows = query_start_loc.numel() - 1
    if uniform and values.shape[0] == num_rows * executed_width:
        aligned_rows = (
            allow_strided_rows
            and values.ndim == 2
            and values.stride(1) == 1
            and values.stride(0) >= values.shape[1]
            and values.stride(0) * values.element_size() % _CUTE_ALIGNMENT_BYTES == 0
        )
        if values.data_ptr() % _CUTE_ALIGNMENT_BYTES or not (
            values.is_contiguous() or aligned_rows
        ):
            values = values.clone(memory_format=torch.contiguous_format)
        return values.view(num_rows, executed_width, *values.shape[1:])
    if num_rows == 0:
        return values.new_empty((0, executed_width, *values.shape[1:]))
    if values.shape[0] == 0:
        raise ValueError("cannot pack non-empty ReplaySSM rows from an empty tensor")
    if offsets is None:
        offsets = torch.arange(
            executed_width, dtype=query_start_loc.dtype, device=query_start_loc.device
        )
    else:
        offsets = offsets[:executed_width]
    row_offsets = offsets.unsqueeze(0).expand(num_rows, -1)
    query_lens = torch.diff(query_start_loc).unsqueeze(1)
    valid = row_offsets < query_lens
    source_indices = query_start_loc[:-1].unsqueeze(1) + row_offsets
    source_indices = source_indices.clamp(max=values.shape[0] - 1)
    packed = values.index_select(0, source_indices.reshape(-1).long()).view(
        num_rows, executed_width, *values.shape[1:]
    )
    mask = valid.reshape(num_rows, executed_width, *([1] * (values.dim() - 1)))
    return packed.masked_fill(~mask, padding_value)


def run_gdn_replayssm(
    *,
    kernel: Callable[..., None],
    executed_width: int,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replayssm_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ring_start: torch.Tensor,
    num_committed: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    output_indices: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    offsets: torch.Tensor | None = None,
    uniform_query_length: int | None = None,
) -> torch.Tensor:
    """Run one FlashInfer decode step over the replay rows.

    Returns the outputs of real tokens in ``output_indices`` order. The kernel
    reads the committed cursors but does not publish them; vLLM commits the
    accepted tokens once per model step, after every layer runs.
    """
    uniform = uniform_query_length == executed_width
    packed_qkv = pack_replayssm_rows(
        mixed_qkv,
        query_start_loc,
        executed_width,
        offsets,
        uniform=uniform,
        # UMMA makes contiguous Q/K/V copies after splitting the packed view.
        allow_strided_rows=True,
    )
    # Padded token lanes must not change the state: a=-inf makes the decay
    # exp(-softplus(a + dt_bias) * exp(A_log)) one, and zero q/k/v/b make the
    # update zero. a and b are packed separately so each gets an aligned
    # allocation.
    packed_a = pack_replayssm_rows(
        a,
        query_start_loc,
        executed_width,
        offsets,
        padding_value=-torch.inf,
        uniform=uniform,
    )
    packed_b = pack_replayssm_rows(
        b, query_start_loc, executed_width, offsets, uniform=uniform
    )
    local_k_dim = num_k_heads * head_k_dim
    local_v_dim = num_v_heads * head_v_dim
    q_flat, k_flat, v_flat = packed_qkv.split(
        (local_k_dim, local_k_dim, local_v_dim), dim=-1
    )
    num_rows = packed_qkv.size(0)
    q = q_flat.view(num_rows, executed_width, num_k_heads, head_k_dim)
    k = k_flat.view_as(q)
    v = v_flat.view(num_rows, executed_width, num_v_heads, head_v_dim)
    # vLLM pads with block zero; FlashInfer skips only negative state indices.
    # Gather cursors through the padding slot, then pass -1 for padding rows.
    valid_slots = state_indices > NULL_BLOCK_ID
    safe_slots = torch.where(valid_slots, state_indices, NULL_BLOCK_ID).long()
    fi_state_indices = torch.where(valid_slots, state_indices, -1)
    hist_len = num_committed.index_select(0, safe_slots).masked_fill(~valid_slots, 0)
    cache_base = ring_start.index_select(0, safe_slots).masked_fill(~valid_slots, 0)
    u_cache, k_cache, g_cache = replayssm_cache
    # Zero, not empty: padded token rows keep flowing through the model.
    output = torch.zeros(
        num_rows,
        executed_width,
        num_v_heads,
        head_v_dim,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    kernel(
        # Detached views, not the Parameters: CuTe rejects tensors that require
        # grad, and FlashInfer caches its dtype cast by tensor object. A new
        # view per call picks up in-place weight reloads.
        A_log=A_log.detach(),
        a=packed_a,
        dt_bias=dt_bias.detach(),
        q=q,
        k=k,
        v=v,
        b=packed_b,
        initial_state_source=checkpoint_state,
        initial_state_indices=fi_state_indices,
        use_qk_l2norm_in_kernel=True,
        scale=head_k_dim**-0.5,
        output=output,
        k_cache=k_cache,
        u_cache=u_cache,
        g_cache=g_cache,
        hist_len=hist_len,
        cache_base=cache_base,
        # Fold history into the checkpoint once history plus this launch's
        # tokens would exceed the logical window.
        flush_min=GDN_REPLAYSSM_LOGICAL_WINDOW + 1 - executed_width,
        restart_hist_on_flush=False,
    )
    return output.flatten(0, 1).index_select(0, output_indices.long())
