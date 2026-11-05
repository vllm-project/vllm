# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapters for FlashInfer's GDN ReplaySSM decode kernels."""

import os
from collections.abc import Callable
from functools import cache

import torch

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

GDN_REPLAY_LOGICAL_WINDOW = 16


@cache
def check_gdn_replayssm_dependencies(
    executed_width: int, needs_materializer: bool
) -> Callable[..., torch.Tensor]:
    """Validate the environment and return only the selected decode API."""
    if executed_width not in (1, 4, 8):
        raise ValueError(f"Unsupported GDN ReplaySSM executed width: {executed_width}")
    if os.environ.get("SGLANG_GDN_WY_STRIDED_QKV") != "1":
        raise RuntimeError(
            "FlashInfer GDN ReplaySSM requires "
            "SGLANG_GDN_WY_STRIDED_QKV=1 before importing FlashInfer"
        )
    api = (
        "gated_delta_rule_stp_ucache_flush"
        if executed_width == 1
        else "gated_delta_rule_mtp_ucache_flush"
    )
    mode = f"executed width={executed_width}, cache mode=" + (
        "align" if needs_materializer else "none"
    )
    try:
        # MTP-only providers need not ship the separate STP API.
        if executed_width == 1:
            from flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_stp import (
                gated_delta_rule_stp_ucache_flush as kernel,
            )
        else:
            from flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush import (
                gated_delta_rule_mtp_ucache_flush as kernel,
            )
    except ImportError as e:
        raise ImportError(
            f"FlashInfer GDN ReplaySSM ({mode}) requires {api}: {e}"
        ) from e
    if needs_materializer:
        from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
            _load_gdn_replayssm_materialize,
        )

        try:
            _load_gdn_replayssm_materialize()
        except ImportError as e:
            raise ImportError(
                f"FlashInfer GDN ReplaySSM ({mode}) requires "
                f"gdn_prefix_materialize: {e}"
            ) from e
    return kernel


def pack_replayssm_rows(
    values: torch.Tensor,
    query_start_loc: torch.Tensor,
    executed_query_width: int,
    offsets: torch.Tensor | None = None,
    padding_value: float = 0.0,
    uniform: bool = False,
    single_token: bool = False,
    allow_strided_rows: bool = False,
) -> torch.Tensor:
    """Pack ragged rows; single_token requires at most one token per row."""
    num_rows = query_start_loc.numel() - 1
    if single_token and padding_value == 0.0 and values.shape[0] == num_rows:
        packed = values.new_zeros((num_rows, executed_query_width, *values.shape[1:]))
        packed[:, 0].copy_(values)
        return packed
    if (
        uniform
        and values.shape[0] == num_rows * executed_query_width
        and (
            values.is_contiguous()
            or (
                allow_strided_rows
                and values.ndim == 2
                and values.stride(1) == 1
                and values.stride(0) >= values.shape[1]
                and values.stride(0) * values.element_size() % 16 == 0
            )
        )
        and values.data_ptr() % 16 == 0
    ):
        return values.view(num_rows, executed_query_width, *values.shape[1:])
    if num_rows == 0:
        return values.new_empty((0, executed_query_width, *values.shape[1:]))
    if values.shape[0] == 0:
        raise ValueError("cannot pack non-empty ReplaySSM rows from an empty tensor")
    if offsets is None:
        offsets = torch.arange(
            executed_query_width,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
    else:
        offsets = offsets[:executed_query_width]
    row_offsets = offsets.unsqueeze(0).expand(num_rows, -1)
    query_lens = torch.diff(query_start_loc).unsqueeze(1)
    valid = row_offsets < query_lens
    source_indices = query_start_loc[:-1].unsqueeze(1) + row_offsets
    source_indices = source_indices.clamp(max=max(values.shape[0] - 1, 0))
    packed = values.index_select(0, source_indices.reshape(-1).long()).view(
        num_rows, executed_query_width, *values.shape[1:]
    )
    mask = valid.reshape(num_rows, executed_query_width, *([1] * (values.dim() - 1)))
    return packed.masked_fill(~mask, padding_value)


def run_gdn_replayssm(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    replayssm_cache: tuple[torch.Tensor, ...],
    ring_start: torch.Tensor,
    num_committed: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    output_indices: torch.Tensor,
    executed_query_width: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    offsets: torch.Tensor | None = None,
    kernel: Callable[..., torch.Tensor] | None = None,
    uniform_query_length: int | None = None,
) -> torch.Tensor:
    """Pack rows, invoke FlashInfer, and return only real-token outputs.

    STP receives one real row per request. The adapter supplies FlashInfer's
    four-token input buffer: one real token followed by three zero rows.
    vLLM publishes cursors once per model step, after every layer runs.
    """
    if executed_query_width not in (1, 4, 8):
        raise ValueError(
            f"FlashInfer GDN ReplaySSM requires executed width 1, 4, or 8, got "
            f"{executed_query_width}"
        )
    if len(replayssm_cache) != 3:
        raise ValueError(
            "FlashInfer GDN ReplaySSM requires exactly three auxiliary rings"
        )
    if mixed_qkv.dtype != torch.bfloat16 or checkpoint_state.dtype != torch.bfloat16:
        raise ValueError(
            "FlashInfer GDN ReplaySSM requires bfloat16 inputs and checkpoint state"
        )

    # Native STP's public contract is T=1, but its serving fast path consumes a
    # caller-owned zero-padded T=4 buffer. Packing a shared QKV allocation here
    # preserves the common token stride required by block-strided cache pools.
    packed_width = 4 if executed_query_width == 1 else executed_query_width
    uniform = executed_query_width > 1 and uniform_query_length == packed_width
    # STP metadata permits only zero or one token per row. Eager zero-length
    # rows use the packer's fallback. Captured padding can have input storage;
    # its negative state index makes FlashInfer skip that row.
    single_token = executed_query_width == 1
    packed_qkv = pack_replayssm_rows(
        mixed_qkv,
        query_start_loc,
        packed_width,
        offsets,
        uniform=uniform,
        single_token=single_token,
        # FlashInfer accepts Q/K/V views with a common aligned token stride.
        # Gates keep their compact layout independently of this QKV path.
        allow_strided_rows=True,
    )
    # CuTe requires every input pointer to be 16-byte aligned. Packing the two
    # gates together and splitting them can leave ``b`` at a small byte offset
    # into the shared allocation when the local head count is not a multiple of
    # eight. Give each gate its own aligned allocation instead.
    # The MTP kernel advances every padded lane. Make those lanes exact
    # recurrent no-ops: a=-inf gives zero decay after softplus(a + dt_bias),
    # while zero q/k/v and b give a zero state update. STP has no ragged real
    # rows, so only invalid graph-padding requests can be padded here; their
    # negative state index skips the kernel CTA. STP requires three zero
    # input rows after the real token. It returns and commits only that token.
    packed_a = pack_replayssm_rows(
        a,
        query_start_loc,
        packed_width,
        offsets,
        padding_value=0.0 if executed_query_width == 1 else -torch.inf,
        uniform=uniform,
        single_token=single_token,
    )
    packed_b = pack_replayssm_rows(
        b,
        query_start_loc,
        packed_width,
        offsets,
        uniform=uniform,
        single_token=single_token,
    )
    local_k_dim = num_k_heads * head_k_dim
    local_v_dim = num_v_heads * head_v_dim
    q_flat, k_flat, v_flat = packed_qkv.split(
        (local_k_dim, local_k_dim, local_v_dim), dim=-1
    )
    q = q_flat.view(packed_qkv.size(0), packed_width, num_k_heads, head_k_dim)
    k = k_flat.view_as(q)
    v = v_flat.view(packed_qkv.size(0), packed_width, num_v_heads, head_v_dim)
    # vLLM reserves block zero for padding, while FlashInfer exits a null-row
    # CTA only for a negative index. Gather trackers through the safe padding
    # slot, then translate every non-live row to FlashInfer's -1 sentinel.
    valid_slots = state_indices > NULL_BLOCK_ID
    safe_slots = torch.where(valid_slots, state_indices, NULL_BLOCK_ID).long()
    fi_state_indices = torch.where(valid_slots, state_indices, -1)
    hist_len = num_committed.index_select(0, safe_slots).masked_fill(~valid_slots, 0)
    cache_base = ring_start.index_select(0, safe_slots).masked_fill(~valid_slots, 0)
    u_cache, k_cache, g_cache = replayssm_cache
    output = torch.zeros(
        packed_qkv.size(0),
        executed_query_width,
        num_v_heads,
        head_v_dim,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )

    if kernel is None:
        kernel = check_gdn_replayssm_dependencies(executed_query_width, False)
    if executed_query_width == 1:
        # STP updates these cursor copies. Every layer must start from the
        # same pre-step values; model-wide postprocess publishes the update.
        mode_kwargs = {"prepadded": True}
        flush_min = GDN_REPLAY_LOGICAL_WINDOW - 1
    else:
        mode_kwargs = {"restart_hist_on_flush": False}
        flush_min = GDN_REPLAY_LOGICAL_WINDOW + 1 - executed_query_width
    kernel(
        # These are immutable model parameters. CuTe's DLPack bridge rejects
        # tensors whose Parameter wrapper still advertises autograd tracking,
        # even while vLLM executes under inference mode.
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
        flush_min=flush_min,
        **mode_kwargs,
    )
    return output.flatten(0, 1).index_select(0, output_indices.long())


__all__ = [
    "GDN_REPLAY_LOGICAL_WINDOW",
    "check_gdn_replayssm_dependencies",
    "pack_replayssm_rows",
    "run_gdn_replayssm",
]
