# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrappers for TokenSpeed-kernel AMD MHA kernels."""

from functools import cache

import torch


@cache
def _get_prefill_kernel():
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (
        launch_gluon_mha_prefill_gfx950,
    )

    return launch_gluon_mha_prefill_gfx950


@cache
def _get_decode_kernel():
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.decode import (
        launch_gluon_mha_decode_gfx950,
    )

    return launch_gluon_mha_decode_gfx950


@cache
def _get_extend_kernel():
    from tokenspeed_kernel_amd.ops.gfx950.attention.mha.extend import (
        launch_gluon_mha_extend_gfx950,
    )

    return launch_gluon_mha_extend_gfx950


def _write_output(result: torch.Tensor, output: torch.Tensor | None) -> torch.Tensor:
    # The TokenSpeed launchers allocate their own output.
    if output is None:
        return result
    output.copy_(result)
    return output


def rocm_tokenspeed_mha_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    sliding_window: int,
    sinks: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    result = _get_prefill_kernel()(
        q=query,
        k=key,
        v=value,
        cu_seqlens=query_start_loc,
        cu_seqlens_cpu=query_start_loc_cpu.tolist(),
        max_seqlen=max_query_len,
        window_left=sliding_window,
        sinks=sinks,
    )
    return _write_output(result, output)


def rocm_tokenspeed_mha_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    max_query_len: int,
    sliding_window: int,
    sinks: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    result = _get_decode_kernel()(
        q=query,
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=block_table,
        cache_seqlens=seq_lens,
        max_seqlen_k=max_seq_len,
        max_seqlen_q=max_query_len,
        window_left=sliding_window,
        sinks=sinks,
    )
    return _write_output(result, output)


def rocm_tokenspeed_mha_extend(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    max_query_len: int,
    sliding_window: int,
    sinks: torch.Tensor | None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    result = _get_extend_kernel()(
        q=query,
        cu_seqlens_q=query_start_loc,
        cu_seqlens_kv=query_start_loc,
        k_cache=key_cache,
        v_cache=value_cache,
        page_table=block_table,
        cache_seqlens=seq_lens,
        is_causal=True,
        window_left=sliding_window,
        sinks=sinks,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_seq_len,
    )
    return _write_output(result, output)
