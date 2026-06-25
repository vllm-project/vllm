# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrappers for TokenSpeed-kernel AMD MHA kernels."""

import torch


def rocm_tokenspeed_mha_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    sliding_window: int,
    sinks: torch.Tensor | None,
) -> torch.Tensor:
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_fp16_gfx950 import (
        gluon_mha_prefill_fp16_gfx950,
    )

    return gluon_mha_prefill_fp16_gfx950(
        q=query,
        k=key,
        v=value,
        cu_seqlens=query_start_loc,
        cu_seqlens_cpu=query_start_loc_cpu.tolist(),
        max_seqlen=max_query_len,
        window_left=sliding_window,
        sinks=sinks,
    )


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
) -> torch.Tensor:
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_decode_fp16_gfx950 import (
        gluon_mha_decode_fp16_gfx950,
    )

    return gluon_mha_decode_fp16_gfx950(
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
