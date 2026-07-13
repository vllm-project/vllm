# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin wrappers for TokenSpeed-kernel AMD MHA kernels."""

import math
from functools import cache

import torch


@cache
def _get_prefill_kernel():
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_fp16_gfx950 import (
        gluon_mha_prefill_fp16_gfx950,
    )

    return gluon_mha_prefill_fp16_gfx950


@cache
def _get_prefill_launch_symbols():
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_prefill_fp16_gfx950 import (
        _mha_prefill_fp16,
        _mha_prefill_sliding_fp16,
        get_config,
    )

    return get_config, _mha_prefill_fp16, _mha_prefill_sliding_fp16


@cache
def _get_decode_kernel():
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_decode_fp16_gfx950 import (
        gluon_mha_decode_fp16_gfx950,
    )

    return gluon_mha_decode_fp16_gfx950


@cache
def _get_decode_launch_symbols():
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_decode_fp16_gfx950 import (
        _mha_decode_fp16,
        _mha_decode_reduce_fp16,
        _mha_decode_sliding_fp16,
        get_config,
    )

    return (
        get_config,
        _mha_decode_fp16,
        _mha_decode_reduce_fp16,
        _mha_decode_sliding_fp16,
    )


@cache
def _get_extend_kernel():
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_extend_fp16_gfx950 import (
        gluon_mha_extend_fp16_gfx950,
    )

    return gluon_mha_extend_fp16_gfx950


@cache
def _get_extend_launch_symbols():
    from tokenspeed_kernel_amd.ops.attention.gluon.mha_extend_fp16_gfx950 import (
        _INV_LN2_VALUE,
        _mha_extend_fp16,
        _select_extend_tile,
    )

    return _select_extend_tile, _mha_extend_fp16, _INV_LN2_VALUE


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
    if output is not None:
        get_config, prefill_kernel, sliding_prefill_kernel = (
            _get_prefill_launch_symbols()
        )
        config = get_config(
            q=query,
            k=key,
            cu_seqlens_q=query_start_loc,
            max_seqlen=max_query_len,
            window_left=sliding_window,
        )
        has_sink = sinks is not None
        sink_arg = sinks if sinks is not None else query
        lse_arg = query
        kernel = sliding_prefill_kernel if config.window_left >= 0 else prefill_kernel
        kernel[config.grid](
            query,
            key,
            value,
            query_start_loc,
            output,
            sink_arg,
            lse_arg,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.sm_scale,
            config.block_m,
            config.block_n,
            config.num_warps,
            config.batch_size,
            config.max_seqlen,
            has_sink,
            False,
            config.window_left,
            num_warps=config.num_warps,
        )
        return output

    return _get_prefill_kernel()(
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
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is not None:
        total_q = query.shape[0]
        (
            get_config,
            decode_kernel,
            reduce_kernel,
            sliding_decode_kernel,
        ) = _get_decode_launch_symbols()
        config = get_config(
            q=query,
            k_cache=key_cache,
            max_seqlen_k=max_seq_len,
            window_left=sliding_window,
        )
        has_sink = sinks is not None
        sink_arg = sinks if sinks is not None else query
        if config.is_sliding:
            grid = (total_q, config.num_kv_heads * config.num_groups, 1)
            sliding_decode_kernel[grid](
                query,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                output,
                sink_arg,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                config.sm_scale,
                block_table.stride(0),
                config.page_size,
                max_query_len,
                config.num_q_heads,
                config.num_kv_heads,
                config.head_dim,
                config.block_m,
                config.block_n,
                config.is_sliding,
                config.window_left,
                has_sink,
                num_warps=1,
            )
        else:
            mid_o = torch.empty(
                (
                    total_q,
                    config.num_q_heads,
                    config.num_kv_splits,
                    config.head_dim,
                ),
                device=query.device,
                dtype=torch.float32,
            )
            mid_lse = torch.empty(
                (total_q, config.num_q_heads, config.num_kv_splits),
                device=query.device,
                dtype=torch.float32,
            )
            grid = (
                total_q,
                config.num_kv_heads * config.num_groups,
                config.num_kv_splits,
            )
            decode_kernel[grid](
                query,
                key_cache,
                value_cache,
                block_table,
                seq_lens,
                mid_o,
                mid_lse,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                config.sm_scale,
                block_table.stride(0),
                config.page_size,
                config.num_kv_splits,
                max_query_len,
                config.num_q_heads,
                config.num_kv_heads,
                config.head_dim,
                config.block_m,
                config.block_n,
                config.is_sliding,
                config.window_left,
                num_warps=1,
            )

            grid = (total_q, config.num_q_heads)
            reduce_kernel[grid](
                mid_o,
                mid_lse,
                output,
                seq_lens,
                sink_arg,
                config.sm_scale,
                block_table.stride(0),
                config.num_kv_splits,
                max_query_len,
                config.page_size,
                config.num_q_heads,
                config.num_kv_heads,
                config.head_dim,
                config.block_m,
                config.block_n,
                has_sink,
                num_warps=1,
            )
        return output

    return _get_decode_kernel()(
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
    if output is not None:
        _select_extend_tile, extend_kernel, inv_ln2_value = _get_extend_launch_symbols()
        head_dim = query.shape[2]
        n_heads = query.shape[1]
        n_kv_heads = key_cache.shape[2]
        page_size = key_cache.shape[1]
        block_m, block_n, num_warps = _select_extend_tile(max_query_len)
        sm_scale = (1.0 / math.sqrt(head_dim)) * inv_ln2_value

        batch = query_start_loc.shape[0] - 1
        safe_max_q = max_query_len if max_query_len > 0 else 1
        blocks_per_req = (safe_max_q + block_m - 1) // block_m
        has_sink = sinks is not None
        sink_arg = sinks if has_sink else query
        cu_q_i32 = query_start_loc.to(torch.int32).contiguous()
        cache_i32 = seq_lens.to(torch.int32).contiguous()
        lse_arg = query

        grid = (blocks_per_req, batch, n_heads)
        extend_kernel[grid](
            query,
            key_cache,
            value_cache,
            block_table,
            output,
            lse_arg,
            sink_arg,
            cu_q_i32,
            cache_i32,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            sm_scale,
            block_table.stride(0),
            page_size,
            n_heads,
            n_kv_heads,
            head_dim,
            block_m,
            block_n,
            num_warps,
            True,
            has_sink,
            False,
            sliding_window,
            num_warps=num_warps,
        )
        return output

    return _get_extend_kernel()(
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
