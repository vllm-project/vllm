# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 HD192 prefill using the unmodified FlashAttention-4 SM100 kernel."""

from typing import Any

import torch

_COMPILED: dict[tuple, Any] = {}


def swa_prefill_fa4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    max_query_len: int,
    scale: float,
    window_left: int,
) -> torch.Tensor:
    """Launch paged BF16 HD192 attention with a 128x64 tile and one KV split."""
    import cutlass.cute as cute

    from vllm.vllm_flash_attn.cute import interface, utils
    from vllm.vllm_flash_attn.cute.cute_dsl_utils import (
        get_broadcast_dims,
        to_cute_tensor,
    )
    from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

    assert query.ndim == 3 and query.shape[-1] == 192
    assert key.ndim == 4 and key.shape[-1] == 192
    assert value.shape == key.shape and output.shape == query.shape
    assert key.shape[2] > 0 and query.shape[1] % key.shape[2] == 0
    assert window_left >= 0 and max_query_len > 4
    for tensor in (query, key, value, output):
        assert tensor.dtype == torch.bfloat16 and tensor.stride(-1) == 1
        assert tensor.is_cuda and tensor.device == query.device
    batch = seq_lens.numel()
    assert seq_lens.shape == (batch,) and query_start_loc.shape == (batch + 1,)
    assert block_table.ndim == 2 and block_table.shape[0] == batch
    for tensor in (query_start_loc, seq_lens, block_table):
        assert tensor.dtype == torch.int32 and tensor.stride(-1) == 1
        assert tensor.device == query.device

    group = query.shape[1] // key.shape[2]
    pack_gqa = group > 1
    q_stage = 2 if max_query_len * group > 128 else 1
    persistent = max_query_len * group <= q_stage * 128
    use_clc = utils._get_use_clc_scheduler_default() and pack_gqa
    cu_blocks = cu_split_blocks = blocks_to_batch = None
    if batch > interface.BIN_BATCH_SEARCH_THRESH:
        cu_blocks, cu_split_blocks = interface._compute_tile_cumsum(
            cu_seqlens=query_start_loc,
            tile_size=128,
            q_stage=q_stage,
            cluster_shape_m=1,
            qhead_per_kvhead=group,
            pack_gqa=pack_gqa,
        )
        if interface.USE_BLOCKS_TO_BATCH:
            blocks_to_batch = interface._compute_blocks_to_batch(
                cu_blocks,
                interface._blocks_to_batch_size(
                    query.shape[0], batch, 128, group, pack_gqa
                ),
                query.device,
            )

    metadata = {
        "mCuSeqlensQ": query_start_loc,
        "mSeqUsedK": seq_lens,
        "mPageTable": block_table,
        "mCuTotalMBlocks": cu_blocks,
        "mCuTotalSplitsMBlocks": cu_split_blocks,
        "mBlocksToBatchIdx": blocks_to_batch,
    }
    scalars = {
        "window_size_left": window_left,
        "window_size_right": 0,
        "max_seqlen_q": max_query_len,
    }
    tensors = (query, key, value, output, *metadata.values())
    cache_key = (
        query.device,
        group,
        q_stage,
        persistent,
        use_clc,
        key.shape[1] != 64,
        tuple(get_broadcast_dims(t) if t is not None else None for t in tensors),
    )
    if cache_key not in _COMPILED:
        # The generic FA4 dispatcher rejects HD192; its SM100 kernel supports it
        # with a 64-token KV tile and no split-KV partial outputs.
        kernel = FlashAttentionForwardSm100(
            192,
            192,
            qhead_per_kvhead=group,
            is_local=True,
            pack_gqa=pack_gqa,
            m_block_size=128,
            n_block_size=64,
            q_stage=q_stage,
            is_static_persistent=persistent,
            paged_kv_non_tma=key.shape[1] != 64,
            is_varlen_q=True,
            use_clc_scheduler=use_clc,
        )
        cute_metadata = {
            name: to_cute_tensor(
                tensor, assumed_align=4, leading_dim=1 if name == "mPageTable" else 0
            )
            for name, tensor in metadata.items()
        }
        _COMPILED[cache_key] = cute.compile(
            kernel,
            *(to_cute_tensor(t) for t in (query, key, value, output)),
            None,
            scale,
            **cute_metadata,
            **scalars,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _COMPILED[cache_key](
        query.detach(),
        key.detach(),
        value.detach(),
        output,
        None,
        scale,
        **metadata,
        **scalars,
    )
    return output
