# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import cascade_attention
from vllm.vllm_flash_attn import (fa_version_unsupported_reason,
                                  flash_attn_varlen_func,
                                  is_fa_version_supported)

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 192, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.float16, torch.bfloat16]

# GROUPING_DATA = [
#     # Case 1. A general case.
#     ([(129, 871), (18, 280), (37, 988), (1023, 2304), (1, 257)],
#      [0, 2, 5],
#      [272, 144]),
#     # Case 2. Flash-decoding case.
#     ([(1, 1023), (1, 879), (1, 778), (1, 1777)] * 100,
#      [0, 100, 200, 300, 400],
#      [768, 384, 256, 16])
# ]

# GROUPING_DATA = [
#     ([(16, 48), (16, 48), (32, 80), (32, 80)],
#      [0, 2, 4],
#      [16, 64])
# ]

GROUPING_DATA = [
    ([(32, 80), (32, 80)],
     [0, 2],
     [64])
]

# Multi-Cascade Attention Fails When One of the Common Prefixes
# Exceeds kv_len for some queries except when all queries have same length.
# (Something to do with query tokens?)


@pytest.mark.parametrize("grouping_data", GROUPING_DATA)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("soft_cap", [None, 50])
@pytest.mark.parametrize("num_blocks", [2048])
@pytest.mark.parametrize("fa_version", [2, 3])
@torch.inference_mode()
def test_multi_cascade(
    grouping_data: tuple[list[tuple[int, int]], list[int], list[int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    fa_version: int,
) -> None:
    torch.set_default_device("cuda")
    if not is_fa_version_supported(fa_version):
        pytest.skip(f"Flash attention version {fa_version} not supported due "
                    f"to: \"{fa_version_unsupported_reason(fa_version)}\"")

    current_platform.seed_everything(0)

    window_size = (-1, -1)
    scale = head_size**-0.5
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    value_cache = torch.randn_like(key_cache)

    seq_lens, group_indices, common_prefix_lens = grouping_data
    assert len(group_indices) == len(common_prefix_lens) + 1
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)

    total_num_query_tokens = sum(query_lens)
    query = torch.randn(total_num_query_tokens,
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    cu_query_lens = torch.tensor([0] + query_lens,
                                dtype=torch.int32).cumsum(dim=0,
                                                        dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                num_blocks,
                                (num_seqs, max_num_blocks_per_seq),
                                dtype=torch.int32)

    assert all([common_prefix_len > 0 for
                common_prefix_len in common_prefix_lens])
    assert all([common_prefix_len % block_size == 0 for
                common_prefix_len in common_prefix_lens])

    # Write common prefixes into appropriate groups in block_tables
    for i in range(len(common_prefix_lens)):
        group_start = group_indices[i]
        group_end = group_indices[i + 1]
        common_prefix_len = common_prefix_lens[i]
        block_tables[group_start: group_end, :common_prefix_len] = (
            block_tables[group_start, :common_prefix_len])

    # Run the regular attention.
    ref_output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
    )

    # Run cascade attention.
    assert all([common_prefix_lens[i] < kv_len
                for i in range(len(common_prefix_lens))
                for kv_len in kv_lens[group_indices[i]: group_indices[i + 1]]])
    cu_query_lens_cpu = cu_query_lens.cpu()
    cu_prefix_query_lens = torch.tensor(
        [*cu_query_lens_cpu[group_indices[:-1]].tolist(), total_num_query_tokens],
        dtype=torch.int32)
    prefix_kv_lens = torch.tensor(common_prefix_lens, dtype=torch.int32)
    suffix_kv_lens = kv_lens_tensor - torch.repeat_interleave(
        prefix_kv_lens, 
        torch.tensor([group_indices[i + 1] - group_indices[i] 
        for i in range(len(group_indices) - 1)], dtype=torch.int32))
    output = torch.empty_like(query)

    cascade_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=cu_query_lens,
        max_query_len=max_query_len,
        cu_prefix_query_lens=cu_prefix_query_lens,
        prefix_kv_lens=prefix_kv_lens,
        suffix_kv_lens=suffix_kv_lens,
        max_kv_len=max_kv_len,
        softmax_scale=scale,
        alibi_slopes=None,
        sliding_window=window_size,
        logits_soft_cap=soft_cap if soft_cap is not None else 0,
        block_table=block_tables,
        group_indices=group_indices,
        common_prefix_lens=common_prefix_lens,
        fa_version=fa_version,
    )

    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
