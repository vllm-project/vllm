# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import (cascade_attention,
                                                   merge_attn_states)
from vllm.vllm_flash_attn import (fa_version_unsupported_reason,
                                  flash_attn_varlen_func,
                                  is_fa_version_supported)

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 192, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("num_tokens", [1, 39, 16912])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_merge_kernel(
    num_tokens: int,
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    # Prepare inputs.
    prefix_output = torch.randn(num_tokens,
                                num_query_heads,
                                head_size,
                                dtype=dtype)
    suffix_output = torch.randn(num_tokens,
                                num_query_heads,
                                head_size,
                                dtype=dtype)
    prefix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)
    suffix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)

    # Run the kernel.
    output = torch.empty(num_tokens, num_query_heads, head_size, dtype=dtype)
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse)

    # Reference implementation.
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p_lse = torch.exp(prefix_lse - max_lse)
    s_lse = torch.exp(suffix_lse - max_lse)
    p_scale = p_lse / (p_lse + s_lse)
    s_scale = s_lse / (p_lse + s_lse)
    p_scale = p_scale.transpose(0, 1).unsqueeze(2)
    s_scale = s_scale.transpose(0, 1).unsqueeze(2)
    ref_output = p_scale * prefix_output + s_scale * suffix_output
    ref_output = ref_output.to(dtype)

    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


CASES = [
    # Case 1. A general case.
    ([(129, 871), (18, 280), (37, 988), (1023, 2304), (1, 257)], 256),
    # Case 2. Flash-decoding case.
    ([(1, 1023), (1, 879), (1, 778), (1, 1777)] * 100, 512),
]


@pytest.mark.parametrize("seq_lens_and_common_prefix", CASES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("soft_cap", [None, 50])
@pytest.mark.parametrize("num_blocks", [2048])
@pytest.mark.parametrize("fa_version", [2, 3])
@torch.inference_mode()
def test_cascade(
    seq_lens_and_common_prefix: Tuple[List[Tuple[int, int]], int],
    num_heads: Tuple[int, int],
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

    seq_lens, common_prefix_len = seq_lens_and_common_prefix
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

    assert common_prefix_len > 0
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    # Make sure the first `num_common_kv_blocks` blocks are the same.
    block_tables[:, :num_common_kv_blocks] = \
        block_tables[0, :num_common_kv_blocks]

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
    assert all(common_prefix_len < kv_len for kv_len in kv_lens)
    cu_prefix_query_lens = torch.tensor([0, total_num_query_tokens],
                                        dtype=torch.int32)
    prefix_kv_lens = torch.tensor([common_prefix_len], dtype=torch.int32)
    suffix_kv_lens = kv_lens_tensor - common_prefix_len
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
        common_prefix_len=common_prefix_len,
        fa_version=fa_version,
    )

    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
