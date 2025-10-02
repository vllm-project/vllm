# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import ANY, patch

import torch

from vllm.attention.backends.abstract import AttentionType
from vllm.v1.attention.backends.pallas import (PallasAttentionBackendImpl,
                                               PallasMetadata)


def test_ragged_paged_attention():
    # We verify that the kernel inputs such as sliding_window, etc. are passed
    # in from the model correctly.
    # The correctness of the paged attention kernel is tested in the kernel
    # library.
    num_heads = 4
    head_size = 128
    scale = 1.0
    num_kv_heads = 4
    sliding_window = 128
    logits_soft_cap = 50.0
    attn_impl = PallasAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
        logits_soft_cap=logits_soft_cap,
        attn_type=AttentionType.DECODER,
    )

    class FakeAttentionLayer:
        _q_scale_float: float
        _k_scale_float: float
        _v_scale_float: float

    layer = FakeAttentionLayer()
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0

    num_tokens = 16
    num_blocks = 1024
    block_size = 16
    query = torch.zeros(num_tokens, num_heads * head_size)
    key = torch.zeros(num_tokens, num_kv_heads * head_size)
    value = torch.zeros(num_tokens, num_kv_heads * head_size)
    kv_cache = torch.zeros(num_blocks, block_size, num_kv_heads * 2, head_size)
    slot_mapping = torch.zeros((3, num_tokens), dtype=torch.int64)
    max_num_reqs = 8
    max_num_blocks_per_req = 8
    num_kv_update_slices = torch.tensor([num_tokens], dtype=torch.int32)
    block_tables = torch.zeros((max_num_reqs, max_num_blocks_per_req),
                               dtype=torch.int32)
    context_lens = torch.ones((max_num_reqs, ), dtype=torch.int32)
    query_lens = [1] * max_num_reqs
    query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                dtype=torch.int32),
                                   dim=0,
                                   dtype=torch.int32)
    num_seqs = torch.tensor([max_num_reqs], dtype=torch.int32)
    attn_metadata = PallasMetadata(
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        num_kv_update_slices=num_kv_update_slices,
        num_slices_per_kv_cache_update_block=8,
    )

    with patch("torch.ops.xla.ragged_paged_attention"
               ) as mock_ragged_paged_attention:
        attn_impl.forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        mock_ragged_paged_attention.assert_called_once_with(
            ANY,  # query
            ANY,  # kv_cache
            ANY,  # context_lens
            ANY,  # block_tables
            ANY,  # query_start_loc
            ANY,  # num_seqs
            num_kv_pages_per_block=None,
            num_queries_per_block=None,
            vmem_limit_bytes=None,
            use_kernel=True,
            sm_scale=scale,
            sliding_window=sliding_window,
            soft_cap=logits_soft_cap,
            k_scale=1.0,
            v_scale=1.0,
        )
