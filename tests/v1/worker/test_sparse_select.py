# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.sparse_select import kv_repr_gen, sparse_kv_selection


def test_kv_repr_gen_torch_fallback_cpu():
    # [2, num_blocks, block_size, num_heads, head_dim]
    kv_cache = torch.zeros((2, 4, 2, 1, 2), dtype=torch.float32)
    kv_cache[0, 1, :, :, :] = torch.tensor([[[1.0, 3.0]], [[5.0, 7.0]]])
    kv_cache[0, 3, :, :, :] = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]]])

    block_repr = torch.zeros((8, 1, 2), dtype=torch.float32)
    mapping = torch.tensor([[1, 5], [3, 6]], dtype=torch.int64)

    kv_repr_gen(
        kv_cache=kv_cache,
        block_repr=block_repr,
        mapping=mapping,
        num_mappings=2,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
    )

    torch.testing.assert_close(block_repr[5, 0], torch.tensor([3.0, 5.0]))
    torch.testing.assert_close(block_repr[6, 0], torch.tensor([4.0, 6.0]))


def test_sparse_kv_selection_torch_fallback_keeps_boundary_blocks():
    batch_size = 1
    block_size = 2
    max_blocks = 4
    top_k = 2

    block_table = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([9], dtype=torch.int32)  # kv_len=8, query_len=1
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    query = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    k_repr = torch.ones((4, 1, 2), dtype=torch.float32)
    scores = torch.empty((batch_size, max_blocks), dtype=torch.float32)

    topk_choices, _ = sparse_kv_selection(
        block_table=block_table,
        batch_size=batch_size,
        block_size=block_size,
        max_num_blocks_this_batch=max_blocks,
        seq_lens=seq_lens,
        k_repr=k_repr,
        query=query,
        query_start_loc=query_start_loc,
        top_k=top_k,
        scores=scores,
    )

    selected = set(topk_choices[0].tolist())
    assert 0 in selected
    assert 3 in selected
