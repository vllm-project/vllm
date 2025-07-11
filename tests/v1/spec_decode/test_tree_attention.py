# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.tree_attn import TreeAttentionBackend


class NoOpLayerModule(torch.nn.Module):
    _q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    _k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    _v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def forward_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    seqlen_k: int,
    qq_attn_mask: Optional[torch.Tensor],
    backend_cls: type[AttentionBackend],
) -> torch.Tensor:
    batch_size, q_len, num_heads, dim_per_head = q.shape
    num_kv_heads = k.shape[-2]
    # Initialize the query and KV sequence lengths.
    cu_seqlens_q = q_len * torch.arange(
        batch_size + 1, device=q.device, dtype=torch.int32)
    seqlens_kv = torch.full(
        (batch_size, ),
        seqlen_k,
        device=q.device,
        dtype=torch.int32,
    )
    max_seqlen_q = q_len
    num_actual_tokens = cu_seqlens_q[-1]

    softmax_scale = q.shape[-1]**(-0.5)
    layer = NoOpLayerModule()

    # Set common metadata.
    attn_metadata_dict = {
        "num_actual_tokens": num_actual_tokens,
        "max_query_len": max_seqlen_q,
        "query_start_loc": cu_seqlens_q,
        "max_seq_len": seqlen_k,
        "seq_lens": seqlens_kv,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
    }

    # Set backend-specific metadata.
    if backend_cls == FlashAttentionBackend:
        attn_metadata_dict["use_cascade"] = False
        attn_metadata_dict["common_prefix_len"] = 0
        attn_metadata_dict["cu_prefix_query_lens"] = None
        attn_metadata_dict["prefix_kv_lens"] = None
        attn_metadata_dict["suffix_kv_lens"] = None
    elif backend_cls == TreeAttentionBackend:
        attn_metadata_dict["tree_attn_bias"] = torch.where(
            qq_attn_mask == 1, torch.tensor(0.0), torch.tensor(float("-inf")))
        attn_metadata_dict["num_prefill_tokens"] = 0
        attn_metadata_dict["num_prefills"] = 0
        attn_metadata_dict["num_decode_tokens"] = num_actual_tokens
        attn_metadata_dict["num_decodes"] = batch_size

    # Initialize the backend implementation.
    instance = backend_cls.get_impl_cls()(
        num_heads=num_heads,
        head_size=dim_per_head,
        scale=softmax_scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Run forward pass and return output.
    query = q.view(-1, num_heads, dim_per_head)
    key = k.view(-1, num_kv_heads, dim_per_head)
    value = v.view(-1, num_kv_heads, dim_per_head)
    output = torch.empty_like(query)
    return instance.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache.clone(),
        attn_metadata=backend_cls.get_metadata_cls()(**attn_metadata_dict),
        output=output,
    )


def test_tree_attn_correctness() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda"
    tree_attn_masks = [
        # Chain.
        torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            device=device,
            dtype=torch.int32,
        ),
        # Tree.
        torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0],
                [1, 1, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.int32,
        ),
    ]

    dim_per_head = 128
    num_kv_heads = 2
    block_size = 128
    max_sequence_length = 8192
    randomize_blocks = True
    for batch_size in [1, 16, 64]:
        for num_heads in [2, 4]:
            for sequence_position in [16, 1024, 2048]:
                for tree_attn_mask in tree_attn_masks:
                    # Assert that the number of heads is divisible
                    # by the number of KV heads.
                    assert num_heads % num_kv_heads == 0

                    # Initialize q, k, and v.
                    tree_size_q = tree_attn_mask.shape[0]
                    seqlen_k = sequence_position + tree_size_q
                    q = torch.randn(
                        (batch_size, tree_size_q, num_heads, dim_per_head),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    k = torch.randn(
                        (batch_size, tree_size_q, num_kv_heads, dim_per_head),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    v = torch.randn(
                        (batch_size, tree_size_q, num_kv_heads, dim_per_head),
                        device=device,
                        dtype=torch.bfloat16,
                    )

                    # Setup the block table and KV cache for paged KV.
                    assert max_sequence_length % block_size == 0
                    max_blocks_per_batch = max_sequence_length // block_size
                    kv_cache = torch.randn(
                        (
                            2,
                            batch_size * max_blocks_per_batch,
                            block_size,
                            num_kv_heads,
                            dim_per_head,
                        ),
                        device=q.device,
                        dtype=torch.bfloat16,
                    )
                    num_alloc_blocks_per_batch = math.ceil(seqlen_k /
                                                           block_size)
                    block_table = torch.zeros(
                        (batch_size, max_blocks_per_batch),
                        device=q.device,
                        dtype=torch.int32,
                    )
                    block_ids = torch.arange(
                        0,
                        batch_size * num_alloc_blocks_per_batch,
                        device=q.device,
                        dtype=torch.int32,
                    )
                    if randomize_blocks:
                        # Randomize the block ids.
                        block_ids = block_ids[torch.randperm(
                            block_ids.numel())]
                    block_table[:, :
                                num_alloc_blocks_per_batch] = block_ids.view(
                                    -1, num_alloc_blocks_per_batch)

                    # Setup the slot mapping for the input KVs.
                    tree_positions = sequence_position + torch.arange(
                        0,
                        tree_size_q,
                        device=q.device,
                        dtype=torch.int64,
                    ).repeat(batch_size, 1)
                    tree_slot_mapping = _gen_slot_mapping(
                        tree_positions, block_table, block_size)

                    # Compute attention for the tree.
                    tree_attn_output = forward_attention(
                        q=q,
                        k=k,
                        v=v,
                        kv_cache=kv_cache,
                        block_table=block_table,
                        slot_mapping=tree_slot_mapping,
                        seqlen_k=seqlen_k,
                        qq_attn_mask=tree_attn_mask,
                        backend_cls=TreeAttentionBackend,
                    ).view(batch_size, -1, num_heads, dim_per_head)

                    # Verify that the chain attention output for each
                    # branch of the tree (computed using FA3) matches
                    # the tree attention output.
                    for q_index in range(tree_size_q):
                        # Get the q, k, and v for the branch.
                        branch_mask = tree_attn_mask[q_index, :]
                        branch_indices = torch.nonzero(branch_mask,
                                                       as_tuple=True)[0]
                        q_len = branch_indices.shape[0]
                        q_branch = q[:, branch_indices]
                        k_branch = k[:, branch_indices]
                        v_branch = v[:, branch_indices]

                        # Setup slot mapping for the branch.
                        branch_positions = sequence_position + torch.arange(
                            0,
                            q_len,
                            device=q.device,
                            dtype=torch.int64,
                        ).repeat(batch_size, 1)
                        branch_slot_mapping = _gen_slot_mapping(
                            branch_positions, block_table, block_size)

                        # Compute flash attention for the branch.
                        flash_attn_output = forward_attention(
                            q=q_branch,
                            k=k_branch,
                            v=v_branch,
                            kv_cache=kv_cache,
                            block_table=block_table,
                            slot_mapping=branch_slot_mapping,
                            seqlen_k=sequence_position + q_len,
                            qq_attn_mask=None,
                            backend_cls=FlashAttentionBackend,
                        ).view(batch_size, -1, num_heads, dim_per_head)

                        # Compare the outputs.
                        assert torch.allclose(
                            tree_attn_output[:, branch_indices],
                            flash_attn_output,
                            atol=7.81e-3,
                        ), (f"outputs are not close for "
                            f"batch_size: {batch_size}, "
                            f"num_heads: {num_heads}, "
                            f"sequence_position: {sequence_position}, "
                            f"tree_attn_mask: {tree_attn_mask}, "
                            f"q_index: {q_index}.")


def _gen_slot_mapping(positions: torch.Tensor, block_table: torch.Tensor,
                      block_size: int):
    block_indices = positions // block_size
    blocks = block_table.gather(dim=1, index=block_indices)
    return (blocks * block_size + positions % block_size).view(-1)
