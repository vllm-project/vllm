# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
from xformers.ops.fmha.attn_bias import PagedBlockDiagonalPaddedKeysMask

from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.tree_attn import TreeAttentionBackend


class NoOpLayerModule(torch.nn.Module):
    _q_scale = torch.tensor(1.0, dtype=torch.float32)
    _k_scale = torch.tensor(1.0, dtype=torch.float32)
    _v_scale = torch.tensor(1.0, dtype=torch.float32)

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def forward_attention(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    dim_per_head: int,
    block_size: int,
    max_sequence_length: int,
    sequence_position: int,
    q_len: int,
    backends: list[type[AttentionBackend]],
    randomize_blocks: bool,
) -> list[torch.Tensor]:
    # Assert that the number of heads is divisible by the number of KV heads.
    assert num_heads % num_kv_heads == 0

    device = "cuda"
    # Initialize q, k, and v.
    q = torch.randn(
        (batch_size * q_len, num_heads, dim_per_head),
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (batch_size * q_len, num_kv_heads, dim_per_head),
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        (batch_size * q_len, num_kv_heads, dim_per_head),
        device=device,
        dtype=torch.bfloat16,
    )

    # Initialize the query and KV sequence lengths.
    cu_seqlens_q = q_len * torch.arange(
        batch_size + 1, device=device, dtype=torch.int32)
    seqlens_q = torch.diff(cu_seqlens_q)
    seqlens_kv = torch.full(
        (batch_size, ),
        sequence_position + q_len,
        device=device,
        dtype=torch.int32,
    )
    max_seqlen_q = q_len
    max_seqlen_k = sequence_position + q_len
    num_actual_tokens = cu_seqlens_q[-1]

    # Setup the block table and KV cache for paged KV.
    assert max_sequence_length % block_size == 0
    max_block_count_per_batch = max_sequence_length // block_size
    kv_cache = torch.randn(
        (
            2,
            batch_size * max_block_count_per_batch,
            block_size,
            num_kv_heads,
            dim_per_head,
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    num_allocated_blocks_per_batch = math.ceil(max_seqlen_k / block_size)
    block_table = torch.zeros(
        (batch_size, max_block_count_per_batch),
        device=device,
        dtype=torch.int32,
    )
    block_ids = torch.arange(
        0,
        batch_size * num_allocated_blocks_per_batch,
        device=device,
        dtype=torch.int32,
    )
    if randomize_blocks:
        block_ids = block_ids[torch.randperm(block_ids.numel())]
    block_table[:, :num_allocated_blocks_per_batch] = block_ids.view(
        -1, num_allocated_blocks_per_batch)

    # Setup the slot mapping for the input KVs.
    positions = sequence_position + torch.arange(
        0,
        q_len,
        device=device,
        dtype=torch.int64,
    ).repeat(batch_size, 1)
    block_indices = positions // block_size
    blocks = block_table.gather(dim=1, index=block_indices)
    slot_mapping = (blocks * block_size + positions % block_size).view(-1)

    softmax_scale = q.shape[-1]**(-0.5)
    layer = NoOpLayerModule()

    # Run attention for each backend and collect the outputs.
    outputs = []
    for backend_cls in backends:
        # Set common metadata.
        attn_metadata_dict = {
            "num_actual_tokens": num_actual_tokens,
            "max_query_len": max_seqlen_q,
            "query_start_loc": cu_seqlens_q,
            "max_seq_len": max_seqlen_k,
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
            # Construct the prefix bias.
            prefix_kv_seqlens = seqlens_kv - seqlens_q
            prefix_attn_bias = PagedBlockDiagonalPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens_q.tolist(),
                kv_seqlen=prefix_kv_seqlens.tolist(),
                page_size=block_size,
                block_tables=block_table,
                device=device,
            )
            attn_metadata_dict["prefix_attn_bias"] = prefix_attn_bias
            # Create a chain attn bias.
            chain_attn_bias = torch.triu(
                torch.full((q_len, q_len),
                           float("-inf"),
                           device=device,
                           dtype=torch.bfloat16),
                diagonal=1,
            )
            attn_metadata_dict["spec_attn_bias"] = chain_attn_bias
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

        # Run forward pass and store output.
        output = torch.empty_like(q)
        outputs.append(
            instance.forward(
                layer=layer,
                query=q,
                key=k,
                value=v,
                kv_cache=kv_cache.clone(),
                attn_metadata=backend_cls.get_metadata_cls()(
                    **attn_metadata_dict),
                output=output,
            ))
    return outputs


def test_tree_attn_correctness() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    for batch_size in [1, 2, 16, 32, 64]:
        for num_heads in [2, 4]:
            for sequence_position in [16, 1024, 2048]:
                for q_len in [1, 3, 7]:
                    flash_attn_output, tree_attn_output = forward_attention(
                        batch_size=batch_size,
                        num_heads=num_heads,
                        num_kv_heads=2,
                        dim_per_head=128,
                        block_size=128,
                        max_sequence_length=8192,
                        sequence_position=sequence_position,
                        q_len=q_len,
                        backends=[FlashAttentionBackend, TreeAttentionBackend],
                        randomize_blocks=True,
                    )
                    assert torch.allclose(
                        flash_attn_output, tree_attn_output, atol=7.81e-3
                    ), (f"outputs are not close for batch_size: {batch_size}, "
                        f"num_heads: {num_heads}, "
                        f"sequence_position: {sequence_position}, "
                        f"q_len: {q_len}.")
