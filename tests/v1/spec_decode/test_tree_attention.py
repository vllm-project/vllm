# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from tests.v1.attention.utils import (
    create_standard_kv_cache_spec,
    create_vllm_config,
    try_get_attention_backend,
)
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available
from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

if not is_flash_attn_varlen_func_available():
    pytest.skip(
        "This test requires flash_attn_varlen_func, but it's not available.",
        allow_module_level=True,
    )


class MockAttentionLayer(torch.nn.Module):
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
    backend: AttentionBackendEnum,
    spec_token_tree: str | None = None,
    num_spec_tokens: int = 0,
) -> torch.Tensor:
    batch_size, q_len, num_heads, dim_per_head = q.shape
    num_kv_heads = k.shape[-2]
    # Initialize the query and KV sequence lengths.
    query_start_loc = q_len * torch.arange(
        batch_size + 1, device=q.device, dtype=torch.int32
    )
    query_lens = torch.diff(query_start_loc)
    seq_lens = torch.full(
        (batch_size,),
        seqlen_k,
        device=q.device,
        dtype=torch.int32,
    )
    context_lens = seq_lens - query_lens
    max_seq_len = int(seq_lens.max())
    max_query_len = q_len
    num_actual_tokens = query_start_loc[-1]

    softmax_scale = q.shape[-1] ** (-0.5)
    layer = MockAttentionLayer()

    # Build common metadata.
    model_name = "meta-llama/Meta-Llama-3-8B"
    builder_cls, impl_cls = try_get_attention_backend(backend)
    vllm_config = create_vllm_config(model_name=model_name, max_model_len=max(seq_lens))
    if spec_token_tree is not None:
        # Create speculative config if token tree is specified.
        vllm_config.speculative_config = SpeculativeConfig(
            target_model_config=vllm_config.model_config,
            target_parallel_config=ParallelConfig(),
            model=model_name,
            method="eagle",
            num_speculative_tokens=num_spec_tokens,
            speculative_token_tree=spec_token_tree,
        )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    builder = builder_cls(kv_cache_spec, [], vllm_config, q.device)
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=context_lens.cpu(),
        num_reqs=batch_size,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
    )

    # Build attention metadata.
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Initialize the backend implementation.
    instance = impl_cls(
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
        attn_metadata=attn_metadata,
        output=output,
    )


def test_tree_attn_correctness() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda"
    tree_attn_masks = {
        # Chain.
        "[(0,), (0, 0), (0, 0, 0)]": torch.tensor(
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
        "[(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]": torch.tensor(
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
    }

    dim_per_head = 128
    num_kv_heads = 2
    block_size = 32
    max_sequence_length = 8192
    randomize_blocks = True
    for batch_size in [1, 16, 32]:
        for num_heads in [2, 4]:
            for sequence_position in [16, 1024, 2048]:
                for spec_token_tree, tree_attn_mask in tree_attn_masks.items():
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

                    # Set up the block table and KV cache for paged KV.
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
                    num_alloc_blocks_per_batch = math.ceil(seqlen_k / block_size)
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
                        block_ids = block_ids[torch.randperm(block_ids.numel())]
                    block_table[:, :num_alloc_blocks_per_batch] = block_ids.view(
                        -1, num_alloc_blocks_per_batch
                    )

                    # Set up the slot mapping for the input KVs.
                    tree_positions = sequence_position + torch.arange(
                        0,
                        tree_size_q,
                        device=q.device,
                        dtype=torch.int64,
                    ).repeat(batch_size, 1)
                    tree_slot_mapping = _gen_slot_mapping(
                        tree_positions, block_table, block_size
                    )

                    # Compute attention for the tree.
                    tree_attn_output = forward_attention(
                        q=q,
                        k=k,
                        v=v,
                        kv_cache=kv_cache,
                        block_table=block_table,
                        slot_mapping=tree_slot_mapping,
                        seqlen_k=seqlen_k,
                        backend=AttentionBackendEnum.TREE_ATTN,
                        spec_token_tree=spec_token_tree,
                        num_spec_tokens=tree_size_q - 1,
                    ).view(batch_size, -1, num_heads, dim_per_head)

                    # Verify that the chain attention output for each
                    # branch of the tree (computed using FA3) matches
                    # the tree attention output.
                    for q_index in range(tree_size_q):
                        # Get the q, k, and v for the branch.
                        branch_mask = tree_attn_mask[q_index, :]
                        branch_indices = torch.nonzero(branch_mask, as_tuple=True)[0]
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
                            branch_positions, block_table, block_size
                        )

                        # Compute flash attention for the branch.
                        flash_attn_output = forward_attention(
                            q=q_branch,
                            k=k_branch,
                            v=v_branch,
                            kv_cache=kv_cache,
                            block_table=block_table,
                            slot_mapping=branch_slot_mapping,
                            seqlen_k=sequence_position + q_len,
                            backend=AttentionBackendEnum.FLASH_ATTN,
                        ).view(batch_size, -1, num_heads, dim_per_head)

                        # Compare the outputs.
                        assert torch.allclose(
                            tree_attn_output[:, branch_indices],
                            flash_attn_output,
                            atol=7.81e-3,
                        ), (
                            f"outputs are not close for "
                            f"batch_size: {batch_size}, "
                            f"num_heads: {num_heads}, "
                            f"sequence_position: {sequence_position}, "
                            f"tree_attn_mask: {tree_attn_mask}, "
                            f"q_index: {q_index}."
                        )


def _gen_slot_mapping(
    positions: torch.Tensor, block_table: torch.Tensor, block_size: int
):
    block_indices = positions // block_size
    blocks = block_table.gather(dim=1, index=block_indices)
    return (blocks * block_size + positions % block_size).view(-1)
