# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec


def _make_common_attention_metadata(
    *,
    query_start_loc: list[int],
    seq_lens: list[int],
    block_table: list[list[int]],
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc_tensor = torch.tensor(
        query_start_loc, dtype=torch.int32, device=device
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    num_actual_tokens = query_start_loc[-1]
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc_tensor,
        query_start_loc_cpu=query_start_loc_tensor.cpu(),
        seq_lens=seq_lens_tensor,
        num_reqs=len(seq_lens),
        num_actual_tokens=num_actual_tokens,
        max_query_len=max(
            end - start for start, end in zip(query_start_loc[:-1], query_start_loc[1:])
        ),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.tensor(block_table, dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(num_actual_tokens, dtype=torch.int64, device=device),
    )


def test_linear_attn_builder_cache_all_keeps_generic_metadata_minimal():
    device = torch.device("cpu")
    builder = LinearAttentionMetadataBuilder(
        MambaSpec(
            block_size=8,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="all",
        ),
        ["model.layers.0"],
        SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="all")),
        device,
    )
    common_metadata = _make_common_attention_metadata(
        query_start_loc=[0, 1, 4],
        seq_lens=[8, 10],
        block_table=[[10, 11], [20, 21]],
        device=device,
    )

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_metadata)

    assert metadata.num_decodes == 1
    assert metadata.num_prefills == 1
    assert metadata.num_decode_tokens == 1
    assert metadata.num_prefill_tokens == 3
    torch.testing.assert_close(
        metadata.state_indices_tensor,
        common_metadata.block_table_tensor,
    )
    torch.testing.assert_close(
        metadata.num_computed_tokens,
        torch.tensor([7, 7], dtype=torch.int32, device=device),
    )
    assert not hasattr(metadata, "block_idx_last_computed_token")
    assert not hasattr(metadata, "block_idx_first_scheduled_token")
    assert not hasattr(metadata, "block_idx_last_scheduled_token")
