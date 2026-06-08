# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import CUDAGraphMode, SpeculativeConfig
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


def _create_mamba_spec(num_speculative_blocks: int = 1) -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_speculative_blocks=num_speculative_blocks,
    )


def test_bailing_linear_attention_reports_uniform_batch_cudagraph_support():
    vllm_config = create_vllm_config(
        hf_config_override={
            "architectures": ["BailingMoeV2_5ForCausalLM"],
            "model_type": "bailing_hybrid",
        }
    )

    support = LinearAttentionMetadataBuilder.get_cudagraph_support(
        vllm_config, _create_mamba_spec()
    )

    assert support == AttentionCGSupport.UNIFORM_BATCH


def test_non_bailing_linear_attention_keeps_single_token_cudagraph_support():
    vllm_config = create_vllm_config(
        hf_config_override={
            "architectures": ["MiniMaxText01ForCausalLM"],
            "model_type": "minimax_text_01",
        }
    )

    support = LinearAttentionMetadataBuilder.get_cudagraph_support(
        vllm_config, _create_mamba_spec()
    )

    assert support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


def test_linear_attention_spec_decode_full_graph_metadata_pads_cache_slots():
    vllm_config = create_vllm_config(
        hf_config_override={
            "architectures": ["BailingMoeV2_5ForCausalLM"],
            "model_type": "bailing_hybrid",
        }
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

    builder = LinearAttentionMetadataBuilder(
        kv_cache_spec=_create_mamba_spec(),
        layer_names=["model.layers.0.self_attn"],
        vllm_config=vllm_config,
        device=DEVICE,
    )

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[20, 20, 0], query_lens=[2, 2, 0]),
        BLOCK_SIZE,
        DEVICE,
    )
    common.block_table_tensor[2].fill_(-1)

    metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([1, 2, 1], dtype=torch.int32),
    )

    assert metadata.num_decodes == 3
    assert metadata.num_prefills == 0
    assert metadata.num_decode_tokens == 4
    assert metadata.state_indices_tensor_d is not None
    assert metadata.state_indices_tensor_d.shape == (3, 2)
    assert torch.equal(
        metadata.state_indices_tensor_d[2],
        torch.full((2,), PAD_SLOT_ID, dtype=torch.int32),
    )
    assert torch.equal(
        metadata.state_indices_tensor[2],
        torch.tensor(PAD_SLOT_ID, dtype=torch.int32),
    )
    assert metadata.query_start_loc_d is not None
    assert metadata.query_start_loc_d.tolist() == [0, 2, 4, 4]
    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens.tolist() == [1, 2, 1]


def test_linear_attention_full_graph_metadata_uses_stable_decode_buffers():
    vllm_config = create_vllm_config(
        hf_config_override={
            "architectures": ["BailingMoeV2_5ForCausalLM"],
            "model_type": "bailing_hybrid",
        }
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
    )
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY

    builder = LinearAttentionMetadataBuilder(
        kv_cache_spec=_create_mamba_spec(),
        layer_names=["model.layers.0.self_attn"],
        vllm_config=vllm_config,
        device=DEVICE,
    )

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[20, 20, 0], query_lens=[2, 2, 0]),
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    common.block_table_tensor = torch.tensor(
        [[10, 11], [12, 13], [-1, -1]],
        dtype=torch.int32,
        device=DEVICE,
    )

    first = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=torch.tensor([1, 2, 1], dtype=torch.int32),
    )
    assert first.state_indices_tensor_d is not None
    assert first.query_start_loc_d is not None
    assert first.num_accepted_tokens is not None
    state_ptr = first.state_indices_tensor_d.data_ptr()
    query_ptr = first.query_start_loc_d.data_ptr()
    accepted_ptr = first.num_accepted_tokens.data_ptr()

    common2 = create_common_attn_metadata(
        BatchSpec(seq_lens=[36, 0, 0], query_lens=[2, 0, 0]),
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    common2.block_table_tensor = torch.tensor(
        [[20, 21], [-1, -1], [-1, -1]],
        dtype=torch.int32,
        device=DEVICE,
    )
    second = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common2,
        num_accepted_tokens=torch.tensor([2, 1, 1], dtype=torch.int32),
    )

    assert second.state_indices_tensor_d is not None
    assert second.query_start_loc_d is not None
    assert second.num_accepted_tokens is not None
    assert second.state_indices_tensor_d.data_ptr() == state_ptr
    assert second.query_start_loc_d.data_ptr() == query_ptr
    assert second.num_accepted_tokens.data_ptr() == accepted_ptr
    assert second.state_indices_tensor_d.tolist() == [
        [20, 21],
        [PAD_SLOT_ID, PAD_SLOT_ID],
        [PAD_SLOT_ID, PAD_SLOT_ID],
    ]
    assert second.query_start_loc_d.tolist() == [0, 2, 2, 2]
    assert second.num_accepted_tokens.tolist() == [2, 1, 1]
