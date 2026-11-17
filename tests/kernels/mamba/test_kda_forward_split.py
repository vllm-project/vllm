# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi KDA layer peels the decodes of a mixed non-spec batch off to the
recurrent kernel. Its output and state updates must match the chunk-only path
it replaces, reached here by reclassifying the same decodes as prefills."""

import dataclasses

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
PREFIX = "model.layers.0.self_attn"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA KDA kernels")
@pytest.mark.parametrize("num_decodes,prefill_lens", [(3, [512, 300]), (64, [257])])
@pytest.mark.parametrize("fresh_prefill", [False, True])
def test_mixed_batch_decode_split_matches_chunk_path(
    dist_init, num_decodes, prefill_lens, fresh_prefill
):
    torch.manual_seed(0)
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16, block_size=BLOCK_SIZE
    )
    config = KimiLinearConfig(
        hidden_size=256,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "head_dim": 128,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
        },
    )
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        layer = KimiGatedDeltaNetAttention(config, vllm_config, prefix=PREFIX)
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(std=0.1)

    # Decode-first batch: 1-token decodes with context, then the prefills.
    query_lens = [1] * num_decodes + prefill_lens
    seq_lens = [64] * num_decodes + [
        q if fresh_prefill and i == 0 else q + 37 for i, q in enumerate(prefill_lens)
    ]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BLOCK_SIZE,
        torch.device("cuda"),
        arange_block_indices=True,
    )
    # Block 0 is the null block, which no request holds.
    common = common.replace(block_table_tensor=common.block_table_tensor + 1)
    state_shapes, state_dtypes = layer.get_state_shape(), layer.get_state_dtype()
    builder = GDNAttentionMetadataBuilder(
        MambaSpec(block_size=BLOCK_SIZE, shapes=state_shapes, dtypes=state_dtypes),
        [PREFIX],
        vllm_config,
        torch.device("cuda"),
    )
    with set_current_vllm_config(vllm_config):
        split = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert split.num_decodes == num_decodes and split.spec_sequence_masks is None
    chunk_only = dataclasses.replace(
        split,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=split.num_decodes + split.num_prefills,
        num_prefill_tokens=split.num_decode_tokens + split.num_prefill_tokens,
        prefill_query_start_loc=split.non_spec_query_start_loc,
        prefill_state_indices=split.non_spec_state_indices_tensor,
        prefill_has_initial_state=split.has_initial_state,
    )

    num_slots = int(common.block_table_tensor.max()) + 1
    states0 = [
        torch.randn(num_slots, *shape, device="cuda").to(dtype) * 0.05
        for shape, dtype in zip(state_shapes, state_dtypes)
    ]
    hidden_states = torch.randn(
        sum(query_lens), config.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    positions = torch.zeros(sum(query_lens), dtype=torch.long, device="cuda")

    def run(meta):
        layer.kv_cache = tuple(state.clone() for state in states0)
        with set_forward_context({PREFIX: meta}, vllm_config):
            out = layer(hidden_states, positions)
        return out, layer.kv_cache

    out, (conv_state, recurrent_state) = run(split)
    ref_out, (ref_conv_state, ref_recurrent_state) = run(chunk_only)

    # Both paths convolve the whole batch with the same kernel.
    torch.testing.assert_close(conv_state, ref_conv_state, atol=0, rtol=0)
    # The recurrent and chunk kernels accumulate in different orders; the
    # largest difference seen on B200 is 2e-3 for both.
    tol = 5e-3
    torch.testing.assert_close(out, ref_out, atol=tol, rtol=tol)
    torch.testing.assert_close(recurrent_state, ref_recurrent_state, atol=tol, rtol=tol)
