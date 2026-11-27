# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA ReplaySSM verify metadata gives each row its state block and the ring
block the KV cache manager reserves right after it."""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.models.glm5next.nvidia.kda_replayssm import KDAReplaySSMVerifyBackend
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
NUM_SPEC = 3
DEVICE = torch.device("cpu")


@pytest.mark.parametrize("enable_prefix_caching", [False, True], ids=["no_pc", "pc"])
def test_verify_rows_get_state_and_ring_block(enable_prefix_caching):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    vllm_config.cache_config.use_replayssm = True
    vllm_config.cache_config.enable_prefix_caching = enable_prefix_caching
    builder = KDAReplaySSMVerifyBackend.get_builder_cls()(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )

    seq_lens = [20, 47, 64]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=[NUM_SPEC + 1] * len(seq_lens)),
        BLOCK_SIZE,
        DEVICE,
    )
    # With prefix caching the table holds a block per BLOCK_SIZE tokens and the
    # ring block after the last; without it, just the state and ring blocks.
    num_cols = vllm_config.model_config.max_model_len // BLOCK_SIZE
    state_col = torch.tensor([(s - 1) // BLOCK_SIZE for s in seq_lens])
    if not enable_prefix_caching:
        num_cols, state_col = 2, torch.zeros_like(state_col)
    table = torch.arange(len(seq_lens) * num_cols, dtype=torch.int32).view(
        len(seq_lens), num_cols
    )
    common.block_table_tensor = table

    meta = builder.build(
        0,
        common,
        num_accepted_tokens=torch.ones(len(seq_lens), dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.full((len(seq_lens),), NUM_SPEC),
    )

    expected = table.gather(1, state_col[:, None] + torch.arange(2))
    assert meta.num_spec_decodes == len(seq_lens)
    torch.testing.assert_close(meta.spec_state_indices_tensor, expected)
