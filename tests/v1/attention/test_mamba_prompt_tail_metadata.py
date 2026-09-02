# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for speculative padding on Mamba prompt-tail rows."""

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
NUM_SPEC_TOKENS = 5
DEVICE = torch.device("cpu")


def _build(
    seq_lens: list[int],
    query_lens: list[int],
    is_prefilling: list[bool],
    num_decode_draft_tokens: list[int],
):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC_TOKENS
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
    )
    builder = MockMambaBuilder(mamba_spec, ["layer0"], vllm_config, DEVICE)
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE, arange_block_indices=True
    ).replace(is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool))
    return builder.build(
        0,
        common,
        num_accepted_tokens=torch.ones(len(seq_lens), dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        ),
    )


def test_padded_prompt_tail_is_speculative_decode():
    """A remote prompt-tail token plus K placeholders needs rollback-capable state."""
    meta = _build(
        seq_lens=[100],
        query_lens=[NUM_SPEC_TOKENS + 1],
        is_prefilling=[True],
        num_decode_draft_tokens=[NUM_SPEC_TOKENS],
    )

    assert meta.num_decodes == 1
    assert meta.num_decode_tokens == NUM_SPEC_TOKENS + 1
    assert meta.num_prefills == 0
    assert meta.query_start_loc_d is not None
    assert meta.query_start_loc_d.tolist() == [0, NUM_SPEC_TOKENS + 1]
    assert meta.state_indices_tensor_d.shape == (1, NUM_SPEC_TOKENS + 1)


def test_real_six_token_prefill_is_not_reclassified():
    """The scheduler tag, not q=K+1 alone, distinguishes real prefill work."""
    meta = _build(
        seq_lens=[100],
        query_lens=[NUM_SPEC_TOKENS + 1],
        is_prefilling=[True],
        num_decode_draft_tokens=[-1],
    )

    assert meta.num_decodes == 0
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == NUM_SPEC_TOKENS + 1


def test_first_prompt_chunk_stays_prefill_even_if_tagged():
    """Without prior state, decode/update kernels have no checkpoint to extend."""
    meta = _build(
        seq_lens=[NUM_SPEC_TOKENS + 1],
        query_lens=[NUM_SPEC_TOKENS + 1],
        is_prefilling=[True],
        num_decode_draft_tokens=[NUM_SPEC_TOKENS],
    )

    assert meta.num_decodes == 0
    assert meta.num_prefills == 1
