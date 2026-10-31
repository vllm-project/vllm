# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BaseMambaAttentionMetadataBuilder batch classification."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import MambaSpec


def _make_vllm_config(
    max_model_len: int,
    max_num_seqs: int,
    num_speculative_tokens: int = 0,
    block_size: int | None = None,
):
    """Create a minimal mock VllmConfig with only the fields the builder
    accesses, avoiding any model download / HF config inspection."""
    speculative_config = (
        SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
        )
        if num_speculative_tokens > 0
        else None
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            mamba_cache_mode="none",
            use_replayssm=False,
            replayssm_buffer_len=16,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        num_speculative_tokens=num_speculative_tokens,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def test_mamba_single_token_prompt_runs_as_prefill():
    seq_lens = [8, 9, 1]
    config = _make_vllm_config(256, len(seq_lens), block_size=16)
    metadata = MockMambaBuilder.build_mamba_metadata(
        config,
        seq_lens=seq_lens,
        query_lens=[1] * len(seq_lens),
        is_prefilling=[False, False, True],
    )

    assert metadata.num_decodes == 2
    assert metadata.num_prefills == 1
    assert metadata.has_initial_states_p is not None
    assert metadata.has_initial_states_p.tolist() == [False]


@pytest.mark.parametrize("query_len", [1, 2, 4])
def test_mamba_captures_decode_graphs_up_to_one_plus_k(query_len):
    """A speculator may declare uniform decode shapes below 1 + K (a tree's
    chain fallback), and the CUDA graph manager captures one decode graph per
    shape: each builds decode-only metadata with every row's query length."""
    num_reqs, num_spec_tokens = 4, 3
    config = _make_vllm_config(
        256, 8, num_speculative_tokens=num_spec_tokens, block_size=16
    )
    spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(torch.float32,))
    builder = MockMambaBuilder(spec, ["layer0"], config, torch.device("cpu"))
    common = create_common_attn_metadata(
        # 1 + K state slots per request.
        BatchSpec(seq_lens=[64] * num_reqs, query_lens=[query_len] * num_reqs),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    ).replace(is_prefilling=torch.zeros(num_reqs, dtype=torch.bool))

    metadata = builder.build_for_cudagraph_capture(common)

    assert metadata.num_decodes == num_reqs
    assert metadata.num_prefills == 0
    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens[:num_reqs].tolist() == [query_len] * num_reqs
