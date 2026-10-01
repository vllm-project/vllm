# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BaseMambaAttentionMetadataBuilder batch classification."""

from types import SimpleNamespace

from tests.v1.attention.utils import MockMambaBuilder
from vllm.config.compilation import CUDAGraphMode


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
    assert metadata.has_initial_states_p.tolist() == [False]
