# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that the V2 model runner limits max_model_len to the draft's (#57941).

The V2 speculators draft at every length the target accepts, so a draft model
whose max_model_len is smaller than the target's (and can't be raised to match)
must lower the target's max_model_len. The V1 model runner instead skips
speculation for longer batches, so it keeps the target's max_model_len.
"""

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig

# All repos are public; only config/tokenizer-config files are fetched.
AR_MODEL = "JackFram/llama-68m"  # max_position_embeddings=2048
EAGLE3_DRAFT = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"  # max_position_embeddings=2048
LLAMA3_TARGET = "unsloth/Meta-Llama-3.1-8B-Instruct"  # max_position_embeddings=131072


@pytest.mark.cpu_test
@pytest.mark.parametrize("use_v2_model_runner", [True, False])
def test_draft_model_limits_max_model_len(
    use_v2_model_runner: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    model_config = ModelConfig(AR_MODEL, hf_overrides={"max_position_embeddings": 8192})
    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=AR_MODEL,
        method="draft_model",
        num_speculative_tokens=3,
    )
    vllm_config = VllmConfig(
        model_config=model_config, speculative_config=speculative_config
    )
    assert vllm_config.use_v2_model_runner == use_v2_model_runner
    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config.max_model_len == 2048
    expected = 2048 if use_v2_model_runner else 8192
    assert vllm_config.model_config.max_model_len == expected


@pytest.mark.cpu_test
def test_eagle_draft_keeps_target_max_model_len(monkeypatch: pytest.MonkeyPatch):
    """EAGLE drafts have their max_position_embeddings raised to the target's
    max_model_len, so they don't limit it."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_config = ModelConfig(LLAMA3_TARGET, max_model_len=8192)
    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=EAGLE3_DRAFT,
        method="eagle3",
        num_speculative_tokens=3,
    )
    vllm_config = VllmConfig(
        model_config=model_config, speculative_config=speculative_config
    )
    assert vllm_config.use_v2_model_runner
    assert vllm_config.model_config.max_model_len == 8192
