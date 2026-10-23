# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash-named Qwen3 DSpark drafts must take the dspark path (#49614)."""

import pytest

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import (
    SpeculativeConfig,
    _route_self_contained_qwen3_dspark_draft,
)

# All repos are public; only config/tokenizer-config files are fetched.
QWEN3_TARGET = "Qwen/Qwen3-4B"
DFLASH_DRAFT = "deepseek-ai/dflash_qwen3_4b_block7"  # Qwen3DSparkModel, markov_rank=0
DSPARK_DRAFT = "deepseek-ai/dspark_qwen3_4b_block7"  # Qwen3DSparkModel, markov_rank=256


def _speculative_config(model: str, method: str | None) -> SpeculativeConfig:
    return SpeculativeConfig(
        target_model_config=ModelConfig(QWEN3_TARGET, max_model_len=2048),
        target_parallel_config=ParallelConfig(),
        model=model,
        method=method,
        num_speculative_tokens=7,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["dflash", None])
def test_dflash_named_qwen3_dspark_draft_is_served_by_dspark(method: str | None):
    """Explicit and inferred ``dflash`` both resolve to the dspark path."""
    speculative_config = _speculative_config(DFLASH_DRAFT, method)
    assert speculative_config.method == "dspark"
    assert speculative_config.draft_model_config.architectures == ["Qwen3DSparkModel"]


@pytest.mark.cpu_test
def test_dspark_draft_is_unchanged():
    speculative_config = _speculative_config(DSPARK_DRAFT, "dspark")
    assert speculative_config.method == "dspark"
    assert speculative_config.draft_model_config.architectures == ["Qwen3DSparkModel"]


@pytest.mark.cpu_test
def test_standalone_dflash_draft_is_unchanged():
    """A standalone DFlash draft has its own registered architecture."""
    assert (
        _route_self_contained_qwen3_dspark_draft(
            "dflash", ["DFlashDraftModel"], "z-lab/Qwen3.5-9B-DFlash"
        )
        == "dflash"
    )
