# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DFlash-named self-contained DSpark drafts (#49614).

DeepSeek ships the SAME draft architecture for its DSpark (``markov_rank > 0``)
and DFlash (``markov_rank == 0``) checkpoints: ``dflash_qwen3_4b_block7`` and
``dspark_qwen3_4b_block7`` are both ``Qwen3DSparkModel`` (and the gemma4 pair is
``Gemma4DSparkModel``). ``Qwen3DSparkForCausalLM``/``Gemma4DSparkForCausalLM``
serve both, so these drafts must take the dspark path. With ``method="dflash"``
-- passed explicitly or inferred from the ``dflash`` model name -- the draft was
instead EAGLE-renamed to the unregistered ``DFlashQwen3DSparkModel`` and the
engine failed to start.
"""

import pytest

from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import (
    SpeculativeConfig,
    _route_self_contained_dspark_draft,
)

# All repos are public; only config/tokenizer-config files are fetched.
QWEN3_TARGET = "Qwen/Qwen3-4B"
DFLASH_DRAFT = "deepseek-ai/dflash_qwen3_4b_block7"  # Qwen3DSparkModel, markov_rank=0
DSPARK_DRAFT = "deepseek-ai/dspark_qwen3_4b_block7"  # Qwen3DSparkModel, markov_rank=256


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["dflash", None])
def test_dflash_named_qwen3_dspark_draft_is_served_by_dspark(method: str | None):
    """Explicit and inferred ``dflash`` both resolve to the dspark path."""
    speculative_config = SpeculativeConfig(
        target_model_config=ModelConfig(QWEN3_TARGET, max_model_len=2048),
        target_parallel_config=ParallelConfig(),
        model=DFLASH_DRAFT,
        method=method,
        num_speculative_tokens=7,
    )
    assert speculative_config.method == "dspark"
    assert speculative_config.draft_model_config.architectures == ["Qwen3DSparkModel"]


@pytest.mark.cpu_test
def test_dspark_draft_is_unchanged():
    speculative_config = SpeculativeConfig(
        target_model_config=ModelConfig(QWEN3_TARGET, max_model_len=2048),
        target_parallel_config=ParallelConfig(),
        model=DSPARK_DRAFT,
        method="dspark",
        num_speculative_tokens=7,
    )
    assert speculative_config.method == "dspark"
    assert speculative_config.draft_model_config.architectures == ["Qwen3DSparkModel"]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "architecture", ["Qwen3DSparkModel", "Qwen3OmniDSparkModel", "Gemma4DSparkModel"]
)
def test_self_contained_dspark_architectures_are_routed(architecture: str):
    assert (
        _route_self_contained_dspark_draft(
            "dflash", [architecture], "some/dflash_model"
        )
        == "dspark"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("architecture", ["DFlashDraftModel", "DFlash2DraftModel"])
def test_standalone_dflash_drafts_are_unchanged(architecture: str):
    """A standalone DFlash draft has its own registered architecture."""
    assert (
        _route_self_contained_dspark_draft(
            "dflash", [architecture], "z-lab/Qwen3.5-9B-DFlash"
        )
        == "dflash"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["dspark", "eagle3", "mtp", None])
def test_other_methods_are_unchanged(method: str | None):
    assert (
        _route_self_contained_dspark_draft(method, ["Qwen3DSparkModel"], "some/model")
        == method
    )
