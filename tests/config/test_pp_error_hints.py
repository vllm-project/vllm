# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for the pipeline-parallel `NotImplementedError` hints."""

from types import SimpleNamespace

import pytest

from vllm.config.model import ModelConfig


def _hint(architectures: list[str], is_draft_model: bool) -> str:
    """Invoke `_pp_unsupported_message` against a minimal `self`."""
    fake = SimpleNamespace(architectures=architectures)
    return ModelConfig._pp_unsupported_message(
        fake,  # type: ignore[arg-type]
        is_draft_model=is_draft_model,
    )


def test_target_model_gets_short_message() -> None:
    msg = _hint(["Llama3ForCausalLM"], is_draft_model=False)
    assert "Pipeline parallelism is not supported for this model" in msg
    assert "Llama3ForCausalLM" in msg
    # Draft-only guidance must not appear for a plain target-model failure.
    assert "draft" not in msg.lower()


@pytest.mark.parametrize(
    "arch",
    [
        "DeepSeekV4MTP",
        "Qwen3_5MTP",
        "KimiK3MTP",
        "EagleLlamaForCausalLM",
        "Eagle3Qwen3ForCausalLM",
        "DFlashLagunaForCausalLM",
        "DFlash2Qwen3ForCausalLM",
        "Qwen3DSparkForCausalLM",
    ],
)
def test_draft_looking_arch_gets_draft_hint(arch: str) -> None:
    msg = _hint([arch], is_draft_model=False)
    assert "speculative-decoding draft head" in msg
    assert "SupportsPP" in msg
    assert "make_empty_intermediate_tensors_factory" in msg
    assert arch in msg


def test_explicit_is_draft_forces_draft_hint() -> None:
    # Architecture name that doesn't look like a draft on its own, but the
    # caller (SpeculativeConfig._verify_args) says it is.
    msg = _hint(["LlamaForCausalLM"], is_draft_model=True)
    assert "speculative-decoding draft head" in msg
    assert "SupportsPP" in msg


def test_draft_hint_points_at_prior_fixes() -> None:
    msg = _hint(["DeepSeekV4MTP"], is_draft_model=True)
    # Point users at concrete precedent so they see the exact shape.
    for pr in ("#52069", "#46994", "#53408", "#54635", "#55081"):
        assert pr in msg
