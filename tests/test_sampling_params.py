# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


# ---------------------------------------------------------------------------
# Tests for SamplingType enum
# Co-authored-by: Hermes Agent <hermes-agent@nousresearch.com>
# ---------------------------------------------------------------------------
from vllm.sampling_params import SamplingType  # noqa: E402


def test_sampling_type_greedy():
    """temperature=0 should yield GREEDY sampling."""
    p = SamplingParams(temperature=0.0)
    assert p.sampling_type is SamplingType.GREEDY


def test_sampling_type_random():
    """Non-zero temperature with no seed should yield RANDOM sampling."""
    p = SamplingParams(temperature=0.8)
    assert p.sampling_type is SamplingType.RANDOM


def test_sampling_type_random_seed():
    """Non-zero temperature with a seed should yield RANDOM_SEED sampling."""
    p = SamplingParams(temperature=0.8, seed=42)
    assert p.sampling_type is SamplingType.RANDOM_SEED


def test_sampling_type_values():
    """Enum integer values must remain stable (they are used as array indices)."""
    assert int(SamplingType.GREEDY) == 0
    assert int(SamplingType.RANDOM) == 1
    assert int(SamplingType.RANDOM_SEED) == 2


def test_sampling_type_docstring():
    """SamplingType must have a non-empty docstring."""
    assert SamplingType.__doc__ and len(SamplingType.__doc__.strip()) > 0
