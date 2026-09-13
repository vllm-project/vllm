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


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_extra_args_rejects_nested_integer_overflow(value):
    """Reject extension values before they reach the engine transport."""
    with pytest.raises(VLLMValidationError, match="extra_args integers"):
        SamplingParams(extra_args={"ec_transfer_params": {"nested": [{"x": value}]}})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True])
def test_extra_args_accepts_messagepack_integer_boundaries(value):
    extra_args = {"kv_transfer_params": {"nested": [{"x": value}]}}
    assert SamplingParams(extra_args=extra_args).extra_args == extra_args


def test_extra_args_preserves_custom_objects_and_shared_containers():
    custom = object()
    shared = [custom, (None, "value", 1.5)]
    extra_args = {"first": shared, "second": shared}
    params = SamplingParams(extra_args=extra_args)
    assert params.extra_args["first"][0] is custom
    assert params.extra_args["first"] is params.extra_args["second"]
