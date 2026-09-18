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


@dataclass
class MockDiffusionConfig:
    canvas_length: int = 8


def _verify_diffusion(params: SamplingParams, canvas_length: int | None = None):
    diffusion_config = (
        MockDiffusionConfig(canvas_length) if canvas_length is not None else None
    )
    params.verify(
        MockModelConfig(is_diffusion=True),
        None,
        None,
        None,
        diffusion_config=diffusion_config,
    )


@pytest.mark.parametrize(
    "extra_args, match",
    [
        ({"diffusion_seed_canvas": "abc"}, "list of token ids"),
        ({"diffusion_seed_canvas": [1, 2.5]}, "list of token ids"),
        ({"diffusion_seed_canvas": [1, True]}, "list of token ids"),
        ({"diffusion_seed_canvas": [0, 1024]}, r"in \[0, 1024\)"),
        ({"diffusion_seed_canvas": [0, -1]}, r"in \[0, 1024\)"),
        ({"diffusion_max_steps": 0}, "positive integer"),
        ({"diffusion_max_steps": "1"}, "positive integer"),
        ({"diffusion_max_steps": True}, "positive integer"),
        ({"diffusion_read_only": "yes"}, "boolean"),
        ({"diffusion_read_only": 2}, "boolean"),
    ],
)
def test_diffusion_rejects_bad_extra_args(extra_args: dict, match: str):
    with pytest.raises(VLLMValidationError, match=match):
        _verify_diffusion(SamplingParams(extra_args=extra_args))


def test_diffusion_seed_canvas_must_fill_the_canvas():
    params = SamplingParams(extra_args={"diffusion_seed_canvas": [0] * 7})
    with pytest.raises(VLLMValidationError, match="exactly 8 ids, got 7"):
        _verify_diffusion(params, canvas_length=8)
    # Without the served diffusion config the canvas length is unknown.
    _verify_diffusion(params)


@pytest.mark.parametrize("max_tokens, expected", [(100, 8), (5, 5), (None, 8)])
@pytest.mark.parametrize("flag", [True, 1])
def test_diffusion_read_only_ends_after_one_canvas(max_tokens, expected, flag):
    params = SamplingParams(
        max_tokens=max_tokens, extra_args={"diffusion_read_only": flag}
    )
    _verify_diffusion(params, canvas_length=8)
    assert params.max_tokens == expected
    assert params.ignore_eos


@pytest.mark.parametrize(
    "width, match",
    [(0, "positive integer"), ("4", "positive integer"), (True, "positive integer"), (9, "no larger")],
)
def test_diffusion_rejects_bad_canvas_length(width, match):
    with pytest.raises(VLLMValidationError, match=match):
        _verify_diffusion(
            SamplingParams(extra_args={"diffusion_canvas_length": width}), canvas_length=8
        )


def test_diffusion_canvas_length_sizes_the_seed_and_the_read():
    params = SamplingParams(
        max_tokens=100,
        extra_args={
            "diffusion_canvas_length": 4,
            "diffusion_seed_canvas": [1, 2, 3, 4],
            "diffusion_read_only": True,
        },
    )
    _verify_diffusion(params, canvas_length=8)
    assert params.max_tokens == 4

    params = SamplingParams(
        extra_args={"diffusion_canvas_length": 4, "diffusion_seed_canvas": [0] * 8}
    )
    with pytest.raises(VLLMValidationError, match="exactly 4 ids, got 8"):
        _verify_diffusion(params, canvas_length=8)


def test_diffusion_accepts_extra_args():
    params = SamplingParams(
        extra_args={
            "diffusion_seed_canvas": list(range(8)),
            "diffusion_max_steps": 1,
            "diffusion_read_only": True,
        }
    )
    _verify_diffusion(params, canvas_length=8)


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
