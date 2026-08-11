# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import PoolingParams, SamplingParams
from vllm.entrypoints.offline_utils import OfflineInferenceMixin, _mix_prompt_seeds
from vllm.exceptions import VLLMValidationError
from vllm.lora.request import LoRARequest


@pytest.fixture
def mixin() -> OfflineInferenceMixin:
    return object.__new__(OfflineInferenceMixin)


def test_rejects_mismatched_params(mixin: OfflineInferenceMixin):
    with pytest.raises(VLLMValidationError) as exc_info:
        mixin._params_to_seq([SamplingParams()], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and params (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_lora_requests(mixin: OfflineInferenceMixin):
    with pytest.raises(VLLMValidationError) as exc_info:
        mixin._lora_request_to_seq([None], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and lora_request (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_rejects_mismatched_priority(mixin: OfflineInferenceMixin):
    with pytest.raises(VLLMValidationError) as exc_info:
        mixin._priority_to_seq([0], num_requests=2)

    assert str(exc_info.value) == (
        "The lengths of prompts (2) and priority (1) must be the same."
    )
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


def test_matching_lengths_pass_through(mixin: OfflineInferenceMixin):
    lora = LoRARequest("l", 1, "/tmp/l")

    assert mixin._params_to_seq([SamplingParams()], num_requests=1) == [
        SamplingParams()
    ]
    assert mixin._lora_request_to_seq([lora], num_requests=1) == [lora]
    assert mixin._priority_to_seq([3], num_requests=1) == [3]


def _mixed_seeds(params, prompts):
    mixin = object.__new__(OfflineInferenceMixin)
    seq = _mix_prompt_seeds(mixin._params_to_seq(params, len(prompts)), prompts)
    return [p.seed for p in seq]


def test_mixed_seeds_unique_and_deterministic():
    """One seed shared over a batch must give each prompt an independent
    noise stream, not one shared noise vector (#50440)."""
    params = SamplingParams(seed=42)
    prompts = ["a", "b", "c"]

    seeds = _mixed_seeds(params, prompts)

    assert len(set(seeds)) == len(prompts)
    assert seeds == _mixed_seeds(params, prompts)
    # Original object is not mutated.
    assert params.seed == 42


def test_single_params_and_list_give_identical_seeds():
    """`generate([a], SamplingParams(seed=42))` and
    `generate([a], [SamplingParams(seed=42)])` must give identical
    results: mixing applies after expansion, uniformly to both forms."""
    prompts = ["a", "b"]
    broadcast = _mixed_seeds(SamplingParams(seed=42), prompts)
    explicit = _mixed_seeds([SamplingParams(seed=42), SamplingParams(seed=42)], prompts)
    assert broadcast == explicit


def test_mixed_seed_independent_of_batch_composition():
    """A prompt's seed depends on (seed, prompt), not on its batch position,
    so results are reproducible across reordering and subsetting."""
    params = SamplingParams(seed=7)

    by_prompt = dict(zip(["a", "b", "c"], _mixed_seeds(params, ["a", "b", "c"])))
    assert _mixed_seeds(params, ["c", "a", "b"]) == [
        by_prompt[x] for x in ["c", "a", "b"]
    ]
    assert _mixed_seeds(params, ["b"]) == [by_prompt["b"]]


def test_duplicate_prompts_share_seed():
    """Same (seed, prompt) always produces the same derived seed, so
    repeated identical prompts share their noise by construction; `n>1`
    is the mechanism for multiple distinct samples of one prompt."""
    seeds = _mixed_seeds(SamplingParams(seed=7), ["a", "a", "b"])

    assert seeds[0] == seeds[1]
    assert seeds[0] != seeds[2]


def test_mixed_seed_uses_token_ids_fingerprint():
    params = SamplingParams(seed=0)
    tokens_a = {"prompt_token_ids": [1, 2, 3]}
    tokens_b = {"prompt_token_ids": [1, 2, 4]}

    assert _mixed_seeds(params, [dict(tokens_a)]) == _mixed_seeds(
        params, [dict(tokens_a)]
    )
    assert _mixed_seeds(params, [tokens_a]) != _mixed_seeds(params, [tokens_b])


def test_mixed_seed_uses_image_content():
    """Prompts with the same text but different image content must derive
    different seeds; identical pixels hash identically."""
    from PIL import Image

    params = SamplingParams(seed=0)
    black = Image.new("RGB", (4, 4), color=0)
    white = Image.new("RGB", (4, 4), color=(255, 255, 255))
    black2 = Image.new("RGB", (4, 4), color=0)

    def mm_prompt(image):
        return {"prompt": "describe", "multi_modal_data": {"image": image}}

    assert _mixed_seeds(params, [mm_prompt(black)]) == _mixed_seeds(
        params, [mm_prompt(black2)]
    )
    assert _mixed_seeds(params, [mm_prompt(black)]) != _mixed_seeds(
        params, [mm_prompt(white)]
    )


def test_unseeded_sampling_params_pass_through_unchanged():
    mixin = object.__new__(OfflineInferenceMixin)
    params = SamplingParams()
    seq = _mix_prompt_seeds(mixin._params_to_seq(params, 3), ["a", "b", "c"])
    assert all(p is params for p in seq)


def test_pooling_params_pass_through_unchanged():
    mixin = object.__new__(OfflineInferenceMixin)
    params = PoolingParams()
    seq = _mix_prompt_seeds(mixin._params_to_seq(params, 3), ["a", "b", "c"])
    assert all(p is params for p in seq)
