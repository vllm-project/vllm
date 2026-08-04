# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import PoolingParams, SamplingParams
from vllm.entrypoints.offline_utils import (
    OfflineInferenceMixin,
    _mix_prompt_seeds,
)


def _params_to_seq(params, num_requests):
    return OfflineInferenceMixin._params_to_seq(
        OfflineInferenceMixin(), params, num_requests
    )


def _mixed_seeds(params, prompts):
    seq = _mix_prompt_seeds(_params_to_seq(params, len(prompts)), prompts)
    return [p.seed for p in seq]


def test_mixed_seeds_unique_and_deterministic():
    """One seed shared over a batch must give each prompt an independent
    noise stream, not one shared noise vector (#50440)."""
    params = SamplingParams(seed=42)
    prompts = ["a", "b", "c"]

    seeds = _mixed_seeds(params, prompts)

    assert len(set(seeds)) == len(prompts)
    assert seeds == _mixed_seeds(params, prompts)
    # Original object is not mutated; other fields are preserved.
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


def test_unseeded_sampling_params_pass_through_unchanged():
    params = SamplingParams()
    seq = _mix_prompt_seeds(_params_to_seq(params, 3), ["a", "b", "c"])
    assert all(p is params for p in seq)


def test_pooling_params_pass_through_unchanged():
    params = PoolingParams()
    seq = _mix_prompt_seeds(_params_to_seq(params, 3), ["a", "b", "c"])
    assert all(p is params for p in seq)


def test_params_sequence_passthrough():
    params = [SamplingParams(seed=1), SamplingParams(seed=1)]
    assert _params_to_seq(params, 2) is params
    with pytest.raises(ValueError, match="must be the same"):
        _params_to_seq(params, 3)
