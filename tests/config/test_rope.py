# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.rope import RequestStaticYarnConfig


def make_hf_config(**overrides):
    rope_parameters = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262_144,
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
        **overrides,
    }
    return SimpleNamespace(rope_parameters=rope_parameters)


def test_request_static_yarn_profiles_are_stable_and_cover_boundaries():
    hf_config = make_hf_config()

    config = RequestStaticYarnConfig.from_hf_config([1.0, 2.0, 4.0], hf_config)

    assert config is not None
    assert config.factor_offsets == (
        (1.0, 0),
        (2.0, 1_048_576),
        (4.0, 3_145_728),
    )
    assert config.select_factor(262_144) == 1.0
    assert config.select_factor(262_145) == 2.0
    assert config.select_factor(524_289) == 4.0
    assert len({profile_id for _, profile_id in config.factor_profile_ids}) == 3

    config.apply_to_hf_config(hf_config)
    assert hf_config.rope_parameters["request_static_factors"] == [1.0, 2.0, 4.0]


def test_request_static_yarn_profile_ids_depend_on_rope_parameters():
    first = RequestStaticYarnConfig.from_hf_config([1.0, 4.0], make_hf_config())
    second = RequestStaticYarnConfig.from_hf_config(
        [1.0, 4.0], make_hf_config(rope_theta=1_000_000)
    )

    assert first is not None
    assert second is not None
    assert first.profile_id_for_factor(1.0) != second.profile_id_for_factor(1.0)


@pytest.mark.parametrize(
    ("factors", "overrides", "match"),
    [
        ([2.0, 1.0], {}, "unique, and sorted"),
        ([0.5, 4.0], {}, "at least 1"),
        ([1.0, 2.0], {}, "largest request-static factor"),
        ([1.0, 4.0], {"rope_type": "linear"}, "rope_type='yarn'"),
    ],
)
def test_request_static_yarn_rejects_invalid_profiles(factors, overrides, match):
    with pytest.raises(ValueError, match=match):
        RequestStaticYarnConfig.from_hf_config(factors, make_hf_config(**overrides))


def test_request_static_yarn_rejects_uncovered_budget():
    config = RequestStaticYarnConfig.from_hf_config([1.0, 4.0], make_hf_config())
    assert config is not None

    with pytest.raises(ValueError, match="largest request-static YaRN profile"):
        config.select_factor(1_048_577)
