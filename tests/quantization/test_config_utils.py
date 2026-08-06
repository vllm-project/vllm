# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests quantization configuration matching utilities."""

import pytest

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.online.base import (
    _find_matching_targets,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    check_equal_or_regex_match,
    find_matching_patterns,
    get_layer_name_after_index,
    is_equal_or_regex_match,
)


def test_is_equal_or_regex_match():
    assert is_equal_or_regex_match(
        "model.layers.0.mlp.down_proj", "model.layers.0.mlp.down_proj"
    )
    assert is_equal_or_regex_match("model.layers.0.mlp.down_proj", r"re:.*down_proj")
    assert not is_equal_or_regex_match("model.layers.0.mlp.down_proj", "other")


def test_check_equal_or_regex_match():
    assert check_equal_or_regex_match(
        "model.layers.0.mlp.down_proj",
        ["other", r"re:.*down_proj"],
    )
    assert not check_equal_or_regex_match("model.layers.0.mlp.down_proj", ["other"])


@pytest.mark.parametrize(
    "layer_name,expected",
    [
        ("model.layers.1.self_attn.qkv_proj", "self_attn.qkv_proj"),
        ("model.layers.2.mlp.down_proj", "mlp.down_proj"),
        ("lm_head", "lm_head"),
    ],
)
def test_get_layer_name_after_index(layer_name, expected):
    assert get_layer_name_after_index(layer_name) == expected


@pytest.mark.parametrize(
    "patterns",
    [
        [r"re:.*qkv_proj.*"],
        [r"re:.*\.[qkv]_proj$"],
    ],
    ids=["direct_fused_regex", "fused_shard_regexes"],
)
def test_find_matching_patterns_for_fused_regexes(patterns):
    layer_name = "model.layers.0.self_attn.qkv_proj"
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    matches = find_matching_patterns(layer_name, patterns, fused_mapping)
    assert all(len(shard_matches) == 1 for shard_matches in matches)


@pytest.mark.parametrize(
    "patterns,expected",
    [
        (
            [r"re:.*\.qkv_proj$"],
            [{r"re:.*\.qkv_proj$"}],
        ),
        (
            [r"re:.*\.q_proj$", r"re:.*\.k_proj$", r"re:.*\.v_proj$"],
            [
                {r"re:.*\.q_proj$"},
                {r"re:.*\.k_proj$"},
                {r"re:.*\.v_proj$"},
            ],
        ),
    ],
    ids=["direct_fused_match", "individual_shard_matches"],
)
def test_find_matching_patterns_distinguishes_direct_and_shard_matches(
    patterns, expected
):
    layer_name = "model.layers.0.self_attn.qkv_proj"
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert find_matching_patterns(layer_name, patterns, fused_mapping) == expected
    assert should_ignore_layer(layer_name, patterns, fused_mapping)


def test_should_ignore_layer_rejects_partially_matched_fused_layer():
    layer_name = "model.layers.0.self_attn.qkv_proj"
    patterns = [r"re:.*\.q_proj$"]
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert find_matching_patterns(layer_name, patterns, fused_mapping) == [
        {r"re:.*\.q_proj$"},
        set(),
        set(),
    ]
    with pytest.raises(ValueError, match="different quantization schemes"):
        should_ignore_layer(layer_name, patterns, fused_mapping)


@pytest.mark.parametrize(
    "patterns",
    [
        [r"re:.*qkv_proj.*"],
        [r"re:.*\.[qkv]_proj$"],
    ],
    ids=["direct_fused_regex", "fused_shard_regexes"],
)
def test_ignore_and_targets_match_fused_regexes_identically(patterns):
    layer_name = "model.layers.0.self_attn.qkv_proj"
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    targets = {pattern: "fp8_per_block" for pattern in patterns}

    assert should_ignore_layer(layer_name, patterns, fused_mapping)
    matches = _find_matching_targets(layer_name, targets, fused_mapping)
    assert len(matches) == 1
    assert targets[matches[0]] == "fp8_per_block"


def test_ignore_allows_individually_matched_fused_shards():
    layer_name = "model.layers.0.self_attn.qkv_proj"
    patterns = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert should_ignore_layer(layer_name, patterns, fused_mapping)
