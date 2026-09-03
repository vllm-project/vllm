# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests quantization configuration matching utilities."""

from unittest.mock import Mock

import pytest

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
    _find_matching_targets,
)
from vllm.model_executor.layers.quantization.quark.utils import (
    should_ignore_layer as quark_should_ignore_layer,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
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


def test_fnmatch_is_opt_in():
    layer_name = "model.layers.0.mlp.experts"
    pattern = "*mlp.experts*"

    assert not is_equal_or_regex_match(layer_name, pattern)
    assert is_equal_or_regex_match(layer_name, pattern, use_fnmatch=True)
    assert not should_ignore_layer(layer_name, [pattern])
    assert not quark_should_ignore_layer(layer_name, [pattern])
    assert should_ignore_layer(layer_name, [pattern], use_fnmatch=True)


def test_online_targets_support_fnmatch_patterns():
    layer_name = "model.layers.0.mlp.experts"
    targets = {"*mlp.experts*": "mxfp4"}

    assert _find_matching_targets(layer_name, targets) == ["*mlp.experts*"]


def test_online_ignore_supports_fnmatch_patterns():
    layer_name = "model.layers.0.mlp.experts"
    ignore = ["*mlp.experts*"]
    config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"*mlp.experts*": "mxfp4"}, ignore=ignore)
    )

    with pytest.raises(ValueError, match="matches both quantization_config.ignore"):
        config._dispatch_target(layer_name, Mock())


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


def test_should_ignore_layer_returns_false_when_no_fused_pattern_matches():
    layer_name = "model.layers.0.self_attn.qkv_proj"
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert find_matching_patterns(layer_name, [], fused_mapping) == [
        set(),
        set(),
        set(),
    ]
    assert not should_ignore_layer(layer_name, [], fused_mapping)


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


@pytest.mark.parametrize(
    "patterns",
    [
        ["model.layers.0.self_attn.qkv_proj"],
        [r"re:.*\.qkv_proj$"],
        [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
        ],
    ],
    ids=["direct_fused_name", "direct_fused_regex", "individual_shards"],
)
def test_quark_and_compressed_tensors_ignore_fused_layers_identically(patterns):
    layer_name = "model.layers.0.self_attn.qkv_proj"
    fused_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert should_ignore_layer(layer_name, patterns, fused_mapping)
    assert quark_should_ignore_layer(layer_name, patterns, fused_mapping)
