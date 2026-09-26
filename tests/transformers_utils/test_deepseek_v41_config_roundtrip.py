# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A V4.1 config must survive its own to_dict()/from_dict()."""

import pytest

from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

# Every flattened vision field, with a value distinct from its default so a
# field that silently reverts is visible.
# Every value distinct, so a swapped flat name and nested key shows up as a
# wrong value rather than coincidentally matching another field's.
NESTED = {
    "num_hidden_layers": 3,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "intermediate_size": 5632,
    "patch_size": 16,
    "rope_theta": 50000.0,
    "downsample_ratio": 2,
    "max_image_tokens": 4096,
    "min_pixels": 65536,
    "max_wh_ratio": 4.0,
}
FLAT = {
    "vision_n_layers": 3,
    "vision_dim": 2048,
    "vision_n_heads": 32,
    "vision_inter_dim": 5632,
    "vision_patch_size": 16,
    "vision_rope_theta": 50000.0,
    "vision_downsample_ratio": 2,
    "vision_max_n_token": 4096,
    "vision_min_pixels": 65536,
    "vision_max_wh_ratio": 4.0,
}


def test_every_vision_field_survives_a_round_trip():
    """to_dict() emits flat names and no vision_config.

    Rebuilding from that dict used to reset the tower to the text-only
    defaults, silently turning a multimodal checkpoint into a text model
    wherever a config is serialized (worker handoff, caching, save_pretrained).
    """
    config = DeepseekV41Config(vision_config=NESTED)
    rebuilt = DeepseekV41Config.from_dict(config.to_dict())

    for field, value in FLAT.items():
        assert getattr(config, field) == value, f"{field} before round-trip"
        assert getattr(rebuilt, field) == value, f"{field} after round-trip"


def test_the_nested_block_outranks_a_flat_value():
    """Loading a checkpoint must not be overridden by a stale flat field."""
    config = DeepseekV41Config(
        vision_config={"num_hidden_layers": 5}, vision_n_layers=9
    )

    assert config.vision_n_layers == 5


def test_a_key_absent_from_a_populated_nested_block_falls_back():
    config = DeepseekV41Config(vision_config={"num_hidden_layers": 5}, vision_dim=777)

    assert config.vision_n_layers == 5
    assert config.vision_dim == 777  # flat value, not the 1024 default


def test_an_explicit_none_in_the_nested_block_is_kept():
    """`None` is a real value for max_wh_ratio, not "unset"."""
    config = DeepseekV41Config(
        vision_config={"max_wh_ratio": None}, vision_max_wh_ratio=8.0
    )

    assert config.vision_max_wh_ratio is None


@pytest.mark.parametrize(
    "field,expected",
    sorted(
        {
            "vision_n_layers": 0,
            "vision_dim": 1024,
            "vision_n_heads": 16,
            "vision_inter_dim": 2816,
            "vision_patch_size": 14,
            "vision_rope_theta": 10000.0,
            "vision_downsample_ratio": 3,
            "vision_max_n_token": 1024,
            "vision_min_pixels": 295936,
            "vision_max_wh_ratio": None,
        }.items()
    ),
)
def test_text_only_defaults_are_unchanged(field, expected):
    assert getattr(DeepseekV41Config(), field) == expected


def test_a_stale_vision_key_in_text_config_is_not_vision_configuration():
    """text_config is flattened onto the same object as the vision fields.

    Reading the fallback off `self` would let a leftover `vision_*` key in
    `text_config` switch the tower on with no vision block in sight.
    """
    config = DeepseekV41Config(text_config={"vision_n_layers": 3})

    assert config.vision_n_layers == 0


def test_the_tower_survives_a_trip_through_disk(tmp_path):
    """save_pretrained/from_pretrained is the path a served config takes."""
    config = DeepseekV41Config(vision_config=NESTED)
    config.save_pretrained(tmp_path)

    reloaded = DeepseekV41Config.from_pretrained(tmp_path)

    for field, value in FLAT.items():
        assert getattr(reloaded, field) == value, field
