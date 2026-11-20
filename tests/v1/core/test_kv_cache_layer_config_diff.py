# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The cross-worker KV cache config check must say which field differs."""

import dataclasses

from vllm.v1.core.kv_cache_planning import _layer_config_field_diff


@dataclasses.dataclass
class _LayerConfig:
    block_size: int
    page_size_padded: int | None
    name: str = "layer.0"


def test_diff_reports_only_the_changed_fields():
    first = _LayerConfig(block_size=256, page_size_padded=None)
    second = _LayerConfig(block_size=384, page_size_padded=221184)
    assert _layer_config_field_diff(first, second) == {
        "block_size": (256, 384),
        "page_size_padded": (None, 221184),
    }


def test_diff_is_empty_when_the_configs_match():
    assert (
        _layer_config_field_diff(_LayerConfig(64, None), _LayerConfig(64, None)) == {}
    )


def test_diff_falls_back_to_the_whole_value_without_dataclass_fields():
    assert _layer_config_field_diff(1, 2) == {"config": (1, 2)}
