# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.layer_utils import extract_layer_index


@pytest.mark.parametrize(
    ("layer_name", "expected_index"),
    [
        ("encoder.layers.0", 0),
        ("encoder.layers.1.self_attn", 1),
        ("2.self_attn", 2),
    ],
)
def test_extract_layer_index(layer_name: str, expected_index: int):
    assert extract_layer_index(layer_name) == expected_index


def test_extract_layer_index_with_multiple_attention_modules():
    assert extract_layer_index("model.layers.0.self_attn.1", 2) == 1
    assert extract_layer_index("model.layers.1.self_attn.0", 2) == 2
