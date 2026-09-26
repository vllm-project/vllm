# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import nn

from vllm.compilation.caching import lora_wrap_hash_factor
from vllm.lora.layers import BaseLayerWithLoRA


def _toy(wrapped: bool) -> nn.Module:
    root = nn.Module()
    root.linear = nn.Linear(4, 4)
    if wrapped:
        wrapper = BaseLayerWithLoRA()
        wrapper.base_layer = root.linear
        root.linear = wrapper
    return root


def test_wrap_state_changes_factor():
    # Regression for #55383: wrapping moves weights under base_layer, so a
    # wrapped and an unwrapped tree must not share an AOT artifact directory.
    assert lora_wrap_hash_factor(_toy(False)) != lora_wrap_hash_factor(_toy(True))


def test_same_wrap_state_same_factor():
    assert lora_wrap_hash_factor(_toy(True)) == lora_wrap_hash_factor(_toy(True))


def test_unwrapped_factor_is_stable():
    assert lora_wrap_hash_factor(_toy(False)) == lora_wrap_hash_factor(_toy(False))


def test_wrap_path_not_wrap_count():
    first = nn.Module()
    first.a = nn.Module()
    first.a.linear = BaseLayerWithLoRA()
    first.a.linear.base_layer = nn.Linear(4, 4)
    second = nn.Module()
    second.b = nn.Module()
    second.b.linear = BaseLayerWithLoRA()
    second.b.linear.base_layer = nn.Linear(4, 4)
    assert lora_wrap_hash_factor(first) != lora_wrap_hash_factor(second)
