# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.attention import Attention, MMEncoderAttention


@pytest.mark.parametrize("attention_cls", [Attention, MMEncoderAttention])
def test_num_heads_not_divisble_by_num_kv_heads(attention_cls: type) -> None:
    head_size = 64
    scale = float(1.0 / (head_size**0.5))
    num_heads = 16
    num_kv_heads = 5
    with pytest.raises(AssertionError):
        _ = attention_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
        )
