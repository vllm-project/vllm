# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen2_5_vl import _pad_to_tp


def test_qwen3_vision_tp3_padding_covers_heads_and_mlp():
    assert _pad_to_tp(16, 3) == 18
    assert _pad_to_tp(4304, 3) == 4305
    assert _pad_to_tp(4608, 3) == 4608
