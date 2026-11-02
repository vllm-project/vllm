# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import LoRAConfig
from vllm.model_executor.layers.fused_moe.layer import _is_moe_lora_target


@pytest.mark.parametrize(
    ("target_modules", "expected"),
    [
        (None, True),
        (["q_proj", "k_proj", "v_proj", "o_proj"], False),
        (["experts"], True),
    ],
)
def test_moe_lora_follows_target_modules(target_modules, expected):
    config = LoRAConfig(max_lora_rank=16, target_modules=target_modules)
    assert _is_moe_lora_target(config, "model.layers.3.mlp.experts") is expected
    assert not _is_moe_lora_target(None, "model.layers.3.mlp.experts")
