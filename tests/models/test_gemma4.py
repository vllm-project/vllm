# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.gemma4 import _remap_gemma4_expert_weight_name


def test_gemma4_expert_weight_remap_preserves_moe_prefix():
    name = "model.layers.0.moe.experts.0.down_proj.weight"

    assert _remap_gemma4_expert_weight_name(name) == name


def test_gemma4_expert_weight_remap_adds_missing_moe_prefix():
    name = "model.layers.0.experts.0.down_proj.weight"

    assert (
        _remap_gemma4_expert_weight_name(name)
        == "model.layers.0.moe.experts.0.down_proj.weight"
    )
