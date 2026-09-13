# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 DSpark draft weight-mapping tests."""

from vllm.models.kimi_k3.nvidia.dspark_mla import _build_weights_mapper


def test_mapper_drops_embed_for_target_aliasing():
    mapper = _build_weights_mapper(drop_embed=True)
    assert mapper.apply_list(["embed_tokens.weight"]) == []
    assert mapper.apply_list(["lm_head.weight"]) == []
    # Draft-owned weights still map into the model namespace.
    assert mapper.apply_list(["layers.0.mlp.gate_proj.weight"]) == [
        "model.layers.0.mlp.gate_up_proj.weight"
    ]


def test_mapper_keeps_embed_under_pp():
    # Under PP the drafter cannot alias the target's first-stage table, so the
    # checkpoint's own embed_tokens.weight must flow through.
    mapper = _build_weights_mapper(drop_embed=False)
    assert mapper.apply_list(["embed_tokens.weight"]) == ["model.embed_tokens.weight"]
    assert mapper.apply_list(["lm_head.weight"]) == []
