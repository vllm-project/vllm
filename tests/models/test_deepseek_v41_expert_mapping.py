# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def test_expert_mapping_index_matches_linear_scan():
    """The indexed lookup returns exactly the entries the linear scan would match."""
    from vllm.models.deepseek_v4.nvidia.model import (
        make_deepseek_v4_expert_params_mapping,
    )
    from vllm.models.deepseek_v41.nvidia.model import _expert_mapping_index

    mapping = make_deepseek_v4_expert_params_mapping(384)
    candidates = _expert_mapping_index(mapping)
    names = [
        f"layers.{layer}.ffn.experts.{expert}.{w}.{kind}"
        for layer in (0, 39)
        for expert in (0, 1, 11, 38, 111, 383)
        for w in ("w1", "w2", "w3")
        for kind in ("weight", "scale")
    ] + ["layers.0.ffn.shared_experts.w1.weight", "layers.0.attn.wq_a.weight"]
    for name in names:
        assert candidates(name) == [m for m in mapping if m[1] in name], name

    fallback = [("experts.w13_", "experts.0.gate_proj.", 0, "w1")]
    assert _expert_mapping_index(fallback)("experts.0.gate_proj.weight") == fallback
