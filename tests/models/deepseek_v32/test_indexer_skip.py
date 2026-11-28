# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.amd import model as deepseek_v32_amd_model
from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_nvidia_model


@pytest.mark.parametrize(
    "model_module",
    [deepseek_v32_nvidia_model, deepseek_v32_amd_model],
    ids=["nvidia", "amd"],
)
def test_deepseek_v32_load_weights_skips_unbuilt_indexers(model_module):
    """When index_topk_freq > 1, checkpoint weights for layers that do not

    instantiate an indexer must be skipped rather than raising a KeyError.
    """
    model_cls = model_module.DeepseekV32Model
    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        n_routed_experts=8,
        n_shared_experts=0,
    )
    model.num_redundant_experts = 0
    model.fused_shared_expert_layers = []

    # Layer 0 builds an indexer; layer 1 does not.
    p0 = nn.Parameter(torch.zeros(10))
    p1 = nn.Parameter(torch.zeros(10))
    model.named_parameters = lambda: [
        ("layers.0.self_attn.indexer.k_norm.weight", p0),
        ("layers.1.self_attn.q_proj.weight", p1),
    ]

    weights = [
        ("layers.0.self_attn.indexer.k_norm.weight", torch.ones(10)),
        ("layers.1.self_attn.indexer.k_norm.weight", torch.ones(10)),
    ]

    loaded = model.load_weights(weights)
    assert "layers.0.self_attn.indexer.k_norm.weight" in loaded
    assert "layers.1.self_attn.indexer.k_norm.weight" not in loaded
