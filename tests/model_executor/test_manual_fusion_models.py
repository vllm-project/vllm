# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests verifying that MLP layers call maybe_fused_act_quant.

These tests ensure that models with manual fusion support properly route
activation + quantization through the maybe_fused_act_quant function.
"""

import importlib
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.mark.parametrize(
    "module_path,class_name,extra_attrs",
    [
        ("vllm.model_executor.models.llama", "LlamaMLP", {}),
        ("vllm.model_executor.models.mistral", "MistralMLP", {}),
        ("vllm.model_executor.models.qwen2", "Qwen2MLP", {}),
        ("vllm.model_executor.models.qwen3_moe", "Qwen3MoeMLP", {"expert_gate": None}),
        ("vllm.model_executor.models.deepseek_v2", "DeepseekV2MLP", {}),
    ],
    ids=["llama", "mistral", "qwen2", "qwen3_moe", "deepseek_v2"],
)
def test_mlp_uses_maybe_fused_act_quant(module_path, class_name, extra_attrs):
    """Verify MLP.forward() routes through maybe_fused_act_quant."""
    module = importlib.import_module(module_path)
    mlp_class = getattr(module, class_name)

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = mlp_class.__new__(mlp_class)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    # Set any extra attributes (e.g., expert_gate for Qwen3MoeMLP)
    for attr, value in extra_attrs.items():
        setattr(mlp, attr, value)

    with patch(
        f"{module_path}.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp.forward(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused
