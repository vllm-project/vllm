# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests verifying that MLP layers call maybe_fused_act_quant.

These tests ensure that models with manual fusion support properly route
activation + quantization through the maybe_fused_act_quant function.
"""

from unittest.mock import Mock, patch

import torch


def test_llama_mlp_uses_maybe_fused_act_quant():
    """Verify LlamaMLP.forward() routes through maybe_fused_act_quant."""
    from vllm.model_executor.models.llama import LlamaMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = LlamaMLP.__new__(LlamaMLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.llama.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp.forward(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused


def test_mistral_mlp_uses_maybe_fused_act_quant():
    """Verify MistralMLP.forward() routes through maybe_fused_act_quant."""
    from vllm.model_executor.models.mistral import MistralMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = MistralMLP.__new__(MistralMLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.mistral.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp.forward(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused


def test_qwen2_mlp_uses_maybe_fused_act_quant():
    """Verify Qwen2MLP.forward() routes through maybe_fused_act_quant."""
    from vllm.model_executor.models.qwen2 import Qwen2MLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = Qwen2MLP.__new__(Qwen2MLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.qwen2.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp.forward(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused


def test_qwen3_moe_mlp_uses_maybe_fused_act_quant():
    """Verify Qwen3MoeMLP (shared expert) routes through maybe_fused_act_quant."""
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    down_proj = Mock(side_effect=lambda x: (x, None))

    mlp = Qwen3MoeMLP.__new__(Qwen3MoeMLP)
    torch.nn.Module.__init__(mlp)
    mlp.gate_up_proj = Mock(return_value=(projected, None))
    mlp.down_proj = down_proj
    mlp.act_fn = act_fn
    mlp.expert_gate = None

    with patch(
        "vllm.model_executor.models.qwen3_moe.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp.forward(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, down_proj)
    act_fn.assert_not_called()
    assert result is fused
