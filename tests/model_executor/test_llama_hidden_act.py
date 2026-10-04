# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LlamaMLP resolving `hidden_act` through the activation registry.

The reference computation mirrors how `transformers` applies the configured
activation in `LlamaMLP`: `down_proj(ACT2FN[hidden_act](gate_proj(x)) * up_proj(x))`.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.llama import LlamaMLP

HIDDEN_SIZE = 16
INTERMEDIATE_SIZE = 32

# Reference definitions of the supported gate activations, matching the
# semantics of `transformers.activations.ACT2FN` for the same config names.
REFERENCE_ACT_FNS = {
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "swish": F.silu,
}


def _make_mlp(hidden_act: str, seed: int = 0) -> LlamaMLP:
    mlp = LlamaMLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act=hidden_act,
    )
    # Give the (otherwise uninitialized) projection weights deterministic,
    # finite values so outputs can be compared numerically.
    generator = torch.Generator().manual_seed(seed)
    for param in mlp.parameters():
        param.data.normal_(mean=0.0, std=0.1, generator=generator)
    return mlp


@pytest.mark.parametrize("activation_name", sorted(REFERENCE_ACT_FNS))
def test_llama_mlp_matches_reference_hidden_act(
    activation_name: str,
    default_vllm_config,
    dist_init,
) -> None:
    """LlamaMLP must apply the activation selected by `hidden_act`, i.e.
    compute `down_proj(act(gate) * up)` like the transformers implementation."""
    mlp = _make_mlp(activation_name)
    x = torch.randn(3, HIDDEN_SIZE, generator=torch.Generator().manual_seed(1))

    gate_up = x @ mlp.gate_up_proj.weight.T
    gate, up = gate_up.chunk(2, dim=-1)
    expected = (REFERENCE_ACT_FNS[activation_name](gate) * up) @ mlp.down_proj.weight.T

    torch.testing.assert_close(mlp(x), expected)


def test_llama_mlp_rejects_unknown_hidden_act(
    default_vllm_config,
    dist_init,
) -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        _make_mlp("not_a_real_activation")


def test_llama_mlp_rejects_incompatible_operand_layout(
    default_vllm_config,
    dist_init,
) -> None:
    """`swigluoai` is in the activation-and-mul registry, but SwigluOAIAndMul
    reads the gate and up operands interleaved rather than as the concatenated
    halves produced by this MLP's fused gate_up_proj. It must be rejected
    rather than silently computing the wrong result."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        _make_mlp("swigluoai")
