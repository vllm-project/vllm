# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE activation function enum and utilities."""

from enum import Enum

import torch
import torch.nn.functional as F


class MoEActivation(Enum):
    """Activation functions for MoE layers."""

    # Gated activations (gate * activation(up)) expect input of shape
    # [..., 2*d] and produce output of shape [..., d].
    SILU = "silu"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    RELU2 = "relu2"
    # Packed-w13 SwiGLU step (Triton).
    SWIGLUSTEP = "swiglustep"

    # Non-gated activations: input and output have the same shape [..., d].
    # Names must end with the "_no_mul" suffix.
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    GELU_TANH_NO_MUL = "gelu_tanh_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.value.endswith("_no_mul")

    @property
    def custom_op_name(self) -> str:
        return _CUSTOM_OP_NAMES[self]

    def without_mul(self) -> "MoEActivation":
        return _WITHOUT_MUL.get(self, self)

    @classmethod
    def from_str(cls, s: str) -> "MoEActivation":
        s = _STR_ALIASES.get(s, s)
        for member in cls:
            if member.value == s:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown MoE activation: {s!r}. Valid activations: {valid}")


_STR_ALIASES: dict[str, str] = {
    "gelu_pytorch_tanh": "gelu_tanh",
}

_CUSTOM_OP_NAMES: dict[MoEActivation, str] = {
    MoEActivation.SILU: "silu_and_mul",
    MoEActivation.GELU: "gelu_and_mul",
    MoEActivation.GELU_TANH: "gelu_tanh_and_mul",
    MoEActivation.SWIGLUSTEP: "swiglustep_and_mul",
    MoEActivation.RELU2: "relu2",
    MoEActivation.SILU_NO_MUL: "silu_and_mul",
    MoEActivation.GELU_NO_MUL: "gelu_and_mul",
    MoEActivation.GELU_TANH_NO_MUL: "gelu_tanh_and_mul",
    MoEActivation.RELU2_NO_MUL: "relu2",
}

_WITHOUT_MUL: dict[MoEActivation, MoEActivation] = {
    MoEActivation.SILU: MoEActivation.SILU_NO_MUL,
    MoEActivation.GELU: MoEActivation.GELU_NO_MUL,
    MoEActivation.GELU_TANH: MoEActivation.GELU_TANH_NO_MUL,
    MoEActivation.RELU2: MoEActivation.RELU2_NO_MUL,
}


def activation_without_mul(activation: str) -> str:
    """Get the non-gated variant of an activation function name."""
    return MoEActivation.from_str(activation).without_mul().value


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    clamp_limit: float | None = None,
) -> torch.Tensor:
    """Apply an MoE activation function. ``clamp_limit`` selects the
    clamped SiLU fused op (only honored when ``activation == SILU``)."""
    assert input.dim() == 2, "Input must be 2D"
    assert output.dim() == 2, "Output must be 2D"
    if activation.is_gated:
        assert output.size(-1) * 2 == input.size(-1), (
            f"{activation.value} expects 2x ratio: "
            f"{output.size(-1) * 2} vs {input.size(-1)}"
        )
    else:
        assert output.size(-1) == input.size(-1), (
            f"{activation.value} expects equal sizes: "
            f"{output.size(-1)} vs {input.size(-1)}"
        )

    # Activations with gated multiplication (gate × activation(up)). The gate
    # is the first half of the packed [gate || up] input; the up projection is
    # the second half.
    if activation == MoEActivation.SILU:
        if clamp_limit is not None:
            # Clamped SwiGLU has non-trivial alpha/beta/pre-clamp semantics;
            # keep the fused kernel for it.
            torch.ops._C.silu_and_mul_with_clamp(output, input, clamp_limit, 1.0, 0.0)
        else:
            gate, up = input.chunk(2, dim=-1)
            torch.mul(F.silu(gate), up, out=output)
    elif activation == MoEActivation.GELU:
        gate, up = input.chunk(2, dim=-1)
        torch.mul(F.gelu(gate), up, out=output)
    elif activation == MoEActivation.GELU_TANH:
        gate, up = input.chunk(2, dim=-1)
        torch.mul(F.gelu(gate, approximate="tanh"), up, out=output)
    elif activation == MoEActivation.SWIGLUSTEP:
        from vllm.model_executor.hw_agnostic.layers.activation import (
            swiglustep_and_mul_triton,
        )

        swiglustep_and_mul_triton(output, input)

    # Activations without gated multiplication
    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.GELU_TANH_NO_MUL:
        output.copy_(F.gelu(input, approximate="tanh"))
    elif activation == MoEActivation.RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported MoE activation: {activation}")

    return output
