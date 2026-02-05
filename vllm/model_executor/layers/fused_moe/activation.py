# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE activation function enum and utilities."""

from enum import Enum

import torch
import torch.nn.functional as F


class MoEActivation(Enum):
    """Activation functions for MoE layers."""

    # Gated activations (gate * activation(up)) expect input of shape [..., 2*d]
    # and produce output of shape [..., d]
    SILU = "silu"
    GELU = "gelu"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"

    # Non-gated activations (no mul with gate) expect input of shape [..., d]
    # and produce output of shape [..., d].
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        """Returns True if activation expects gate*activation(up) pattern.

        Gated activations expect input tensor with 2x the output size,
        where the first half is the gate and second half is the up projection.
        """
        return not self.value.endswith("_no_mul")

    @property
    def op_name(self) -> str:
        """Return the base activation function name without the _no_mul suffix."""
        return self.value.replace("_no_mul", "")

    def without_mul(self) -> "MoEActivation":
        """Get the non-gated variant of this activation.

        For activations that have a _no_mul variant, returns that variant.
        For activations without a _no_mul variant (or already _no_mul),
        returns self.
        """
        mapping = {
            MoEActivation.SILU: MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU: MoEActivation.GELU_NO_MUL,
        }
        return mapping.get(self, self)

    @classmethod
    def from_str(cls, s: str) -> "MoEActivation":
        """Parse from string for backward compatibility."""
        for member in cls:
            if member.value == s:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown MoE activation: {s!r}. Valid activations: {valid}")


def activation_without_mul(activation: str) -> str:
    """Get the non-gated variant of an activation function.

    Args:
        activation: The activation function name (e.g., "silu", "gelu")

    Returns:
        The non-gated activation name (e.g., "silu_no_mul", "gelu_no_mul")
    """
    return MoEActivation.from_str(activation).without_mul().value


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """Apply MoE activation function."""
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

    # Activations with gated multiplication (gate × activation(up))
    if activation == MoEActivation.SILU:
        torch.ops._C.silu_and_mul(output, input)
    elif activation == MoEActivation.GELU:
        torch.ops._C.gelu_and_mul(output, input)
    elif activation == MoEActivation.SWIGLUOAI:
        torch.ops._C.swigluoai_and_mul(output, input)
    elif activation == MoEActivation.SWIGLUSTEP:
        from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

        swiglustep_and_mul_triton(output, input)

    # Activations without gated multiplication
    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output
