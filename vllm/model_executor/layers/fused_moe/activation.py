# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE activation function enum and utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEQuantConfig,
    )


class MoEActivation(Enum):
    """Activation functions for MoE layers."""

    # Gated activations (gate * activation(up)) expect input of shape [..., 2*d]
    # and produce output of shape [..., d]
    SILU = "silu"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    RELU2 = "relu2"
    # SWIGLUOAI expects gate/up *interleaved* in w13 ([gate0, up0, gate1, ...]),
    # as in gpt-oss checkpoints. SWIGLUOAI_UNINTERLEAVE has identical math but
    # expects the *packed* layout ([all gates; all ups]), as produced by a
    # MergedColumnParallelLinear gate_up_proj (e.g. MiniMax-M3).
    SWIGLUOAI = "swigluoai"
    SITU = "situ"
    SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
    SWIGLUSTEP = "swiglustep"

    # Non-gated activations (no mul with gate) expect input of shape [..., d]
    # and produce output of shape [..., d].
    # NOTE: Non-gated activations require the "_no_mul" suffix to be present.
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    GELU_TANH_NO_MUL = "gelu_tanh_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        """Returns True if activation expects gate*activation(up) pattern.

        Gated activations expect input tensor with 2x the output size,
        where the first half is the gate and second half is the up projection.
        """
        return not self.value.endswith("_no_mul")

    @property
    def custom_op_name(self) -> str:
        """Maps to the CustomOp name of activations
        in vllm/model_executor/layers/activation.py."""
        return _CUSTOM_OP_NAMES[self]

    def without_mul(self) -> "MoEActivation":
        """Get the non-gated variant of this activation.

        For activations that have a _no_mul variant, returns that variant.
        For activations without a _no_mul variant (or already _no_mul),
        returns self.
        """
        return _WITHOUT_MUL.get(self, self)

    @classmethod
    def from_str(cls, s: str) -> "MoEActivation":
        """Parse from string for backward compatibility."""
        s = _STR_ALIASES.get(s, s)
        for member in cls:
            if member.value == s:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown MoE activation: {s!r}. Valid activations: {valid}")


# Module-level lookup tables used by MoEActivation functions.
_STR_ALIASES: dict[str, str] = {
    "gelu_pytorch_tanh": "gelu_tanh",
}

_CUSTOM_OP_NAMES: dict[MoEActivation, str] = {
    MoEActivation.SILU: "silu_and_mul",
    MoEActivation.GELU: "gelu_and_mul",
    MoEActivation.GELU_TANH: "gelu_tanh_and_mul",
    MoEActivation.SITU: "situ_and_mul",
    MoEActivation.SWIGLUOAI: "swigluoai_and_mul",
    MoEActivation.SWIGLUOAI_UNINTERLEAVE: "silu_and_mul_with_clamp",
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
    """Get the non-gated variant of an activation function.

    Args:
        activation: The activation function name (e.g., "silu", "gelu")

    Returns:
        The non-gated activation name (e.g., "silu_no_mul", "gelu_no_mul")
    """
    return MoEActivation.from_str(activation).without_mul().value


_APPLY_MOE_ACTIVATIONS = frozenset(
    {
        MoEActivation.SILU,
        MoEActivation.GELU,
        MoEActivation.GELU_TANH,
        MoEActivation.SITU,
        MoEActivation.SWIGLUOAI,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        MoEActivation.SWIGLUSTEP,
        MoEActivation.SILU_NO_MUL,
        MoEActivation.GELU_NO_MUL,
        MoEActivation.GELU_TANH_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
    }
)


_MASKED_MOE_ACTIVATION_NAMES: dict[MoEActivation, str] = {
    MoEActivation.SILU: "silu",
    MoEActivation.GELU: "gelu",
    MoEActivation.GELU_TANH: "gelu_tanh",
    MoEActivation.SITU: "situ",
    MoEActivation.SWIGLUOAI: "swigluoai",
    MoEActivation.SWIGLUOAI_UNINTERLEAVE: "swigluoai_uninterleave",
    MoEActivation.SWIGLUSTEP: "swiglustep",
    MoEActivation.SILU_NO_MUL: "silu_no_mul",
    MoEActivation.GELU_NO_MUL: "gelu_no_mul",
    MoEActivation.GELU_TANH_NO_MUL: "gelu_tanh_no_mul",
    MoEActivation.RELU2_NO_MUL: "relu2_no_mul",
}


def apply_moe_activation_supported(activation: MoEActivation) -> bool:
    """Whether ``apply_moe_activation`` supports an activation."""
    return activation in _APPLY_MOE_ACTIVATIONS


def apply_moe_activation_masked_supported(activation: MoEActivation) -> bool:
    """Whether the masked ``apply_moe_activation`` path supports an activation."""
    return activation in _MASKED_MOE_ACTIVATION_NAMES


@dataclass(frozen=True)
class ApplyMoEActivationConfig:
    """Configuration forwarded to ``apply_moe_activation``."""

    clamp_limit: float | None = None
    alpha: float = 1.0
    beta: float = 0.0
    activation_situ_beta: float | None = None
    activation_situ_linear_beta: float | None = None

    @classmethod
    def from_configs(
        cls,
        moe_config: "FusedMoEConfig",
        quant_config: "FusedMoEQuantConfig",
    ) -> "ApplyMoEActivationConfig":
        """Build from the model and quantization configurations."""
        clamp_limit = quant_config.gemm1_clamp_limit
        if clamp_limit is None:
            clamp_limit = moe_config.swiglu_limit
        alpha = quant_config.gemm1_alpha
        if alpha is None:
            alpha = moe_config.swiglu_alpha
        beta = quant_config.gemm1_beta
        if beta is None:
            beta = moe_config.swiglu_beta
        return cls(
            clamp_limit=clamp_limit,
            alpha=1.0 if alpha is None else alpha,
            beta=0.0 if beta is None else beta,
            activation_situ_beta=moe_config.activation_situ_beta,
            activation_situ_linear_beta=moe_config.activation_situ_linear_beta,
        )


_DEFAULT_APPLY_MOE_ACTIVATION_CONFIG = ApplyMoEActivationConfig()


def _validate_moe_activation_shapes(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    expected_dim: int,
) -> None:
    assert input.dim() == expected_dim, f"Input must be {expected_dim}D"
    assert output.dim() == expected_dim, f"Output must be {expected_dim}D"
    assert input.shape[:-1] == output.shape[:-1], (
        f"Input/output leading shapes must match: {input.shape} vs {output.shape}"
    )
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


def _apply_moe_activation_masked(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    valid_token_counts: torch.Tensor,
    config: ApplyMoEActivationConfig,
) -> torch.Tensor:
    masked_activation = _MASKED_MOE_ACTIVATION_NAMES.get(activation)
    if masked_activation is None:
        raise NotImplementedError(
            f"Masked MoE activation is not implemented for {activation.value}"
        )

    assert input.dim() in (2, 3), "Masked input must be 2D or 3D"
    _validate_moe_activation_shapes(activation, output, input, expected_dim=input.dim())
    assert input.dtype == output.dtype, "Input and output dtypes must match"
    assert input.device == output.device, "Input and output devices must match"
    assert input.is_contiguous(), "Input must be contiguous"
    assert output.is_contiguous(), "Output must be contiguous"
    assert valid_token_counts.dtype == torch.int32, (
        "valid_token_counts must use torch.int32"
    )
    assert valid_token_counts.dim() == 1, "valid_token_counts must be 1D"
    assert valid_token_counts.device == input.device, (
        "valid_token_counts must be on the input device"
    )
    assert valid_token_counts.is_contiguous(), "valid_token_counts must be contiguous"
    expected_counts = input.size(0) if input.dim() == 3 else 1
    assert valid_token_counts.size(0) == expected_counts, (
        f"valid_token_counts must have {expected_counts} element(s) for "
        f"{input.dim()}D input"
    )

    masked_clamp_limit = 0.0 if config.clamp_limit is None else config.clamp_limit
    masked_alpha = config.alpha
    if activation == MoEActivation.SILU and config.clamp_limit is not None:
        masked_activation = "silu_with_clamp"
    elif activation == MoEActivation.SITU:
        assert config.activation_situ_beta is not None, (
            "SITU requires activation_situ_beta from FusedMoEConfig"
        )
    elif activation == MoEActivation.SWIGLUOAI:
        masked_clamp_limit = 7.0
        masked_alpha = 1.702
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        assert config.clamp_limit is not None, (
            "SWIGLUOAI_UNINTERLEAVE requires clamp_limit"
        )
    elif activation == MoEActivation.SWIGLUSTEP:
        masked_clamp_limit = 7.0

    torch.ops._C.masked_moe_activation(
        output,
        input,
        valid_token_counts,
        masked_activation,
        masked_clamp_limit,
        masked_alpha,
        config.beta,
        1.0 if config.activation_situ_beta is None else config.activation_situ_beta,
        -1.0
        if config.activation_situ_linear_beta is None
        else config.activation_situ_linear_beta,
    )
    return output


def silu_and_mul_with_clamp(
    output: torch.Tensor,
    input: torch.Tensor,
    clamp_limit: float,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> None:
    if topk_ids is not None and expert_map is not None:
        from vllm.model_executor.layers.fused_moe.utils import swiglu_limit_func

        swiglu_limit_func(output, input, clamp_limit, topk_ids, expert_map)
    else:
        # Fused silu(clamp(gate)) * clamp(up); equivalent to swiglu_limit_func.
        torch.ops._C.silu_and_mul_with_clamp(output, input, clamp_limit, 1.0, 0.0)


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    activation_config: ApplyMoEActivationConfig | None = None,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
    valid_token_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply MoE activation function.

    The configuration drives specialized activation behavior. Routing tensors
    and valid token counts remain per-call inputs because they depend on the
    current token assignment. A single count masks a flat ``[T, D]`` buffer;
    one count per expert masks each prefix in a padded ``[E, T, D]`` buffer.
    """
    config = (
        _DEFAULT_APPLY_MOE_ACTIVATION_CONFIG
        if activation_config is None
        else activation_config
    )

    if valid_token_counts is not None:
        return _apply_moe_activation_masked(
            activation,
            output,
            input,
            valid_token_counts,
            config,
        )

    _validate_moe_activation_shapes(activation, output, input, expected_dim=2)

    # Activations with gated multiplication (gate × activation(up))
    if activation == MoEActivation.SILU:
        if config.clamp_limit is not None:
            silu_and_mul_with_clamp(
                output, input, config.clamp_limit, topk_ids, expert_map
            )
        else:
            torch.ops._C.silu_and_mul(output, input)
    elif activation == MoEActivation.GELU:
        torch.ops._C.gelu_and_mul(output, input)
    elif activation == MoEActivation.GELU_TANH:
        torch.ops._C.gelu_tanh_and_mul(output, input)
    elif activation == MoEActivation.SITU:
        # Fused CUDA kernel: writes straight to `output`, no fp32 temporaries.
        # (The pure-torch fallback below upcast both halves to fp32 and
        # allocated ~8 temporaries per call, blowing up MoE memory.)
        # Both betas come from FusedMoEConfig; a missing beta means the caller
        # bypassed the config plumbing, so fail rather than silently use 1.0.
        # linear_beta is genuinely optional: <= 0 signals "unset" to the kernel
        # (up passed through), matching SituAndMul(linear_beta=None).
        assert config.activation_situ_beta is not None, (
            "SITU requires activation_situ_beta from FusedMoEConfig"
        )
        torch.ops._C.situ_and_mul(
            output,
            input,
            config.activation_situ_beta,
            -1.0
            if config.activation_situ_linear_beta is None
            else config.activation_situ_linear_beta,
        )
    elif activation == MoEActivation.SWIGLUOAI:
        torch.ops._C.swigluoai_and_mul(output, input)
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        # SwiGLU-OAI on packed w13 (gate = first half, up = second half).
        assert config.clamp_limit is not None, (
            "SWIGLUOAI_UNINTERLEAVE requires clamp_limit"
        )
        torch.ops._C.silu_and_mul_with_clamp(
            output, input, config.clamp_limit, config.alpha, config.beta
        )
    elif activation == MoEActivation.SWIGLUSTEP:
        from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

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
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output
