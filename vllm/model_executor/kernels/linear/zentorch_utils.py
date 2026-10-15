# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gates zentorch CPU linear dispatch on platform/op availability."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

__all__ = [
    "has_zentorch_op",
    "is_zentorch_moe_config_supported",
    "is_zentorch_moe_supported",
    "w4_group_scale_from_e8m0",
    "check_w4_group_shape",
    "w4_group_linear",
    "is_zentorch_e2m1_supported",
    "_ZENTORCH_MOE_ACTIVATIONS",
]

# Shape gates the W4A8 custom kernel enforces. ZenDNN serves a layer it refuses
# from an int8 copy of the weight, so reject it at load to keep weights 4-bit.
_W4_K_OCTET = 8
_W4_MIN_PACK_NR = 32

_ZENTORCH_MOE_ACTIVATIONS = frozenset({"gelu", "gelu_tanh", "silu", "swigluoai"})


def _moe_activation_to_str(activation: object) -> str:
    """Normalize activation to a lowercase string (enum-safe)."""
    if hasattr(activation, "value"):
        return str(activation.value).lower()
    return str(activation).lower()


def has_zentorch_op(op_names: list[str]) -> bool:
    """Return `True` when running on Zen CPU with all named ops registered."""
    if not op_names:
        raise ValueError("has_zentorch_op requires at least one op name")
    if not current_platform.is_zen_cpu():
        return False
    ns = getattr(torch.ops, "zentorch", None)
    if ns is None:
        return False
    return all(hasattr(ns, op_name) for op_name in op_names)


def w4_group_scale_from_e8m0(e8m0: torch.Tensor) -> torch.Tensor:
    """MXFP4's E8M0 block exponents laid out {..., G, N}, bytes unchanged: the
    kernel takes them as float8_e8m0fnu and applies them itself."""
    return e8m0.view(torch.float8_e8m0fnu).transpose(-1, -2).contiguous()


def check_w4_group_shape(n: int, k: int, groups: int) -> int:
    """Validate a four-bit linear against the kernel's gates, returning group_size."""
    if k % groups:
        raise ValueError(
            f"zentorch four-bit linear: K ({k}) must be divisible by the "
            f"group count ({groups})"
        )
    group_size = k // groups
    if group_size % _W4_K_OCTET or n % _W4_MIN_PACK_NR:
        raise ValueError(
            f"zentorch four-bit linear: group_size ({group_size}) must be a "
            f"multiple of {_W4_K_OCTET} and N ({n}) a multiple of "
            f"{_W4_MIN_PACK_NR}"
        )
    return group_size


def is_zentorch_e2m1_supported() -> tuple[bool, str | None]:
    """Whether the installed zentorch decodes E2M1 weights, and why not.

    E2M1 is newer than the ops that take it, so their presence does not imply
    it and only the version distinguishes the two.
    """
    # TODO: compare against the first zentorch release that carries E2M1 once
    # it is cut; no released version can be named yet.
    return True, None


def w4_group_linear(
    weight: torch.Tensor,
    group_scale: torch.Tensor,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run an E2M1 linear through zentorch's dynamic-quant (DA8W4) linear."""
    # The kernel takes only bf16, and the layer config carries no activation
    # dtype to gate on, so an fp16 or f32 model is served through a cast.
    if bias is not None:
        bias = bias.to(torch.bfloat16).contiguous()
    out = torch.ops.zentorch.zentorch_dynamic_qlinear.default(
        x.to(torch.bfloat16), weight, group_scale, bias
    )
    return out.to(x.dtype)


def is_zentorch_moe_config_supported(moe_config: FusedMoEConfig) -> bool:
    if not has_zentorch_op(["zentorch_fused_moe"]):
        return False
    if not moe_config.is_act_and_mul:
        return False
    act = _moe_activation_to_str(moe_config.activation)
    return act in _ZENTORCH_MOE_ACTIVATIONS


def is_zentorch_moe_supported(layer: torch.nn.Module) -> bool:
    if not has_zentorch_op(["zentorch_fused_moe"]):
        logging.debug(
            "Skipping zentorch fused-MoE: not a Zen CPU or "
            "zentorch not loaded; using default MoE."
        )
        return False
    moe_config = getattr(layer, "moe_config", None)
    if moe_config is not None and not moe_config.is_act_and_mul:
        logging.debug(
            "Skipping zentorch fused-MoE: layer is not a gated "
            "(act-and-mul, e.g. SwiGLU) MLP, the only structure supported."
        )
        return False
    activation = getattr(layer, "activation", None)
    if activation is None:
        logging.debug(
            "Skipping zentorch fused-MoE: layer has no 'activation' "
            "attribute, so the activation can't be verified."
        )
        return False
    act = str(getattr(activation, "value", activation)).lower()
    if act not in _ZENTORCH_MOE_ACTIVATIONS:
        logging.debug(
            "Skipping zentorch fused-MoE: activation %r unsupported (supported: %s).",
            act,
            _ZENTORCH_MOE_ACTIVATIONS,
        )
        return False
    return True
