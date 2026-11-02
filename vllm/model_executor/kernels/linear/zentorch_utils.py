# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gates zentorch CPU linear dispatch on platform/op availability."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING

import torch
from packaging.version import Version

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

__all__ = [
    "has_zentorch_op",
    "is_zentorch_moe_config_supported",
    "is_zentorch_moe_supported",
    "w4_group_scale_from_e8m0",
    "w4_group_scale_from_e4m3",
    "check_w4_group_shape",
    "w4_group_linear",
    "warn_if_zentorch_predates_e2m1",
    "_ZENTORCH_MOE_ACTIVATIONS",
]

# Shape gates the W4A8 custom kernel enforces. ZenDNN serves a layer it refuses
# from an int8 copy of the weight, so reject it at load to keep weights 4-bit.
_W4_K_OCTET = 8
_W4_MIN_PACK_NR = 32

# The first zentorch whose zentorch_dynamic_qlinear decodes E2M1, and the
# release the vllm[zen] pin moves to. Older builds read the codes as int8.
_ZENTORCH_E2M1_MIN_VERSION = "2.14.0.4"


_ZENTORCH_MOE_ACTIVATIONS = frozenset({"gelu", "gelu_tanh", "silu", "swigluoai"})


def _moe_activation_to_str(activation: object) -> str:
    """Normalize activation to a lowercase string (enum-safe)."""
    if hasattr(activation, "value"):
        return str(activation.value).lower()
    return str(activation).lower()


def has_zentorch_op(op_names: list[str]) -> bool:
    """Return ``True`` when running on Zen CPU with all named ops registered."""
    if not op_names:
        raise ValueError("has_zentorch_op requires at least one op name")
    if not current_platform.is_zen_cpu():
        return False
    ns = getattr(torch.ops, "zentorch", None)
    if ns is None:
        return False
    return all(hasattr(ns, op_name) for op_name in op_names)


def w4_group_scale_from_e8m0(e8m0: torch.Tensor) -> torch.Tensor:
    """MXFP4's E8M0 block exponents as the kernel's {..., G, N} float scales.

    Halved because the kernel decodes E2M1 at twice the true magnitude, which
    is what makes every level an exact integer.
    """
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _upcast_e8m0_to_fp32,
    )

    return (_upcast_e8m0_to_fp32(e8m0) * 0.5).transpose(-1, -2).contiguous()


def w4_group_scale_from_e4m3(
    block_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """NVFP4's E4M3 block scales times the per-tensor global, as {..., G, N}.

    Halved for the same doubled-E2M1 reason as the E8M0 path. Both checkpoint
    formats normalize the global to the dequant multiplier before this point.
    """
    scale = block_scale.to(torch.float32)
    g = global_scale.to(torch.float32).reshape(-1)
    # Per-expert for a 3D MoE scale, a single value for a 2D linear one.
    g = g.view(-1, 1, 1) if scale.ndim == 3 else g.view(1, 1)
    return (scale * g * 0.5).transpose(-1, -2).contiguous()


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


def warn_if_zentorch_predates_e2m1() -> None:
    try:
        installed = importlib.metadata.version("zentorch")
    except importlib.metadata.PackageNotFoundError:
        return
    if Version(installed) < Version(_ZENTORCH_E2M1_MIN_VERSION):
        logger.warning_once(
            "zentorch %s predates E2M1 support in zentorch_dynamic_qlinear "
            "(needs >= %s, the version vllm[zen] installs): MXFP4 and NVFP4 "
            "linear layers will produce wrong output. Reinstall with vllm[zen].",
            installed,
            _ZENTORCH_E2M1_MIN_VERSION,
        )


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
