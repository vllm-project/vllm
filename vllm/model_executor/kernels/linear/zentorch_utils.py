# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gates zentorch CPU linear dispatch on platform/op availability."""

from __future__ import annotations

import logging

import torch

from vllm.platforms import current_platform

__all__ = [
    "has_zentorch_op",
    "is_zentorch_moe_supported",
    "_ZENTORCH_MOE_ACTIVATIONS",
    "_moe_activation_to_str",
]


_ZENTORCH_MOE_ACTIVATIONS = frozenset({"gelu", "gelu_tanh", "silu", "swigluoai"})


def _moe_activation_to_str(activation: object) -> str:
    """Normalize activation to a lowercase string (enum-safe)."""
    try:
        # Local import to avoid import cycles and keep this module lightweight.
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    except Exception:  # pragma: no cover - defensive for optional imports
        MoEActivation = None  # type: ignore[assignment,misc]

    raw: object
    if MoEActivation is not None and isinstance(activation, MoEActivation):
        raw = activation.value
    elif isinstance(activation, str):
        raw = activation
    elif hasattr(activation, "value"):
        raw = activation.value  # type: ignore[attr-defined]
    else:
        raw = activation
    return str(raw).lower()


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


def is_zentorch_moe_supported(layer: torch.nn.Module) -> bool:
    if not has_zentorch_op(["zentorch_fused_moe"]):
        logging.info("torch.ops.zentorch.zentorch_fused_moe is not registered")
        return False
    moe_config = getattr(layer, "moe_config", None)
    if moe_config is not None and not moe_config.is_act_and_mul:
        logging.info("is_act_and_mul=False is not supported")
        return False
    activation = getattr(layer, "activation", None)
    if activation is None:
        logging.info("layer has no activation attribute")
        return False
    act = _moe_activation_to_str(activation)
    if act not in _ZENTORCH_MOE_ACTIVATIONS:
        logging.info(
            "activation %r is not supported (supported: %s)",
            act,
            _ZENTORCH_MOE_ACTIVATIONS,
        )
        return False
    return True
