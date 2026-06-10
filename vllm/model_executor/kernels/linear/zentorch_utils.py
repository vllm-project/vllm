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
]


_ZENTORCH_MOE_ACTIVATIONS = frozenset({"gelu", "gelu_tanh", "silu", "swigluoai"})


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
    act = str(activation).lower()
    if act not in _ZENTORCH_MOE_ACTIVATIONS:
        logging.debug(
            "Skipping zentorch fused-MoE: activation %r unsupported (supported: %s).",
            act,
            _ZENTORCH_MOE_ACTIVATIONS,
        )
        return False
    return True
