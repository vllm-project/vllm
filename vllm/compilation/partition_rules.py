# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from torch._library.utils import lookup_op

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)


def _resolve_operator_overload(op_name: str) -> torch._ops.OpOverload:
    """Resolve operator name to torch.ops OpOverload.

    Uses PyTorch's lookup_op utility.

    Args:
        op_name (str): Operator name in PyTorch format "namespace::name.overload"
            Example: "aten::addmm.default"

    Returns:
        torch._ops.OpOverload: The resolved operator overload object
    """
    try:
        return lookup_op(op_name)
    except Exception as exc:
        raise ValueError(f"Failed to resolve operator '{op_name}'") from exc


def _resolve_operators_safely(op_names: list[str]) -> list[torch._ops.OpOverload]:
    """Safely resolve operator names to OpOverload objects.

    Filters out built-in PyTorch operators (aten::*, torch::*) and only resolves
    vLLM custom operators (vllm::*). Skips operators that fail to resolve.

    Built-in PyTorch operators are filtered because they are often decomposed,
    fused, or transformed during Inductor's compilation passes (lowering, fusion,
    decomposition). This makes them unreliable for Inductor partition rules, as
    the scheduler may fail to find corresponding FX nodes after transformations.

    Args:
        op_names: List of operator names in PyTorch format
            (e.g., "vllm::unified_attention")

    Returns:
        List of successfully resolved vLLM custom operator overloads
    """
    resolved = []
    for op_name in op_names:
        # Only register vLLM custom operators for Inductor partitioning
        # Skip aten/torch operators as they may be decomposed/fused
        if not op_name.startswith("vllm::"):
            logger.debug(
                "Skipping non-vLLM operator for Inductor partition: %s", op_name
            )
            continue

        try:
            resolved.append(_resolve_operator_overload(op_name))
        except ValueError:
            # Skip operators that don't exist (e.g., model-specific ops)
            logger.warning(
                "Failed to resolve operator for Inductor partition: %s", op_name
            )
            continue

    return resolved


@contextlib.contextmanager
def inductor_partition_rule_context(op_names: list[str]):
    """Context manager to temporarily register Inductor partition rules.

    Registers custom partition rules for specified operators, forcing the
    Inductor scheduler to partition the graph at these operators. The rules
    are automatically restored to their previous state on exit.

    Note: Only vLLM custom operators (vllm::*) are registered. Built-in
    PyTorch operators (aten::*, torch::*) are filtered out because they may
    be decomposed, fused, or transformed during Inductor compilation, which
    can cause the scheduler to fail when looking up FX nodes.

    Args:
        op_names (list[str]): List of operator names in PyTorch format
            (e.g., ["aten::addmm.default", "vllm::unified_attention"]).
    """
    if not op_names:
        logger.debug("No partition ops provided; skipping rule registration.")
        yield
        return

    from torch._inductor.scheduler import (  # type: ignore
        _custom_should_partition_fns,
        register_should_partition_rule,
    )

    # Safely resolve and filter operators
    overloads = _resolve_operators_safely(op_names)

    if not overloads:
        logger.debug("No valid vLLM operators to register for Inductor partition.")
        yield
        return

    def _always_partition(*_args, **_kwargs):
        return True

    # Save current state before registering
    saved_rules = _custom_should_partition_fns.copy()

    for overload in overloads:
        register_should_partition_rule(
            overload,
            _always_partition,
        )

    logger.debug(
        "Registered inductor partition rules for %d vLLM operators", len(overloads)
    )

    try:
        yield
    finally:
        # Clear and restore previous state
        _custom_should_partition_fns.clear()
        _custom_should_partition_fns.update(saved_rules)
        logger.debug("Restored previous partition rules state.")
