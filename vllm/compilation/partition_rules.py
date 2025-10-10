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


def resolve_defined_ops(op_names: list[str]) -> list[torch._ops.OpOverload]:
    """Resolve operator names to OpOverload objects.

    Skips operators that fail to resolve (e.g., operators not registered or
    model-specific operators not present in the current model).

    Note: Users should inspect the operator graph before lowering and ensure
    the specified operators are present in the final graph. Built-in PyTorch
    operators (aten::*, torch::*) may be decomposed, fused, or transformed
    during Inductor's compilation passes, so use them with caution.

    Args:
        op_names: List of operator names in PyTorch format
            (e.g., "vllm::unified_attention")

    Returns:
        List of successfully resolved operator overloads
    """
    resolved = []
    for op_name in op_names:
        try:
            resolved.append(lookup_op(op_name))
        except Exception:
            # Skip operators that don't exist (e.g., model-specific ops)
            logger.warning(
                "Failed to resolve operator for Inductor partition: %s", op_name
            )
            continue

    return resolved


@contextlib.contextmanager
def inductor_partition_rule_context(overloads: list[torch._ops.OpOverload]):
    """Context manager to temporarily register Inductor partition rules.

    Registers custom partition rules for specified operators, forcing the
    Inductor scheduler to partition the graph at these operators. The rules
    are automatically restored to their previous state on exit.

    Note: Callers should use resolve_defined_ops() to convert operator names
    to OpOverload objects before calling this function.

    Args:
        overloads: List of resolved operator overload objects.
    """
    if not overloads:
        logger.debug("No partition ops provided; skipping rule registration.")
        yield
        return

    from torch._inductor.scheduler import (  # type: ignore
        _custom_should_partition_fns,
        register_should_partition_rule,
    )

    def _always_partition(*_args, **_kwargs):
        return True

    # Save current state before registering
    saved_rules = _custom_should_partition_fns.copy()

    for overload in overloads:
        register_should_partition_rule(
            overload,
            _always_partition,
        )

    logger.debug("Registered inductor partition rules for %d operators", len(overloads))

    try:
        yield
    finally:
        # Clear and restore previous state
        _custom_should_partition_fns.clear()
        _custom_should_partition_fns.update(saved_rules)
        logger.debug("Restored previous partition rules state.")
