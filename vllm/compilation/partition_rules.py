# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import logging

import torch
from torch._library.utils import lookup_op

from vllm.logger import init_logger

logger = init_logger(__name__)


def resolve_defined_ops(op_names: list[str]) -> list["torch._ops.OpOverload"]:
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
            # Do not warn for attention ops, warn for others
            # (most likely manually specified)
            from vllm.config import CompilationConfig

            logger.log(
                logging.DEBUG
                if op_name in CompilationConfig._attention_ops
                else logging.WARNING,
                "Failed to resolve operator for CUDAGraph partition: %s",
                op_name,
            )
            continue

    return resolved


@contextlib.contextmanager
def inductor_partition_rule_context(splitting_ops: list[str]):
    """Context manager to temporarily register Inductor partition rules.

    Registers custom partition rules for specified operators, forcing the
    Inductor scheduler to partition the graph at these operators. The rules
    are automatically restored to their previous state on exit.

    Args:
        splitting_ops: List of operator names to partition on.
    """
    if not splitting_ops:
        logger.debug("No partition ops provided; skipping rule registration.")
        yield
        return

    # Save current state before registering

    saved_splitting_ops: list[str] = list(
        torch._inductor.config.custom_should_partition_ops
    )
    torch._inductor.config.custom_should_partition_ops = splitting_ops

    logger.debug(
        "Registered inductor partition rules for %d operators", len(splitting_ops)
    )

    try:
        yield
    finally:
        # Clear and restore previous state
        torch._inductor.config.custom_should_partition_ops = saved_splitting_ops
        logger.debug("Restored previous partition rules state.")
