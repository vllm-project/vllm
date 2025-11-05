# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def should_split(node: torch.fx.Node, splitting_ops: list[str]) -> bool:
    """Checks if a node should be split for pieceiwse cudagraph."""
    # Match node.target against resolved_ops
    # node.target can be OpOverloadPacket, need to check .default

    if node.op != "call_function":
        return False

    op_overload = node.target
    assert isinstance(op_overload, torch._ops.OpOverload)

    # Example: "aten::add"
    op_overload_packet_name = op_overload.name()

    # Example: "aten::add.default"
    op_overload_name = f"{op_overload_packet_name}.{op_overload._overloadname}"

    return op_overload_packet_name in splitting_ops or op_overload_name in splitting_ops


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
