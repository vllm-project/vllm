# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from collections.abc import Generator

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def should_split(node: torch.fx.Node, splitting_ops: list[str]) -> bool:
    """
    Check if a node should be split for dynamo graph partition.
    It operates on dynamo graph, so the node.target can be anything.
    We need to check and split only on OpOverload and OpOverloadPacket.
    """

    if node.op != "call_function":
        return False

    target = node.target

    if isinstance(target, torch._ops.OpOverloadPacket):
        # Example: "aten::add"
        return target._qualified_op_name in splitting_ops

    if isinstance(target, torch._ops.OpOverload):
        # Example: "aten::add"
        packet_name = target.name()

        # Example: "aten::add.default"
        op_overload_name = f"{packet_name}.{target._overloadname}"
        return op_overload_name in splitting_ops or packet_name in splitting_ops

    return False


@contextlib.contextmanager
def inductor_partition_rule_context(
    splitting_ops: list[str] | None,
) -> Generator[None, None, None]:
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

    # Guarded for older torch builds which don't have this config knob.
    saved_cudagraph_unsafe_unbacked_ops: list[str] | None = None
    if hasattr(torch._inductor.config, "cudagraph_unsafe_unbacked_ops"):
        saved_cudagraph_unsafe_unbacked_ops = list(
            torch._inductor.config.cudagraph_unsafe_unbacked_ops  # type: ignore[attr-defined]
        )
        # Only mark known producers of data-dependent SymInts here. 
        known_unsafe_unbacked_ops = {"vllm::mla_split_batch"}
        torch._inductor.config.cudagraph_unsafe_unbacked_ops = [  # type: ignore[attr-defined]
            op for op in splitting_ops if op in known_unsafe_unbacked_ops
        ]

    logger.debug(
        "Registered inductor partition rules for %d operators", len(splitting_ops)
    )

    try:
        yield
    finally:
        # Clear and restore previous state
        torch._inductor.config.custom_should_partition_ops = saved_splitting_ops
        if saved_cudagraph_unsafe_unbacked_ops is not None:
            torch._inductor.config.cudagraph_unsafe_unbacked_ops = (  # type: ignore[attr-defined]
                saved_cudagraph_unsafe_unbacked_ops
            )
        logger.debug("Restored previous partition rules state.")
