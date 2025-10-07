# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib

from torch._library.utils import lookup_op

from vllm.logger import init_logger

logger = init_logger(__name__)


def _resolve_operator_overload(op_name: str):
    """Resolve operator name to torch.ops OpOverload.

    Uses PyTorch's lookup_op utility.
    
    Args:
        op_name: Operator name in PyTorch format "namespace::name.overload"
                 Example: "aten::addmm.default"
    
    Returns:
        torch.ops OpOverload object
    """
    try:
        return lookup_op(op_name)
    except Exception as exc:
        raise ValueError(f"Failed to resolve operator '{op_name}'") from exc


@contextlib.contextmanager
def inductor_partition_rule_context(op_names: list[str]):
    if not op_names:
        logger.debug("No partition ops provided; skipping rule registration.")
        yield
        return

    from torch._inductor.scheduler import (  # type: ignore
        _custom_should_partition_fns,
        register_should_partition_rule,
    )

    unique_names = list(dict.fromkeys(op_names))
    overloads = [_resolve_operator_overload(name) for name in unique_names]

    def _always_partition(*_args, **_kwargs):
        return True

    # Save current state before registering
    saved_rules = _custom_should_partition_fns.copy()

    for overload in overloads:
        register_should_partition_rule(
            overload,
            _always_partition,
        )

    logger.debug("Registered inductor partition rules for ops: %s", unique_names)

    try:
        yield
    finally:
        # Clear and restore previous state
        _custom_should_partition_fns.clear()
        _custom_should_partition_fns.update(saved_rules)
        logger.debug("Restored previous partition rules state.")
