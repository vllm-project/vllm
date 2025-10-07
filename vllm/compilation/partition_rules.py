# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib

from torch._library.utils import lookup_op

from vllm.logger import init_logger

logger = init_logger(__name__)


def _resolve_operator_overload(op_name: str):
    """Resolve vLLM operator name to torch.ops OpOverload.

    Uses PyTorch's lookup_op utility.
    Example: "aten.addmm.default" -> torch.ops.aten.addmm.default
    """
    if "." not in op_name:
        raise ValueError(f"Invalid operator name: {op_name}")

    # Convert vLLM format to PyTorch format (only first dot)
    # "aten.addmm.default" -> "aten::addmm.default"
    namespace, rest = op_name.split(".", 1)
    pytorch_qualname = f"{namespace}::{rest}"

    # Use PyTorch's official lookup_op
    try:
        return lookup_op(pytorch_qualname)
    except Exception as exc:
        raise ValueError(f"Failed to resolve operator '{op_name}': {exc}") from exc


@contextlib.contextmanager
def inductor_partition_rule_context(op_names: list[str]):
    if not op_names:
        logger.debug("No partition ops provided; skipping rule registration.")
        yield
        return

    from torch._inductor import scheduler as inductor_scheduler  # type: ignore

    unique_names = list(dict.fromkeys(op_names))
    overloads = [_resolve_operator_overload(name) for name in unique_names]

    def _always_partition(*_args, **_kwargs):
        return True

    # Save current state before registering
    saved_rules = inductor_scheduler._custom_should_partition_fns.copy()

    for overload in overloads:
        inductor_scheduler.register_should_partition_rule(
            overload,
            _always_partition,
        )

    logger.debug("Registered inductor partition rules for ops: %s", unique_names)

    try:
        yield
    finally:
        # Clear and restore previous state
        inductor_scheduler._custom_should_partition_fns.clear()
        inductor_scheduler._custom_should_partition_fns.update(saved_rules)
        logger.debug("Restored previous partition rules state.")
