# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib

import torch
from torch._ops import OpOverload, OpOverloadPacket

from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_operator_name(op_name: str) -> tuple[str, str, str]:
    if not op_name:
        raise ValueError("Operator name must be non-empty")

    parts = op_name.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Operator name '{op_name}' must include a namespace and operator")

    namespace, remainder = parts[0], ".".join(parts[1:])
    if not remainder:
        raise ValueError(
            f"Operator name '{op_name}' must include an operator identifier")

    if "." in remainder:
        operator, overload = remainder.split(".", 1)
    else:
        operator, overload = remainder, "default"
    overload = overload or "default"
    return namespace, operator, overload


def _resolve_operator_overload(op_name: str):
    namespace, operator, overload = _parse_operator_name(op_name)
    target_overload = overload or "default"

    try:
        namespace_obj = getattr(torch.ops, namespace)
        operator_obj = getattr(namespace_obj, operator)
    except AttributeError as exc:
        if not hasattr(torch.ops, namespace):
            raise ValueError(
                f"Unknown operator namespace '{namespace}'") from exc
        raise ValueError(
            f"Unknown operator '{namespace}::{operator}'") from exc

    if isinstance(operator_obj, OpOverload):
        if overload not in ("default", ""):
            raise ValueError(
                f"Operator '{namespace}::{operator}' has no overload "
                f"'{overload}'")
        return operator_obj

    if isinstance(operator_obj, OpOverloadPacket):
        try:
            return getattr(operator_obj, target_overload)
        except AttributeError as exc:
            raise ValueError(
                f"Operator '{namespace}::{operator}' has no overload "
                f"'{target_overload}'") from exc

    try:
        return getattr(operator_obj, target_overload)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            f"Unsupported operator type for '{op_name}'") from exc


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

    for overload in overloads:
        inductor_scheduler.register_should_partition_rule(
            overload,
            _always_partition,
        )

    logger.debug("Registered inductor partition rules for ops: %s",
                 unique_names)

    try:
        yield
    finally:
        logger.debug("Partition rules remain registered; PyTorch does not "
                     "expose a clear API.")
