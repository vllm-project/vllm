# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib

import torch
from torch._library.utils import parse_namespace
from torch._ops import OpOverload, OpOverloadPacket

from vllm.logger import init_logger

logger = init_logger(__name__)


def _resolve_operator_overload(op_name: str):
    # Convert vLLM's dot notation (e.g., "aten.addmm.default")
    # to PyTorch's double-colon notation (e.g., "aten::addmm::default")
    # that parse_namespace expects
    pytorch_format = op_name.replace(".", "::")
    namespace, operator, overload = parse_namespace(pytorch_format)
    target_overload = overload or "default"

    try:
        namespace_obj = getattr(torch.ops, namespace)
        operator_obj = getattr(namespace_obj, operator)
    except AttributeError as exc:
        if not hasattr(torch.ops, namespace):
            raise ValueError(f"Unknown operator namespace '{namespace}'") from exc
        raise ValueError(f"Unknown operator '{namespace}::{operator}'") from exc

    if isinstance(operator_obj, OpOverload):
        if overload not in ("default", ""):
            raise ValueError(
                f"Operator '{namespace}::{operator}' has no overload '{overload}'"
            )
        return operator_obj

    if isinstance(operator_obj, OpOverloadPacket):
        try:
            return getattr(operator_obj, target_overload)
        except AttributeError as exc:
            raise ValueError(
                f"Operator '{namespace}::{operator}' has no overload "
                f"'{target_overload}'"
            ) from exc

    try:
        return getattr(operator_obj, target_overload)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"Unsupported operator type for '{op_name}'") from exc


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

    logger.debug("Registered inductor partition rules for ops: %s", unique_names)

    try:
        yield
    finally:
        logger.debug(
            "Partition rules remain registered; PyTorch does not expose a clear API."
        )
