# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import types
import typing
from collections.abc import Sequence as SequenceABC
from typing import Any, Callable

import torch

try:
    from torch.library import Library, infer_schema as _infer_schema
except ImportError:
    from torch._library.infer_schema import infer_schema as _infer_schema
    from torch.library import Library


def _normalize_annotation(annotation: Any) -> Any:
    if isinstance(annotation, str):
        annotation = _resolve_string_annotation(annotation)

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is list and args:
        return typing.List[_normalize_annotation(args[0])]  # noqa: UP006

    if origin is SequenceABC and args:
        return typing.Sequence[_normalize_annotation(args[0])]  # noqa: UP006

    if origin in (typing.Union, types.UnionType) and args:
        normalized_args = tuple(_normalize_annotation(arg) for arg in args)
        none_type = type(None)
        if len(normalized_args) == 2 and none_type in normalized_args:
            value_type = normalized_args[0]
            if value_type is none_type:
                value_type = normalized_args[1]
            return typing.Optional[value_type]  # noqa: UP045
        return typing.Union[normalized_args]  # noqa: UP007

    return annotation


def _resolve_string_annotation(annotation: str) -> Any:
    namespace = {
        "None": type(None),
        "bool": bool,
        "dict": dict,
        "float": float,
        "int": int,
        "list": list,
        "str": str,
        "torch": torch,
        "typing": typing,
    }
    try:
        return eval(annotation, {"__builtins__": {}}, namespace)
    except Exception:
        return annotation


def _strip_unsupported_defaults(func: Callable[..., Any]) -> bool:
    signature = inspect.signature(func)
    parameters = []
    changed = False
    supported_defaults = (int, float, bool, type(None))
    for parameter in signature.parameters.values():
        default = parameter.default
        if default is not inspect.Parameter.empty and not isinstance(
            default, supported_defaults
        ):
            parameter = parameter.replace(default=inspect.Parameter.empty)
            changed = True
        parameters.append(parameter)
    if changed:
        func.__signature__ = signature.replace(parameters=parameters)
    return changed


def infer_schema(
    func: Callable[..., Any],
    mutates_args: list[str] | tuple[str, ...] | str,
) -> str:
    original_signature = getattr(func, "__signature__", None)
    try:
        return _infer_schema(func, mutates_args=mutates_args)
    except ValueError as first_error:
        annotations = getattr(func, "__annotations__", None)
        changed_annotations = False
        normalized = {
            name: _normalize_annotation(annotation)
            for name, annotation in annotations.items()
        } if annotations else None
        if normalized is not None and normalized != annotations:
            func.__annotations__ = normalized
            changed_annotations = True
        changed_signature = _strip_unsupported_defaults(func)
        try:
            return _infer_schema(func, mutates_args=mutates_args)
        except ValueError:
            if not changed_annotations and not changed_signature:
                raise first_error
            raise
        finally:
            if changed_annotations:
                func.__annotations__ = annotations
            if changed_signature:
                if original_signature is None:
                    try:
                        del func.__signature__
                    except AttributeError:
                        pass
                else:
                    func.__signature__ = original_signature
