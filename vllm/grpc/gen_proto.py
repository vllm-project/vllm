#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate proto3 message definitions from Python types (Pydantic models, TypedDicts).

This tool introspects Python types at runtime and emits proto3 message
definitions to stdout. Complex union types that cannot be auto-mapped
are annotated with // MANUAL REVIEW comments.

Usage:
    python vllm/grpc/gen_proto.py module:ClassName [module:ClassName ...]

Examples:
    python vllm/grpc/gen_proto.py \
        vllm.entrypoints.openai.completion.protocol:CompletionRequest

    python vllm/grpc/gen_proto.py vllm.entrypoints.chat_utils:ConversationMessage

    python vllm/grpc/gen_proto.py vllm.inputs.data:TokensPrompt
"""

import argparse
import importlib
import sys
import types
import typing
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

# Type mapping from Python scalars to proto3 types
_SCALAR_MAP: dict[type, str] = {
    str: "string",
    int: "int32",
    float: "double",
    bool: "bool",
    bytes: "bytes",
}


@dataclass
class ProtoField:
    """Represents a single field in a proto3 message."""

    name: str
    proto_type: str
    field_number: int
    repeated: bool = False
    optional: bool = False
    is_map: bool = False
    map_key_type: str = ""
    map_value_type: str = ""
    comment: str = ""


@dataclass
class ProtoMessage:
    """Represents a proto3 message definition."""

    name: str
    fields: list[ProtoField] = field(default_factory=list)
    comment: str = ""


class ProtoGenerator:
    """Generates proto3 message definitions from Python types."""

    def __init__(self) -> None:
        self._messages: dict[str, ProtoMessage] = {}
        self._seen: set[str] = set()

    def generate(self, cls: type, name: str | None = None) -> str:
        """Generate proto3 text for a Python type and all nested types."""
        msg_name = name or cls.__name__
        self._generate_message(cls, msg_name)

        lines: list[str] = []
        for msg in self._messages.values():
            lines.extend(self._format_message(msg))
            lines.append("")
        return "\n".join(lines).rstrip()

    def _generate_message(self, cls: type, name: str) -> str:
        """Recursively generate a message for a type, returning its name."""
        if name in self._seen:
            return name
        self._seen.add(name)

        fields_info = self._get_fields(cls)
        proto_fields: list[ProtoField] = []

        for i, (field_name, python_type, required) in enumerate(fields_info, 1):
            pf = self._map_type(python_type, field_name, i)
            if not required and not pf.repeated and not pf.is_map:
                pf.optional = True
            proto_fields.append(pf)

        self._messages[name] = ProtoMessage(name=name, fields=proto_fields)
        return name

    def _get_fields(self, cls: type) -> list[tuple[str, Any, bool]]:
        """Extract fields as (name, type, required) from a type."""
        if self._is_pydantic(cls):
            return self._get_pydantic_fields(cls)
        elif self._is_typeddict(cls):
            return self._get_typeddict_fields(cls)
        else:
            raise TypeError(
                f"Unsupported type: {cls}. Expected Pydantic model or TypedDict."
            )

    def _get_pydantic_fields(self, cls: type) -> list[tuple[str, Any, bool]]:
        """Extract fields from a Pydantic model."""
        result = []
        for name, field_info in cls.model_fields.items():  # type: ignore[attr-defined]
            annotation = field_info.annotation
            required = field_info.is_required()
            result.append((name, annotation, required))
        return result

    def _get_typeddict_fields(self, cls: type) -> list[tuple[str, Any, bool]]:
        """Extract fields from a TypedDict."""
        hints = typing.get_type_hints(cls, include_extras=True)
        required_keys: set[str] = getattr(cls, "__required_keys__", set())
        result = []
        for name, tp in hints.items():
            required = name in required_keys
            # Unwrap NotRequired[T] and Required[T]
            tp = self._unwrap_notreq(tp)
            result.append((name, tp, required))
        return result

    def _map_type(
        self, python_type: Any, field_name: str, field_number: int
    ) -> ProtoField:
        """Map a Python type annotation to a ProtoField."""
        # Unwrap Annotated[T, ...]
        python_type = self._unwrap_annotated(python_type)

        # Check Optional[T] / T | None
        inner, is_optional = self._unwrap_optional(python_type)
        if is_optional:
            pf = self._map_type(inner, field_name, field_number)
            pf.optional = True
            return pf

        # Check scalar types
        if python_type in _SCALAR_MAP:
            return ProtoField(
                name=field_name,
                proto_type=_SCALAR_MAP[python_type],
                field_number=field_number,
            )

        origin = get_origin(python_type)
        args = get_args(python_type)

        # list[T] → repeated T
        if origin is list:
            if args:
                inner_pf = self._map_type(args[0], field_name, field_number)
                inner_pf.repeated = True
                inner_pf.optional = False
                return inner_pf
            return ProtoField(
                name=field_name,
                proto_type="string",
                field_number=field_number,
                repeated=True,
                comment="// list without type param",
            )

        # dict[K, V] → map<K, V> or string (for complex values)
        if origin is dict:
            if args and len(args) == 2:
                key_type = _SCALAR_MAP.get(args[0])
                val_type = _SCALAR_MAP.get(args[1])
                if key_type and val_type:
                    return ProtoField(
                        name=field_name,
                        proto_type="",
                        field_number=field_number,
                        is_map=True,
                        map_key_type=key_type,
                        map_value_type=val_type,
                    )
                elif key_type:
                    # Complex value type — use JSON string
                    return ProtoField(
                        name=field_name,
                        proto_type="string",
                        field_number=field_number,
                        comment=f"// JSON: dict[{args[0].__name__}, {args[1]}]",
                    )
            return ProtoField(
                name=field_name,
                proto_type="string",
                field_number=field_number,
                comment="// JSON: complex dict",
            )

        # Iterable[T] → repeated T (treat like list)
        if (
            origin is typing.Iterable
            or (hasattr(origin, "__name__") and origin.__name__ == "Iterable")
        ) and args:
            inner_pf = self._map_type(args[0], field_name, field_number)
            inner_pf.repeated = True
            inner_pf.optional = False
            return inner_pf

        # Union types (not Optional — already handled above)
        if origin is types.UnionType or origin is typing.Union:
            # Filter out NoneType (already handled by _unwrap_optional)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._map_type(non_none[0], field_name, field_number)

            # All scalars of same proto type?
            proto_types = {_SCALAR_MAP.get(t) for t in non_none}
            proto_types.discard(None)
            if len(proto_types) == 1:
                proto_type = proto_types.pop()
                assert proto_type is not None
                return ProtoField(
                    name=field_name,
                    proto_type=proto_type,
                    field_number=field_number,
                    comment=f"// Union: {' | '.join(t.__name__ for t in non_none)}",
                )

            # Complex union — emit as string with MANUAL REVIEW
            type_names = []
            for t in non_none:
                type_names.append(getattr(t, "__name__", str(t)))
            return ProtoField(
                name=field_name,
                proto_type="string",
                field_number=field_number,
                comment=f"// MANUAL REVIEW: Union[{', '.join(type_names)}]",
            )

        # Literal types
        if origin is typing.Literal:
            return ProtoField(
                name=field_name,
                proto_type="string",
                field_number=field_number,
                comment=f"// Literal: {args}",
            )

        # Nested Pydantic model or TypedDict
        if self._is_pydantic(python_type) or self._is_typeddict(python_type):
            nested_name = python_type.__name__
            self._generate_message(python_type, nested_name)
            return ProtoField(
                name=field_name,
                proto_type=nested_name,
                field_number=field_number,
            )

        # TypeAlias — try to resolve
        if isinstance(python_type, type):
            return ProtoField(
                name=field_name,
                proto_type="string",
                field_number=field_number,
                comment=f"// MANUAL REVIEW: {python_type.__name__}",
            )

        # Fallback
        return ProtoField(
            name=field_name,
            proto_type="string",
            field_number=field_number,
            comment=f"// MANUAL REVIEW: {python_type}",
        )

    def _format_message(self, msg: ProtoMessage) -> list[str]:
        """Format a ProtoMessage as proto3 text lines."""
        lines = []
        if msg.comment:
            lines.append(f"// {msg.comment}")
        lines.append(f"message {msg.name} {{")
        for f in msg.fields:
            lines.append(self._format_field(f))
        lines.append("}")
        return lines

    def _format_field(self, f: ProtoField) -> str:
        """Format a single ProtoField as a proto3 line."""
        parts = []
        if f.comment:
            parts.append(f"  {f.comment}")

        if f.is_map:
            line = (
                f"  map<{f.map_key_type}, {f.map_value_type}>"
                f" {f.name} = {f.field_number};"
            )
        else:
            prefix = ""
            if f.repeated:
                prefix = "repeated "
            elif f.optional:
                prefix = "optional "
            line = f"  {prefix}{f.proto_type} {f.name} = {f.field_number};"

        if parts:
            return "\n".join(parts) + "\n" + line
        return line

    @staticmethod
    def _is_pydantic(cls: Any) -> bool:
        try:
            from pydantic import BaseModel

            return isinstance(cls, type) and issubclass(cls, BaseModel)
        except ImportError:
            return False

    @staticmethod
    def _is_typeddict(cls: Any) -> bool:
        return (
            isinstance(cls, type)
            and issubclass(cls, dict)
            and hasattr(cls, "__annotations__")
        )

    @staticmethod
    def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
        """Unwrap Optional[T] / T | None → (T, True). Otherwise (tp, False)."""
        origin = get_origin(tp)
        if origin is types.UnionType or origin is typing.Union:
            args = get_args(tp)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) < len(args):
                if len(non_none) == 1:
                    return non_none[0], True
                # Union with None + multiple types
                return typing.Union[tuple(non_none)], True  # noqa: UP007
        return tp, False

    @staticmethod
    def _unwrap_annotated(tp: Any) -> Any:
        """Unwrap Annotated[T, ...] → T."""
        if get_origin(tp) is typing.Annotated:
            return get_args(tp)[0]
        return tp

    @staticmethod
    def _unwrap_notreq(tp: Any) -> Any:
        """Unwrap NotRequired[T] / Required[T] → T."""
        origin = get_origin(tp)
        if origin is typing.NotRequired or origin is typing.Required:  # type: ignore[attr-defined]
            return get_args(tp)[0]
        return tp


def resolve_type(spec: str) -> type:
    """Import and return a type from 'module.path:ClassName' spec."""
    module_path, sep, class_name = spec.partition(":")
    if not sep or not class_name:
        raise ValueError(f"Expected format 'module:ClassName', got '{spec}'")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate proto3 message definitions from Python types.",
        epilog="Example: python vllm/grpc/gen_proto.py "
        "vllm.entrypoints.openai.completion.protocol:CompletionRequest",
    )
    parser.add_argument(
        "types",
        nargs="+",
        metavar="module:ClassName",
        help="Python type specifications (e.g. vllm.inputs.data:TokensPrompt)",
    )
    parser.add_argument(
        "--package",
        default="vllm.grpc.generated",
        help="Proto package name (default: vllm.grpc.generated)",
    )
    args = parser.parse_args()

    print('syntax = "proto3";')
    print()
    print(f"package {args.package};")
    print()

    gen = ProtoGenerator()
    for spec in args.types:
        try:
            cls = resolve_type(spec)
        except (ValueError, ImportError, AttributeError) as e:
            print(f"// ERROR: Could not resolve {spec}: {e}", file=sys.stderr)
            return 1

        print(f"// Generated from {spec}")
        print(gen.generate(cls))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
