# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generic utilities for converting between protobuf messages and Python types.

Uses google.protobuf.json_format for recursive conversion,
with pluggable field transforms for domain-specific renames/parsing.
"""

from typing import Any

from google.protobuf.json_format import (  # type: ignore[import-untyped]
    MessageToDict,
    ParseDict,
)
from google.protobuf.message import Message  # type: ignore[import-untyped]

# Type alias for field transform dicts.
# Format: {proto_field_name: (python_field_name, transform_fn_or_None)}
FieldTransforms = dict[str, tuple[str, Any]]


def proto_to_dict(
    message: Message,
    transforms: FieldTransforms | None = None,
) -> dict:
    """Convert a protobuf message to a dict.

    Uses MessageToDict for recursive conversion, then applies field
    renames and transforms if provided.
    """
    raw = MessageToDict(message, preserving_proto_field_name=True)
    if transforms:
        return _apply_transforms(raw, transforms)
    return raw


def from_proto(
    message: Message,
    request_class: type,
    transforms: FieldTransforms | None = None,
) -> Any:
    """Convert a protobuf message to a Pydantic/dataclass instance."""
    return request_class(**proto_to_dict(message, transforms))


def to_proto(data: dict | tuple | list, message_class: type[Message]) -> Message:
    """Convert a dict/tuple/list to a protobuf message.

    Uses ParseDict for recursive conversion. If data is a tuple or
    non-dict list, elements are mapped to proto fields by definition order.
    """
    if not isinstance(data, dict):
        fields = message_class.DESCRIPTOR.fields
        data = {fields[i].name: v for i, v in enumerate(data)}
    return ParseDict(data, message_class(), ignore_unknown_fields=True)


def _apply_transforms(obj: Any, transforms: FieldTransforms) -> Any:
    """Recursively apply field transforms to a dict/list structure."""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            value = _apply_transforms(value, transforms)
            if key in transforms:
                new_key, fn = transforms[key]
                result[new_key] = fn(value) if fn else value
            else:
                result[key] = value
        return result
    elif isinstance(obj, list):
        return [_apply_transforms(item, transforms) for item in obj]
    return obj
