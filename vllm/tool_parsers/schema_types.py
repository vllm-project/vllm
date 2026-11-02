# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSON types admitted by tool parameter schemas.

XML-style tool parsers receive every argument as raw text. These helpers use
the parameter's schema to decide whether to keep that text or decode it.
"""

from collections.abc import Callable, Iterable
from typing import Any

# Types admitted by each listed parameter of one tool, and by any other
# parameter (additionalProperties). None means unconstrained.
ParameterTypes = tuple[dict[str, set[str] | None], set[str] | None]


def _json_type(value: Any) -> str:
    """JSON Schema type name of a decoded JSON value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _schema_types(
    schema: Any, root: dict, refs: dict[str, set[str] | None]
) -> set[str] | None:
    """JSON types a parameter schema admits, or ``None`` if it admits any.

    Covers ``type``, ``const``, ``enum``, local ``$ref``, ``anyOf``/``oneOf``
    (union) and ``allOf`` (intersection), so that e.g. pydantic's
    ``Optional[str]`` counts as a string. *refs* memoizes each ``$ref``, which
    keeps the walk linear and ends cycles.
    """
    if not isinstance(schema, dict):
        return None
    constraints: list[set[str] | None] = []
    declared = schema.get("type")
    if isinstance(declared, str):
        constraints.append({declared})
    elif isinstance(declared, list):
        constraints.append({t for t in declared if isinstance(t, str)})
    if "const" in schema:
        constraints.append({_json_type(schema["const"])})
    if isinstance(schema.get("enum"), list):
        constraints.append({_json_type(v) for v in schema["enum"]})
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref not in refs:
            refs[ref] = None  # a ref still being resolved is unconstrained
            node = root if ref.startswith("#/") else None  # e.g. #/$defs/Name
            for part in ref[2:].split("/"):
                if isinstance(node, dict):
                    node = node.get(part.replace("~1", "/").replace("~0", "~"))
            refs[ref] = _schema_types(node, root, refs)
        constraints.append(refs[ref])
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list) and schema[key]:
            union: set[str] = set()
            for branch in schema[key]:
                if (types := _schema_types(branch, root, refs)) is None:
                    break  # an unconstrained branch admits anything
                union |= types
            else:
                constraints.append(union)
    if isinstance(schema.get("allOf"), list):
        constraints.extend(_schema_types(s, root, refs) for s in schema["allOf"])
    known = [  # an integer is also a number
        t | {"integer"} if "number" in t else t for t in constraints if t is not None
    ]
    return set.intersection(*known) if known else None


def parameter_types(tools: Iterable[Any] | None) -> dict[str, ParameterTypes]:
    """Map tool name to :data:`ParameterTypes`, for Chat or flat Responses tools."""
    types: dict[str, ParameterTypes] = {}
    for tool in tools or []:
        function = getattr(tool, "function", None) or tool
        if isinstance(function, dict):
            name, parameters = function.get("name"), function.get("parameters")
        else:
            name = getattr(function, "name", None)
            parameters = getattr(function, "parameters", None)
        if not name or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        refs: dict[str, set[str] | None] = {}
        types[name] = (
            {k: _schema_types(s, parameters, refs) for k, s in properties.items()},
            _schema_types(parameters.get("additionalProperties"), parameters, refs),
        )
    return types


class ParameterTypesCache:
    """Parameter types of the last tools list seen, built on first use."""

    _last: tuple[Any, dict[str, ParameterTypes]] = (None, {})

    def for_tools(self, tools: Iterable[Any] | None) -> dict[str, ParameterTypes]:
        if tools is not self._last[0]:
            self._last = tools, parameter_types(tools)
        return self._last[1]


def convert_value(
    raw: str, allowed: set[str] | None, decode: Callable[[str], Any]
) -> Any:
    """Decode *raw*, but keep the text if the schema admits a string, unless it
    decodes to another admitted type (``null`` for ``Optional[str]``).
    """
    if allowed == {"string"}:
        return raw
    decoded = decode(raw)
    if allowed is None or "string" not in allowed:
        return decoded
    return decoded if _json_type(decoded) in allowed - {"string"} else raw
