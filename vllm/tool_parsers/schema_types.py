# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JSON types admitted by tool parameter schemas.

XML-style tool parsers receive every argument as raw text. These helpers use
the parameter's schema to decide whether to keep that text or decode it.
"""

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple


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


def _intersect(a: set[str], b: set[str]) -> set[str]:
    """Intersect two type sets, treating ``integer`` as a subset of ``number``."""
    both = a & b
    if ("number" in a and "integer" in b) or ("integer" in a and "number" in b):
        both.add("integer")
    return both


def _resolve_ref(root: dict, ref: str) -> Any:
    """Resolve a local JSON pointer such as ``#/$defs/Name`` against *root*."""
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(part.replace("~1", "/").replace("~0", "~"))
    return node


def _schema_types(
    schema: Any, root: dict, refs: dict[str, set[str] | None]
) -> set[str] | None:
    """JSON types a parameter schema admits, or ``None`` if it admits any.

    Covers ``type`` (name or list), ``const``, ``enum``, local ``$ref``,
    ``anyOf``/``oneOf`` (union) and ``allOf`` (intersection), so that e.g.
    pydantic's ``Optional[str]`` (``anyOf: [string, null]``) counts as a string.
    *refs* memoizes each ``$ref`` (a ref still being resolved counts as
    unconstrained), keeping the walk linear in the schema size even when
    ``$defs`` reference each other many times or cyclically.
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
            refs[ref] = None
            refs[ref] = _schema_types(_resolve_ref(root, ref), root, refs)
        constraints.append(refs[ref])
    for key in ("anyOf", "oneOf"):
        if not isinstance(schema.get(key), list) or not schema[key]:
            continue
        union: set[str] = set()
        for branch in schema[key]:
            branch_types = _schema_types(branch, root, refs)
            if branch_types is None:
                break
            union |= branch_types
        else:
            constraints.append(union)
    if isinstance(schema.get("allOf"), list):
        constraints.extend(_schema_types(s, root, refs) for s in schema["allOf"])
    allowed: set[str] | None = None
    for c in constraints:
        if c is not None:
            allowed = c if allowed is None else _intersect(allowed, c)
    return allowed


class ParameterTypes(NamedTuple):
    """JSON types admitted by each top-level parameter of one tool.

    ``additional`` covers parameters not listed in ``properties``
    (``additionalProperties``). ``None`` means unconstrained.
    """

    properties: dict[str, set[str] | None]
    additional: set[str] | None

    def get(self, key: str) -> set[str] | None:
        return self.properties.get(key, self.additional)

    @classmethod
    def from_schema(cls, parameters: Any) -> "ParameterTypes":
        if not isinstance(parameters, dict):
            return cls({}, None)
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        refs: dict[str, set[str] | None] = {}
        return cls(
            {k: _schema_types(s, parameters, refs) for k, s in properties.items()},
            _schema_types(parameters.get("additionalProperties"), parameters, refs),
        )


def parameter_types(tools: Iterable[Any] | None) -> dict[str, ParameterTypes]:
    """Map tool name to its :class:`ParameterTypes`.

    Accepts Chat Completions tools (nested ``function``) and flat Responses
    function tools.
    """
    types: dict[str, ParameterTypes] = {}
    for tool in tools or []:
        function = getattr(tool, "function", None) or tool
        if isinstance(function, dict):
            name, parameters = function.get("name"), function.get("parameters")
        else:
            name = getattr(function, "name", None)
            parameters = getattr(function, "parameters", None)
        if name and isinstance(parameters, dict):
            types[name] = ParameterTypes.from_schema(parameters)
    return types


def convert_value(
    raw: str, allowed: set[str] | None, decode: Callable[[str], Any]
) -> Any:
    """Decode a parameter value, keeping the raw text where a string is meant.

    When the schema admits a string, the raw text is kept unless it decodes to
    another type the schema also admits (``null`` for ``Optional[str]``, ``3``
    for ``anyOf: [integer, string]``).
    Otherwise the result of *decode* is returned unchanged.
    """
    if allowed == {"string"}:
        return raw
    decoded = decode(raw)
    if allowed is None or "string" not in allowed:
        return decoded
    kind = _json_type(decoded)
    if kind == "integer" and "number" in allowed:
        kind = "number"
    return decoded if kind != "string" and kind in allowed else raw
