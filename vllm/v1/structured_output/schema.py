# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any


def is_string_schema_with_pattern_length(schema: Any) -> bool:
    """Return True when a string schema mixes a generative constraint
    (``pattern`` or ``format``) with explicit length bounds.

    xgrammar currently compiles a grammar for the generative constraint but
    silently ignores ``minLength``/``maxLength``, so the model can produce
    strings that violate the length bound.  We treat both ``pattern``+length
    and ``format``+length as unsupported to surface a clean error instead of
    producing quietly wrong output.
    """
    return (
        isinstance(schema, dict)
        and schema.get("type") == "string"
        and ("pattern" in schema or "format" in schema)
        and ("minLength" in schema or "maxLength" in schema)
    )


def has_string_pattern_with_length(schema: Any) -> bool:
    """Return whether a JSON schema combines string pattern and length bounds."""
    if is_string_schema_with_pattern_length(schema):
        return True

    if isinstance(schema, dict):
        return any(has_string_pattern_with_length(value) for value in schema.values())

    if isinstance(schema, list):
        return any(has_string_pattern_with_length(item) for item in schema)

    return False
