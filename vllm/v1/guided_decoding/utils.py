# SPDX-License-Identifier: Apache-2.0

import json

import xgrammar

from vllm.sampling_params import SamplingParams


def has_xgrammar_unsupported_json_features(schema: dict) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for pattern restrictions
        if "pattern" in obj:
            return True

        # Check for enum restrictions
        if "enum" in obj:
            return True

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and any(
                key in obj for key in [
                    "minimum", "maximum", "exclusiveMinimum",
                    "exclusiveMaximum", "multipleOf"
                ]):
            return True

        # Check for array unsupported keywords
        if obj.get("type") == "array" and any(key in obj for key in [
                "uniqueItems", "contains", "minContains", "maxContains",
                "minItems", "maxItems"
        ]):
            return True

        # Unsupported keywords for strings
        if obj.get("type") == "string" and any(
                key in obj for key in ["minLength", "maxLength", "format"]):
            return True

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(key in obj for key in [
                "minProperties", "maxProperties", "propertyNames",
                "patternProperties"
        ]):
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def validate_guided_decoding_request(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by guided decoding.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.guided_decoding is None:
        return

    gd_params = sampling_params.guided_decoding

    if gd_params.regex:
        raise ValueError("Regex guided decoding is not supported.")

    if gd_params.choice:
        raise ValueError("Choice guided decoding is not supported.")

    if gd_params.json:
        if isinstance(gd_params.json, str):
            try:
                schema = json.loads(gd_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = gd_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError("The provided JSON schema contains features not "
                             "supported by xgrammar.")

    if gd_params.grammar:
        # EBNF style grammars only right now
        try:
            # parse the grammar, but we aren't compiling it.
            xgrammar.Grammar.from_ebnf(gd_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
