# SPDX-License-Identifier: Apache-2.0


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
