# SPDX-License-Identifier: Apache-2.0

from typing import List

from vllm.v1.guided_decoding.utils import (
    has_xgrammar_unsupported_json_features)


def test_has_xgrammar_unsupported_json_features():
    schemas_with_unsupported_features: List[dict] = [{
        "type": "string",
        "pattern": "^[a-zA-Z]+$"
    }, {
        "type":
        "string",
        "enum": ["active", "inactive", "pending"]
    }, {
        "type": "integer",
        "minimum": 0
    }, {
        "type": "integer",
        "maximum": 120
    }, {
        "type": "integer",
        "exclusiveMinimum": 120
    }, {
        "type": "integer",
        "exclusiveMaximum": 120
    }, {
        "type": "integer",
        "multipleOf": 120
    }, {
        "type": "number",
        "minimum": 0
    }, {
        "type": "number",
        "maximum": 120
    }, {
        "type": "number",
        "exclusiveMinimum": 120
    }, {
        "type": "number",
        "exclusiveMaximum": 120
    }, {
        "type": "number",
        "multipleOf": 120
    }, {
        "type": "array",
        "uniqueItems": True
    }, {
        "type": "array",
        "contains": {
            "type": "string"
        }
    }, {
        "type": "array",
        "minContains": 1
    }, {
        "type": "array",
        "maxContains": 5
    }, {
        "type": "array",
        "minItems": 1
    }, {
        "type": "array",
        "maxItems": 10
    }, {
        "type": "string",
        "minLength": 1
    }, {
        "type": "string",
        "maxLength": 100
    }, {
        "type": "string",
        "format": "email"
    }, {
        "type": "object",
        "minProperties": 1
    }, {
        "type": "object",
        "maxProperties": 5
    }, {
        "type": "object",
        "propertyNames": {
            "pattern": "^[a-z]+$"
        }
    }, {
        "type": "object",
        "patternProperties": {
            "^S": {
                "type": "string"
            }
        }
    }]

    for schema in schemas_with_unsupported_features:
        assert has_xgrammar_unsupported_json_features(schema)

    schema_without_unsupported_features: dict = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
            "status": {
                "type": "string"
            },
            "scores": {
                "type": "array",
                "items": {
                    "type": "number"
                }
            },
            "address": {
                "type": "object",
                "properties": {
                    "street": {
                        "type": "string"
                    },
                    "city": {
                        "type": "string"
                    }
                }
            }
        }
    }

    assert not has_xgrammar_unsupported_json_features(
        schema_without_unsupported_features)
