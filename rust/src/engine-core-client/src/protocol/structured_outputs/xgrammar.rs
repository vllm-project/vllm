// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Request-side xgrammar handling mirrored from Python's xgrammar backend.

use serde_json::{Map, Value};

const STRING_SUPPORTED_FORMATS: &[&str] = &[
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
];

/// Port of Python's `has_xgrammar_unsupported_json_features`. xgrammar
/// ignores these constraints rather than rejecting the schema.
pub(super) fn has_unsupported_json_features(schema: &Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    let types = schema_types(obj);
    let has_type = |name: &str| types.contains(&name);

    if (has_type("integer") || has_type("number")) && obj.contains_key("multipleOf") {
        return true;
    }
    if has_type("array")
        && ["uniqueItems", "contains", "minContains", "maxContains"]
            .iter()
            .any(|key| obj.contains_key(*key))
    {
        return true;
    }
    if has_type("string") {
        let unsupported_format = obj.get("format").is_some_and(|format| {
            !format.as_str().is_some_and(|f| STRING_SUPPORTED_FORMATS.contains(&f))
        });
        if unsupported_format || has_pattern_and_length_bounds(obj) {
            return true;
        }
    }
    if has_type("object") {
        if let Some(property_names) = obj.get("propertyNames")
            && (property_names.as_object().is_some_and(has_pattern_and_length_bounds)
                || obj.contains_key("patternProperties")
                || obj.contains_key("properties")
                || obj
                    .get("unevaluatedProperties")
                    .is_some_and(|value| *value != Value::Bool(true)))
        {
            return true;
        }
        if let Some(Value::Object(pattern_properties)) = obj.get("patternProperties")
            && (obj.contains_key("properties") || pattern_properties.len() > 1)
        {
            return true;
        }
    }
    if obj
        .get("allOf")
        .and_then(Value::as_array)
        .is_some_and(|branches| branches.len() >= 2)
    {
        return true;
    }

    // serde_json's parser recursion limit bounds this recursion.
    obj.values().any(|value| match value {
        Value::Object(_) => has_unsupported_json_features(value),
        Value::Array(items) => items.iter().any(has_unsupported_json_features),
        _ => false,
    })
}

fn schema_types(obj: &Map<String, Value>) -> Vec<&str> {
    match obj.get("type") {
        Some(Value::String(name)) => vec![name.as_str()],
        Some(Value::Array(names)) => names.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn has_pattern_and_length_bounds(obj: &Map<String, Value>) -> bool {
    (obj.contains_key("pattern") || obj.contains_key("format"))
        && (obj.contains_key("minLength") || obj.contains_key("maxLength"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn unsupported_json_features_match_python() {
        let unsupported = [
            json!({"type": "integer", "multipleOf": 2}),
            json!({"type": ["number", "null"], "multipleOf": 0.5}),
            json!({"type": "array", "uniqueItems": true}),
            json!({"type": "array", "contains": {"type": "string"}}),
            json!({"type": "string", "format": "binary"}),
            json!({"type": "string", "format": 7}),
            json!({"type": "string", "pattern": "a+", "maxLength": 3}),
            json!({"type": "object", "propertyNames": {"pattern": "a", "minLength": 1}}),
            json!({"type": "object", "propertyNames": {}, "properties": {}}),
            json!({"type": "object", "propertyNames": {}, "unevaluatedProperties": false}),
            json!({"type": "object", "patternProperties": {"a": {}, "b": {}}}),
            json!({"type": "object", "patternProperties": {"a": {}}, "properties": {}}),
            json!({"allOf": [{"type": "string"}, {"enum": ["yes", "no"]}]}),
            json!({"type": "object", "properties": {"x": {"type": "array", "uniqueItems": true}}}),
            json!({"anyOf": [{"type": "integer", "multipleOf": 3}]}),
        ];
        for schema in unsupported {
            assert!(has_unsupported_json_features(&schema), "{schema}");
        }

        let supported = [
            json!({"type": "object", "properties": {"x": {"type": "string"}}}),
            json!({"type": "string", "format": "email"}),
            json!({"type": "string", "pattern": "a+"}),
            json!({"type": "array", "items": {"type": "integer"}, "minItems": 1}),
            json!({"multipleOf": 2}),
            json!({"type": "object", "propertyNames": {"pattern": "a"}}),
            json!({"type": "object", "propertyNames": {}, "unevaluatedProperties": true}),
            json!({"allOf": [{"type": "string"}]}),
            json!(true),
        ];
        for schema in supported {
            assert!(!has_unsupported_json_features(&schema), "{schema}");
        }
    }
}
