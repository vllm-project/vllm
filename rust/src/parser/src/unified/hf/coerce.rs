// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Cast parsed tool-call arguments using the calling tool's JSON schema.
//!
//! Port of `_schema_types`, `_coerce`, and `ResponseParser._coerce_tool_calls`
//! from `response_parser.py`. This intentionally does not reuse
//! `tool::parameters`: the template's contract is Transformers' coercion.

use std::collections::HashMap;

use serde_json::{Map, Number, Value};

use super::content::{python_float, python_int, python_strip};
use crate::tool::Tool;

/// Maps tool name -> schema `properties`, used to cast parsed tool-call arguments.
#[derive(Debug, Clone, Default)]
pub(super) struct ToolParams {
    params: HashMap<String, Map<String, Value>>,
}

impl ToolParams {
    /// Collect `parameters.properties` for each request tool.
    pub fn new(tools: &[Tool]) -> Self {
        let params = tools
            .iter()
            .map(|tool| {
                let properties = tool
                    .parameters
                    .get("properties")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                (tool.name.clone(), properties)
            })
            .collect();
        Self { params }
    }

    /// Return whether no tools were provided, in which case coercion is skipped.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Cast string arguments of function-shaped values in place.
    pub fn coerce_tool_calls(&self, value: &mut Value) {
        if let Value::Array(items) = value {
            for item in items {
                self.coerce_tool_calls(item);
            }
            return;
        }
        let Some(function) = value.get_mut("function").and_then(Value::as_object_mut) else {
            return;
        };
        let Some(Value::String(name)) = function.get("name") else {
            return;
        };
        let Some(properties) = self.params.get(name).filter(|properties| !properties.is_empty())
        else {
            return;
        };
        let Some(Value::Object(arguments)) = function.get_mut("arguments") else {
            return;
        };
        for (key, argument) in arguments.iter_mut() {
            let Some(schema) = properties.get(key) else {
                continue;
            };
            let types = schema_types(schema);
            if types.is_empty() {
                continue;
            }
            match argument {
                Value::String(raw) => *argument = coerce(raw, &types),
                // Duplicate keys collected by `merge_duplicates`.
                Value::Array(items) if !types.contains(&"array") => {
                    for item in items {
                        if let Value::String(raw) = item {
                            *item = coerce(raw, &types);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Collect the JSON Schema type names a parameter accepts.
fn schema_types(schema: &Value) -> Vec<&str> {
    let Some(schema) = schema.as_object() else {
        return Vec::new();
    };
    let mut types = Vec::new();
    match schema.get("type") {
        Some(Value::String(declared)) => types.push(declared.as_str()),
        Some(Value::Array(declared)) => types.extend(declared.iter().filter_map(Value::as_str)),
        _ => {}
    }
    for union_name in ["anyOf", "oneOf"] {
        if let Some(Value::Array(choices)) = schema.get(union_name) {
            for choice in choices {
                types.extend(schema_types(choice));
            }
        }
    }
    // `nullable` is how get_json_schema marks Optionals.
    let nullable = schema.get("nullable").is_some_and(|value| !is_falsy(value));
    if nullable && !types.contains(&"null") {
        types.push("null");
    }
    types
}

/// Cast `raw` to the first declared type it parses as; `string` params, unknown
/// types and failed casts all keep the original text.
fn coerce(raw: &str, types: &[&str]) -> Value {
    for type_name in types {
        match *type_name {
            "integer" => {
                if let Some(value) = python_int(raw) {
                    return value;
                }
            }
            "number" => {
                // NaN / inf are not valid JSON numbers.
                let Some(number) = python_float(raw).filter(|number| number.is_finite()) else {
                    continue;
                };
                // Preserve ints when the source text had no fractional part.
                if number.fract() == 0.0 && !raw.contains('.') {
                    if let Some(value) = python_int(raw) {
                        return value;
                    }
                    if number.abs() < 9.007_199_254_740_992e15 {
                        return Value::Number((number as i64).into());
                    }
                }
                if let Some(number) = Number::from_f64(number) {
                    return Value::Number(number);
                }
            }
            "boolean" => match python_strip(raw).to_lowercase().as_str() {
                "true" | "1" => return Value::Bool(true),
                "false" | "0" => return Value::Bool(false),
                _ => {}
            },
            "null" if matches!(python_strip(raw), "null" | "None") => return Value::Null,
            "object" | "array" => {
                let decoded: Option<Value> = serde_json::from_str(raw).ok();
                match (decoded, *type_name) {
                    (Some(value @ Value::Object(_)), "object")
                    | (Some(value @ Value::Array(_)), "array") => return value,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Value::String(raw.to_string())
}

/// Python truthiness of a JSON value.
fn is_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(value) => !value,
        Value::Number(number) => number.as_f64() == Some(0.0),
        Value::String(string) => string.is_empty(),
        Value::Array(values) => values.is_empty(),
        Value::Object(object) => object.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tool(parameters: Value) -> Tool {
        Tool {
            name: "set_alarm".to_string(),
            description: None,
            parameters,
            strict: None,
        }
    }

    fn call(arguments: Value) -> Value {
        json!({"type": "function", "function": {"name": "set_alarm", "arguments": arguments}})
    }

    #[test]
    fn casts_declared_types_and_keeps_raw_on_failure() {
        let params = ToolParams::new(&[tool(json!({
            "type": "object",
            "properties": {
                "hour": {"type": "integer"},
                "ratio": {"type": "number"},
                "whole": {"type": "number"},
                "enabled": {"type": "boolean"},
                "label": {"type": "string"},
                "maybe": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "nullable": {"type": "integer", "nullable": true},
                "days": {"type": "array"},
                "meta": {"type": "object"},
                "bad": {"type": "integer"},
            },
        }))]);
        let mut value = call(json!({
            "hour": "7",
            "ratio": "0.5",
            "whole": "1e3",
            "enabled": " False ",
            "label": "7",
            "maybe": "None",
            "nullable": "null",
            "days": "[\"mon\"]",
            "meta": "[1]",
            "bad": "seven",
            "unknown": "1",
        }));
        params.coerce_tool_calls(&mut value);
        assert_eq!(
            value["function"]["arguments"],
            json!({
                "hour": 7,
                "ratio": 0.5,
                "whole": 1000,
                "enabled": false,
                "label": "7",
                "maybe": null,
                "nullable": null,
                "days": ["mon"],
                "meta": "[1]",
                "bad": "seven",
                "unknown": "1",
            })
        );
    }

    #[test]
    fn merged_duplicates_are_cast_element_wise_unless_array_typed() {
        let params = ToolParams::new(&[tool(json!({
            "properties": {"hours": {"type": "integer"}, "tags": {"type": ["array", "string"]}},
        }))]);
        let mut value = json!([call(json!({"hours": ["7", "8", 9], "tags": ["1", "2"]}))]);
        params.coerce_tool_calls(&mut value);
        assert_eq!(
            value[0]["function"]["arguments"],
            json!({"hours": [7, 8, 9], "tags": ["1", "2"]})
        );
    }

    #[test]
    fn unknown_tools_and_non_function_values_are_untouched() {
        let params = ToolParams::new(&[tool(json!({"properties": {"hour": {"type": "integer"}}}))]);
        let mut other =
            json!({"type": "function", "function": {"name": "other", "arguments": {"hour": "7"}}});
        let mut text = json!("7");
        params.coerce_tool_calls(&mut other);
        params.coerce_tool_calls(&mut text);
        assert_eq!(other["function"]["arguments"]["hour"], json!("7"));
        assert_eq!(text, json!("7"));
    }
}
