use std::collections::BTreeMap;

use serde_json::{Number, Value};

use crate::request::ChatTool;

/// Normalized parameter schemas for all tools in one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ToolSchemas {
    tools: BTreeMap<String, ToolSchema>,
}

/// Normalized parameter schema for one tool.
///
/// This is a minimal subset of JSON Schema with some normalization heuristics
/// to support common schema patterns and upstream schema variations, focused on
/// coercing raw string parameter values into more specific JSON types for
/// downstream tool call execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ToolSchema {
    params: BTreeMap<String, JsonParamType>,
}

/// Normalized JSON parameter type used for raw string coercion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum JsonParamType {
    String,
    Integer,
    Number,
    Boolean,
    Object,
    Array,
    Null,
    OneOf(Vec<JsonParamType>),
}

impl ToolSchemas {
    /// Normalize OpenAI-style tool parameter JSON schemas for one request.
    pub(super) fn from_tools(tools: &[ChatTool]) -> Self {
        let tools = tools
            .iter()
            .map(|tool| (tool.name.clone(), ToolSchema::from_schema(&tool.parameters)))
            .collect();

        Self { tools }
    }

    /// Convert raw string parameter values for one named tool.
    ///
    /// Unknown tool names use an empty schema, so all parameters fall back to
    /// strings.
    pub(super) fn convert_params_with_schema(
        &self,
        function_name: &str,
        params: Vec<(String, String)>,
    ) -> serde_json::Map<String, Value> {
        let tool_schema = self.tools.get(function_name).unwrap_or(ToolSchema::empty());
        let mut converted = serde_json::Map::with_capacity(params.len());
        for (name, value) in params {
            let value = tool_schema.convert(&name, &value);
            converted.insert(name, value);
        }
        converted
    }

    /// Convert one raw string parameter value for one named tool.
    pub(super) fn convert_param_with_schema(
        &self,
        function_name: &str,
        name: &str,
        value: &str,
    ) -> Value {
        let tool_schema = self.tools.get(function_name).unwrap_or(ToolSchema::empty());
        tool_schema.convert(name, value)
    }
}

impl ToolSchema {
    /// Return an empty schema with no parameter information, which causes all
    /// parameters to be treated as strings.
    const fn empty() -> &'static Self {
        static EMPTY: ToolSchema = ToolSchema {
            params: BTreeMap::new(),
        };
        &EMPTY
    }

    /// Normalize an OpenAI-style tool parameters JSON schema.
    fn from_schema(parameters: &Value) -> Self {
        let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
            return Self::default();
        };

        let params = properties
            .iter()
            .filter_map(|(name, schema)| {
                JsonParamType::from_schema(schema).map(|param_type| (name.clone(), param_type))
            })
            .collect();

        Self { params }
    }

    /// Convert one raw parameter value using its normalized schema type.
    ///
    /// If the parameter name is unknown, or we don't have a schema for it, or
    /// the value fails to convert, this falls back to returning the raw
    /// string as a JSON string value.
    fn convert(&self, name: &str, value: &str) -> Value {
        if value.eq_ignore_ascii_case("null") {
            return Value::Null;
        }

        let Some(param_type) = self.params.get(name) else {
            return Value::String(value.to_string());
        };

        convert_value(param_type, value).unwrap_or_else(|| Value::String(value.to_string()))
    }
}

impl JsonParamType {
    /// Normalize one parameter property schema.
    fn from_schema(schema: &Value) -> Option<Self> {
        let schema = schema.as_object()?;

        if let Some(type_value) = schema.get("type") {
            return Self::from_type_value(type_value);
        }

        if let Some(composite) = schema.get("anyOf").or_else(|| schema.get("oneOf")) {
            let param_type = composite
                .as_array()
                .map(|schemas| schemas.iter().filter_map(Self::from_schema).collect::<Vec<_>>())
                .filter(|types| !types.is_empty())
                .map(Self::one_of)
                .unwrap_or(Self::Object);
            return Some(param_type);
        }

        if schema.contains_key("enum") {
            return Some(Self::String);
        }
        if schema.contains_key("items") {
            return Some(Self::Array);
        }
        if schema.contains_key("properties") {
            return Some(Self::Object);
        }

        None
    }

    /// Normalize a JSON schema `type` value.
    fn from_type_value(type_value: &Value) -> Option<Self> {
        match type_value {
            Value::String(kind) => Self::from_type_name(kind),
            Value::Array(kinds) => {
                let types = kinds
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(Self::from_type_name)
                    .collect::<Vec<_>>();
                if types.is_empty() {
                    None
                } else {
                    Some(Self::one_of(types))
                }
            }
            _ => None,
        }
    }

    /// Normalize one JSON schema type name.
    fn from_type_name(kind: &str) -> Option<Self> {
        let kind = kind.trim().to_ascii_lowercase();
        match kind.as_str() {
            "string" | "str" | "text" | "varchar" | "char" | "enum" => Some(Self::String),
            "integer" | "int" => Some(Self::Integer),
            "number" | "float" => Some(Self::Number),
            "boolean" | "bool" | "binary" => Some(Self::Boolean),
            "object" => Some(Self::Object),
            "array" | "arr" | "sequence" => Some(Self::Array),
            "null" => Some(Self::Null),
            _ if kind.starts_with("int")
                || kind.starts_with("uint")
                || kind.starts_with("long")
                || kind.starts_with("short")
                || kind.starts_with("unsigned") =>
            {
                Some(Self::Integer)
            }
            _ if kind.starts_with("num") || kind.starts_with("float") => Some(Self::Number),
            _ if kind.starts_with("dict") => Some(Self::Object),
            _ if kind.starts_with("list") => Some(Self::Array),
            _ => None,
        }
    }

    /// Collapse a candidate type list into one normalized type.
    fn one_of(mut types: Vec<Self>) -> Self {
        if types.len() == 1 {
            types.remove(0)
        } else {
            Self::OneOf(types)
        }
    }
}

/// Convert one raw string value to a normalized JSON type.
fn convert_value(param_type: &JsonParamType, value: &str) -> Option<Value> {
    match param_type {
        JsonParamType::String => Some(Value::String(value.to_string())),
        JsonParamType::Integer => value.parse::<i64>().ok().map(Number::from).map(Value::Number),
        JsonParamType::Number => convert_number(value),
        JsonParamType::Boolean => convert_boolean(value),
        JsonParamType::Object | JsonParamType::Array => serde_json::from_str(value).ok(),
        JsonParamType::Null => value.eq_ignore_ascii_case("null").then_some(Value::Null),
        JsonParamType::OneOf(types) => {
            types.iter().find_map(|param_type| convert_value(param_type, value))
        }
    }
}

/// Convert one raw string value to a JSON number.
fn convert_number(value: &str) -> Option<Value> {
    if let Ok(parsed) = value.parse::<i64>() {
        return Some(Value::Number(Number::from(parsed)));
    }
    Number::from_f64(value.parse::<f64>().ok()?).map(Value::Number)
}

/// Convert one raw string value to a boolean.
fn convert_boolean(value: &str) -> Option<Value> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Some(Value::Bool(true)),
        "false" | "0" => Some(Value::Bool(false)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{ToolSchema, ToolSchemas};
    use crate::request::ChatTool;

    fn test_tool(name: &str, parameters: serde_json::Value) -> ChatTool {
        ChatTool {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
        }
    }

    #[test]
    fn invalid_schema_converts_everything_as_string() {
        let params = ToolSchema::from_schema(&json!({ "type": "object" }));

        assert_eq!(params.convert("count", "42"), json!("42"));
        assert_eq!(params.convert("count", "null"), json!(null));
    }

    #[test]
    fn skips_unknown_property_schema_and_unknown_type() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "unknown_schema": true,
                "unknown_type": { "type": "mystery" },
                "known": { "type": "integer" }
            }
        }));

        assert_eq!(params.convert("unknown_schema", "42"), json!("42"));
        assert_eq!(params.convert("unknown_type", "42"), json!("42"));
        assert_eq!(params.convert("known", "42"), json!(42));
    }

    #[test]
    fn converts_supported_types() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "count": { "type": "integer" },
                "size": { "type": "number" },
                "enabled": { "type": "boolean" },
                "payload": { "type": "object" },
                "items": { "type": "array" },
                "nothing": { "type": "null" }
            }
        }));

        assert_eq!(params.convert("text", "42"), json!("42"));
        assert_eq!(params.convert("count", "42"), json!(42));
        assert_eq!(params.convert("size", "5.0"), json!(5.0));
        assert_eq!(params.convert("enabled", "1"), json!(true));
        assert_eq!(params.convert("payload", r#"{"k":1}"#), json!({ "k": 1 }));
        assert_eq!(params.convert("items", "[1,2]"), json!([1, 2]));
        assert_eq!(params.convert("nothing", "null"), json!(null));
    }

    #[test]
    fn number_conversion_parses_int_then_float() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "value": { "type": "number" }
            }
        }));

        assert_eq!(params.convert("value", "5"), json!(5));
        assert_eq!(params.convert("value", "5.0"), json!(5.0));
        assert_eq!(params.convert("value", "5."), json!(5.0));
        assert_eq!(params.convert("value", "+1"), json!(1));
        assert_eq!(params.convert("value", "+1.0"), json!(1.0));
        assert_eq!(
            params.convert("value", "9223372036854775807.5"),
            json!(9223372036854775808.0)
        );
    }

    #[test]
    fn converts_upstream_aliases() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "s": { "type": "varchar" },
                "i": { "type": "unsigned_int" },
                "n": { "type": "float64" },
                "b": { "type": "binary" },
                "a": { "type": "sequence" },
                "o": { "type": "dict" }
            }
        }));

        assert_eq!(params.convert("s", "x"), json!("x"));
        assert_eq!(params.convert("i", "7"), json!(7));
        assert_eq!(params.convert("n", "7.5"), json!(7.5));
        assert_eq!(params.convert("b", "true"), json!(true));
        assert_eq!(params.convert("a", "[1]"), json!([1]));
        assert_eq!(params.convert("o", r#"{"x":1}"#), json!({ "x": 1 }));
    }

    #[test]
    fn preserves_union_type_order() {
        let integer_first = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "value": { "type": ["integer", "string"] }
            }
        }));
        let string_first = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "value": { "type": ["string", "integer"] }
            }
        }));

        assert_eq!(integer_first.convert("value", "42"), json!(42));
        assert_eq!(string_first.convert("value", "42"), json!("42"));
    }

    #[test]
    fn converts_composite_schemas() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "choice": {
                    "anyOf": [
                        { "type": "integer" },
                        { "type": "string" }
                    ]
                },
                "fallback_object": {
                    "oneOf": [
                        { "type": "mystery" }
                    ]
                }
            }
        }));

        assert_eq!(params.convert("choice", "42"), json!(42));
        assert_eq!(
            params.convert("fallback_object", r#"{"x":1}"#),
            json!({ "x": 1 })
        );
    }

    #[test]
    fn infers_type_from_schema_shape_without_type() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "choice": { "enum": ["a", "b"] },
                "items": { "items": { "type": "integer" } },
                "payload": { "properties": { "x": { "type": "integer" } } }
            }
        }));

        assert_eq!(params.convert("choice", "a"), json!("a"));
        assert_eq!(params.convert("items", "[1,2]"), json!([1, 2]));
        assert_eq!(params.convert("payload", r#"{"x":1}"#), json!({ "x": 1 }));
    }

    #[test]
    fn converts_params_for_known_tool() {
        let schemas = ToolSchemas::from_tools(&[test_tool(
            "search",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "topn": { "type": "integer" }
                }
            }),
        )]);

        let converted = schemas.convert_params_with_schema(
            "search",
            vec![
                ("query".to_string(), "rust".to_string()),
                ("topn".to_string(), "5".to_string()),
            ],
        );

        assert_eq!(converted.get("query"), Some(&json!("rust")));
        assert_eq!(converted.get("topn"), Some(&json!(5)));
    }

    #[test]
    fn convert_params_falls_back_to_string_for_failed_coercion() {
        let schemas = ToolSchemas::from_tools(&[test_tool(
            "convert",
            json!({
                "type": "object",
                "properties": {
                    "whole": { "type": "number" },
                    "flag": { "type": "boolean" },
                    "payload": { "type": "object" },
                    "items": { "type": "array" },
                    "missing_type": {}
                }
            }),
        )]);

        let converted = schemas.convert_params_with_schema(
            "convert",
            vec![
                ("whole".to_string(), "not-a-number".to_string()),
                ("flag".to_string(), "maybe".to_string()),
                ("payload".to_string(), "not-json".to_string()),
                ("items".to_string(), "not-json".to_string()),
                ("missing_type".to_string(), "42".to_string()),
                ("unknown_param".to_string(), "42".to_string()),
            ],
        );

        assert_eq!(converted.get("whole"), Some(&json!("not-a-number")));
        assert_eq!(converted.get("flag"), Some(&json!("maybe")));
        assert_eq!(converted.get("payload"), Some(&json!("not-json")));
        assert_eq!(converted.get("items"), Some(&json!("not-json")));
        assert_eq!(converted.get("missing_type"), Some(&json!("42")));
        assert_eq!(converted.get("unknown_param"), Some(&json!("42")));
    }

    #[test]
    fn convert_params_preserves_null_for_known_param() {
        let schemas = ToolSchemas::from_tools(&[test_tool(
            "convert",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            }),
        )]);

        let converted = schemas
            .convert_params_with_schema("convert", vec![("value".to_string(), "NULL".to_string())]);

        assert_eq!(converted.get("value"), Some(&json!(null)));
    }

    #[test]
    fn unknown_tool_converts_values_without_schema() {
        let schemas = ToolSchemas::from_tools(&[test_tool(
            "search",
            json!({ "type": "object", "properties": {} }),
        )]);

        let converted = schemas.convert_params_with_schema(
            "missing",
            vec![
                ("query".to_string(), "rust".to_string()),
                ("topn".to_string(), "5".to_string()),
                ("nullish".to_string(), "null".to_string()),
            ],
        );

        assert_eq!(converted.get("query"), Some(&json!("rust")));
        assert_eq!(converted.get("topn"), Some(&json!("5")));
        assert_eq!(converted.get("nullish"), Some(&json!(null)));
    }
}
