// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde_json::{Map, Number, Value};

use crate::tool::Tool;

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

/// Parameter input for schema-aware conversion.
///
/// It can be either a raw text string, or a structured input with named child elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ParamInput {
    Text(String),
    #[allow(dead_code)]
    Elements(Vec<ParamElement>),
}

impl From<String> for ParamInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// One named structured parameter child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ParamElement {
    pub name: String,
    pub value: ParamInput,
}

/// Normalized JSON parameter type used for raw string coercion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum JsonParamType {
    String,
    Integer,
    Number,
    Boolean,
    Object {
        properties: BTreeMap<String, JsonParamType>,
        additional_properties: Option<Box<JsonParamType>>,
    },
    Array {
        items: Option<Box<JsonParamType>>,
    },
    Null,
    OneOf(Vec<JsonParamType>),
}

impl ToolSchemas {
    /// Normalize OpenAI-style tool parameter JSON schemas for one request.
    pub(super) fn from_tools(tools: &[Tool]) -> Self {
        let tools = tools
            .iter()
            .map(|tool| (tool.name.clone(), ToolSchema::from_schema(&tool.parameters)))
            .collect();

        Self { tools }
    }

    /// Convert parameter values for one named tool.
    ///
    /// Unknown tool names use an empty schema, so all parameters fall back to
    /// strings or object-like JSON for structured inputs.
    pub(super) fn convert_params_with_schema<P>(
        &self,
        function_name: &str,
        params: Vec<(String, P)>,
    ) -> Map<String, Value>
    where
        P: Into<ParamInput>,
    {
        let tool_schema = self.tools.get(function_name).unwrap_or(ToolSchema::empty());
        let mut converted = Map::with_capacity(params.len());
        for (name, value) in params {
            let value = tool_schema.convert(&name, value.into());
            converted.insert(name, value);
        }
        converted
    }

    /// Convert one parameter value for one named tool.
    pub(super) fn convert_param_with_schema<P>(
        &self,
        function_name: &str,
        name: &str,
        value: P,
    ) -> Value
    where
        P: Into<ParamInput>,
    {
        let tool_schema = self.tools.get(function_name).unwrap_or(ToolSchema::empty());
        tool_schema.convert(name, value.into())
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

    /// Convert one parameter value using its normalized schema type.
    ///
    /// If the parameter name is unknown, or we don't have a schema for it, or
    /// the value fails to convert, this falls back to returning the raw
    /// string as a JSON string value, or object-like JSON for structured input.
    fn convert(&self, name: &str, input: ParamInput) -> Value {
        convert_with_optional_schema(self.params.get(name), &input)
    }
}

impl JsonParamType {
    /// Normalize one parameter property schema.
    fn from_schema(schema: &Value) -> Option<Self> {
        let schema = schema.as_object()?;

        if let Some(type_value) = schema.get("type") {
            return Self::from_type_value(type_value, schema);
        }

        if let Some(composite) = schema.get("anyOf").or_else(|| schema.get("oneOf")) {
            let param_type = composite
                .as_array()
                .map(|schemas| schemas.iter().filter_map(Self::from_schema).collect::<Vec<_>>())
                .filter(|types| !types.is_empty())
                .map(Self::one_of)
                .unwrap_or_else(|| Self::object_from_schema(Some(schema)));
            return Some(param_type);
        }

        // Typically, these types are already handled by checking the "type" field, but
        // we can also infer them from their characteristic fields if "type" is missing.
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            // Enum values are treated as strings, except that a `null` member
            // makes the parameter nullable (mirrors Python's enum type
            // inference), so a literal "null" coerces to JSON null.
            if values.iter().any(Value::is_null) {
                return Some(Self::one_of(vec![Self::String, Self::Null]));
            }
            return Some(Self::String);
        }
        if schema.contains_key("items") {
            return Some(Self::array_from_schema(Some(schema)));
        }
        if schema.contains_key("properties") || schema.contains_key("additionalProperties") {
            return Some(Self::object_from_schema(Some(schema)));
        }

        None
    }

    /// Normalize a JSON schema `type` value.
    fn from_type_value(type_value: &Value, schema: &Map<String, Value>) -> Option<Self> {
        match type_value {
            Value::String(kind) => Self::from_type_name(kind, Some(schema)),
            Value::Array(kinds) => {
                let types = kinds
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(|kind| Self::from_type_name(kind, Some(schema)))
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
    fn from_type_name(kind: &str, schema: Option<&Map<String, Value>>) -> Option<Self> {
        let kind = kind.trim().to_ascii_lowercase();
        match kind.as_str() {
            "string" | "str" | "text" | "varchar" | "char" | "enum" => Some(Self::String),
            "integer" | "int" => Some(Self::Integer),
            "number" | "float" | "double" => Some(Self::Number),
            "boolean" | "bool" | "binary" => Some(Self::Boolean),
            "object" | "dict" | "map" => Some(Self::object_from_schema(schema)),
            "array" | "arr" | "list" | "sequence" => Some(Self::array_from_schema(schema)),
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
            _ if kind.starts_with("dict") => Some(Self::object_from_schema(schema)),
            _ if kind.starts_with("list") => Some(Self::array_from_schema(schema)),
            _ => None,
        }
    }

    /// Normalize object schema fields.
    fn object_from_schema(schema: Option<&Map<String, Value>>) -> Self {
        let properties = schema
            .and_then(|schema| schema.get("properties"))
            .and_then(Value::as_object)
            .map(|properties| {
                properties
                    .iter()
                    .filter_map(|(name, schema)| {
                        Self::from_schema(schema).map(|param_type| (name.clone(), param_type))
                    })
                    .collect()
            })
            .unwrap_or_default();

        let additional_properties =
            schema.and_then(|schema| schema.get("additionalProperties")).and_then(|schema| {
                if schema.is_object() {
                    Self::from_schema(schema).map(Box::new)
                } else {
                    None
                }
            });

        Self::Object {
            properties,
            additional_properties,
        }
    }

    /// Normalize array schema fields.
    fn array_from_schema(schema: Option<&Map<String, Value>>) -> Self {
        let items = schema
            .and_then(|schema| schema.get("items"))
            .and_then(Self::from_schema)
            .map(Box::new);

        Self::Array { items }
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

/// Convert one parameter input to a normalized JSON value.
fn convert_with_optional_schema(param_type: Option<&JsonParamType>, input: &ParamInput) -> Value {
    // Coerce the literal text `null` to JSON null, except for `string`-typed
    // params, where it must stay the string "null": a model emitting the literal
    // text "null" for a string field means the string, not a missing value.
    if let ParamInput::Text(value) = input
        && value.eq_ignore_ascii_case("null")
        && param_type != Some(&JsonParamType::String)
    {
        return Value::Null;
    }

    // If we have a schema, try to convert the value using it.
    if let Some(param_type) = param_type
        && let Some(value) = try_convert_value(param_type, input)
    {
        return value;
    }
    // We don't have a schema, or conversion failed, use fallback logic.
    match input {
        ParamInput::Text(value) => Value::String(value.clone()),
        ParamInput::Elements(elements) => {
            // Convert structured input to object without a schema.
            Value::Object(convert_elements_to_object(elements, &BTreeMap::new(), None))
        }
    }
}

/// Convert one parameter input to a normalized JSON type.
fn try_convert_value(param_type: &JsonParamType, input: &ParamInput) -> Option<Value> {
    match input {
        ParamInput::Text(value) => try_convert_text_value(param_type, value),
        ParamInput::Elements(elements) => try_convert_elements_value(param_type, elements),
    }
}

/// Convert one raw string value to a normalized JSON type.
fn try_convert_text_value(param_type: &JsonParamType, value: &str) -> Option<Value> {
    match param_type {
        JsonParamType::String => Some(Value::String(value.to_string())),
        JsonParamType::Integer => value.parse::<i64>().ok().map(Number::from).map(Value::Number),
        JsonParamType::Number => try_convert_number(value),
        JsonParamType::Boolean => try_convert_boolean(value),
        JsonParamType::Object { .. } if value.is_empty() => Some(Value::Object(Map::new())),
        JsonParamType::Array { .. } if value.is_empty() => Some(Value::Array(Vec::new())),
        JsonParamType::Object { .. } | JsonParamType::Array { .. } => {
            // For composite types with string input, simply interpret the string as JSON.
            serde_json::from_str(value).ok()
        }
        JsonParamType::Null => value.eq_ignore_ascii_case("null").then_some(Value::Null),
        JsonParamType::OneOf(types) => {
            types.iter().find_map(|param_type| try_convert_text_value(param_type, value))
        }
    }
}

/// Convert one structured parameter input to a normalized JSON type.
fn try_convert_elements_value(
    param_type: &JsonParamType,
    elements: &[ParamElement],
) -> Option<Value> {
    match param_type {
        JsonParamType::Object {
            properties,
            additional_properties,
        } => Some(Value::Object(convert_elements_to_object(
            elements,
            properties,
            additional_properties.as_deref(),
        ))),
        JsonParamType::Array { items } => Some(Value::Array(
            // Collect all child elements into an array, regardless of their names.
            elements
                .iter()
                .map(|element| convert_with_optional_schema(items.as_deref(), &element.value))
                .collect(),
        )),
        JsonParamType::OneOf(types) => types
            .iter()
            .find_map(|param_type| try_convert_elements_value(param_type, elements)),

        // Primitive types can't be converted from structured input.
        JsonParamType::String
        | JsonParamType::Integer
        | JsonParamType::Number
        | JsonParamType::Boolean
        | JsonParamType::Null => None,
    }
}

/// Convert structured elements to an object, using field schemas when present.
fn convert_elements_to_object(
    elements: &[ParamElement],
    properties: &BTreeMap<String, JsonParamType>,
    additional_properties: Option<&JsonParamType>,
) -> Map<String, Value> {
    let mut object = Map::with_capacity(elements.len());
    for element in elements {
        let param_type = properties.get(&element.name).or(additional_properties);
        let value = convert_with_optional_schema(param_type, &element.value);
        insert_object_value(&mut object, element.name.clone(), value);
    }
    object
}

/// Insert an object field while preserving duplicate keys as arrays.
fn insert_object_value(object: &mut Map<String, Value>, key: String, value: Value) {
    if let Some(existing) = object.get_mut(&key) {
        match existing {
            // Collect values under the same key into an array.
            Value::Array(values) => values.push(value),
            existing => {
                let first = std::mem::replace(existing, Value::Null);
                *existing = Value::Array(vec![first, value]);
            }
        }
    } else {
        object.insert(key, value);
    }
}

/// Convert one raw string value to a JSON number.
fn try_convert_number(value: &str) -> Option<Value> {
    serde_json::from_str::<Number>(value)
        .or_else(|_| value.parse::<i64>().map(Number::from))
        .or_else(|_| value.parse::<f64>().ok().and_then(Number::from_f64).ok_or(()))
        .ok()
        .map(Value::Number)
}

/// Convert one raw string value to a boolean.
fn try_convert_boolean(value: &str) -> Option<Value> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Some(Value::Bool(true)),
        "false" | "0" => Some(Value::Bool(false)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{ParamElement, ParamInput, ToolSchema, ToolSchemas};
    use crate::tool::Tool;

    fn test_tool(name: &str, parameters: serde_json::Value) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
        }
    }

    #[test]
    fn invalid_schema_converts_everything_as_string() {
        let params = ToolSchema::from_schema(&json!({ "type": "object" }));

        assert_eq!(params.convert("count", text("42")), json!("42"));
        assert_eq!(params.convert("count", text("null")), json!(null));
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

        assert_eq!(params.convert("unknown_schema", text("42")), json!("42"));
        assert_eq!(params.convert("unknown_type", text("42")), json!("42"));
        assert_eq!(params.convert("known", text("42")), json!(42));
    }

    #[test]
    fn converts_supported_types() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "count": { "type": "integer" },
                "size": { "type": "number" },
                "ratio": { "type": "double" },
                "enabled": { "type": "boolean" },
                "payload": { "type": "object" },
                "mapping": { "type": "map" },
                "items": { "type": "array" },
                "names": { "type": "list" },
                "nothing": { "type": "null" }
            }
        }));

        assert_eq!(params.convert("text", text("42")), json!("42"));
        assert_eq!(params.convert("count", text("42")), json!(42));
        assert_eq!(params.convert("size", text("5.0")), json!(5.0));
        assert_eq!(params.convert("ratio", text("2.5")), json!(2.5));
        assert_eq!(params.convert("enabled", text("1")), json!(true));
        assert_eq!(
            params.convert("payload", text(r#"{"k":1}"#)),
            json!({ "k": 1 })
        );
        assert_eq!(
            params.convert("mapping", text(r#"{"k":1}"#)),
            json!({ "k": 1 })
        );
        assert_eq!(params.convert("items", text("[1,2]")), json!([1, 2]));
        assert_eq!(
            params.convert("names", text(r#"["a","b"]"#)),
            json!(["a", "b"])
        );
        assert_eq!(params.convert("nothing", text("null")), json!(null));
    }

    #[test]
    fn number_conversion_preserves_json_number_spelling_with_legacy_fallback() {
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "value": { "type": "number" }
            }
        }));

        assert_eq!(converted_number_text(&params, "5"), "5");
        assert_eq!(converted_number_text(&params, "5.0"), "5.0");
        assert_eq!(converted_number_text(&params, "5."), "5.0");
        assert_eq!(converted_number_text(&params, "+1"), "1");
        assert_eq!(converted_number_text(&params, "+1.0"), "1.0");

        // TODO: we cannot preserve the original number precision by enabling `serde_json`'s
        // `arbitrary_precision` feature, otherwise the test
        // `serialized_json_numbers_do_not_leak_serde_private_representation` will fail.
        // See issue: https://github.com/mitsuhiko/minijinja/issues/641

        // assert_eq!(converted_number_text(&params, "5.00"), "5.00");
        // assert_eq!(converted_number_text(&params, "1e0"), "1e+0");
        // assert_eq!(
        //     converted_number_text(&params, "9223372036854775807.5"),
        //     "9223372036854775807.5"
        // );
    }

    fn converted_number_text(params: &ToolSchema, value: &str) -> String {
        serde_json::to_string(&params.convert("value", text(value))).unwrap()
    }

    fn text(value: &str) -> ParamInput {
        ParamInput::Text(value.to_string())
    }

    fn elem(name: &str, value: ParamInput) -> ParamElement {
        ParamElement {
            name: name.to_string(),
            value,
        }
    }

    fn elements(elements: Vec<ParamElement>) -> ParamInput {
        ParamInput::Elements(elements)
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

        assert_eq!(params.convert("s", text("x")), json!("x"));
        assert_eq!(params.convert("i", text("7")), json!(7));
        assert_eq!(params.convert("n", text("7.5")), json!(7.5));
        assert_eq!(params.convert("b", text("true")), json!(true));
        assert_eq!(params.convert("a", text("[1]")), json!([1]));
        assert_eq!(params.convert("o", text(r#"{"x":1}"#)), json!({ "x": 1 }));
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

        assert_eq!(integer_first.convert("value", text("42")), json!(42));
        assert_eq!(string_first.convert("value", text("42")), json!("42"));
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

        assert_eq!(params.convert("choice", text("42")), json!(42));
        assert_eq!(
            params.convert("fallback_object", text(r#"{"x":1}"#)),
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

        assert_eq!(params.convert("choice", text("a")), json!("a"));
        assert_eq!(params.convert("items", text("[1,2]")), json!([1, 2]));
        assert_eq!(
            params.convert("payload", text(r#"{"x":1}"#)),
            json!({ "x": 1 })
        );
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
    fn string_param_preserves_literal_null_text() {
        // A `string`-typed param whose value is the literal text "null"/"NULL"
        // must stay a string (the original case is preserved), rather than being
        // coerced to JSON null. Non-string types keep coercing "null" to null.
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" },
                "anything": {}
            }
        }));

        assert_eq!(params.convert("name", text("null")), json!("null"));
        assert_eq!(params.convert("name", text("NULL")), json!("NULL"));
        // Non-string and schema-less params are unchanged: "null" -> null.
        assert_eq!(params.convert("count", text("null")), json!(null));
        assert_eq!(params.convert("anything", text("null")), json!(null));
    }

    #[test]
    fn nullable_enum_param_coerces_literal_null() {
        // An enum that includes `null` admits a null value, so a literal "null"
        // must coerce to JSON null (matching Python's `extract_types_from_schema`,
        // which infers `null` from the enum values), while a non-null enum keeps
        // "null" as a string.
        let params = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "mode": { "enum": [null, "auto"] },
                "color": { "enum": ["red", "green"] }
            }
        }));

        assert_eq!(params.convert("mode", text("null")), json!(null));
        assert_eq!(params.convert("mode", text("auto")), json!("auto"));
        assert_eq!(params.convert("color", text("null")), json!("null"));
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

    #[test]
    fn converts_structured_inputs_with_recursive_schema() {
        let schemas = ToolSchemas::from_tools(&[test_tool(
            "create_order",
            json!({
                "type": "object",
                "properties": {
                    "user_id": { "type": "integer" },
                    "urgent": { "type": "boolean" },
                    "note": { "type": "string" },
                    "nil": { "type": "string" },
                    "shipping": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "zip": { "type": "integer" }
                        }
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": { "type": "string" },
                                "qty": { "type": "integer" }
                            }
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": { "type": "integer" }
                    },
                    "duplicate_demo": {
                        "type": "object",
                        "properties": {
                            "tag": { "type": "string" }
                        }
                    },
                    "schema_mismatch_array": {
                        "type": "array",
                        "items": { "type": "integer" }
                    },
                    "closed_object": {
                        "type": "object",
                        "additionalProperties": false
                    },
                    "open_object": {
                        "type": "object",
                        "additionalProperties": true
                    },
                    "payload_text": { "type": "object" },
                    "items_text": { "type": "array" }
                }
            }),
        )]);

        let converted = schemas.convert_params_with_schema(
            "create_order",
            vec![
                ("user_id".to_string(), text("42")),
                ("urgent".to_string(), text("true")),
                ("note".to_string(), text("Please leave at front desk.")),
                ("nil".to_string(), text("NULL")),
                (
                    "shipping".to_string(),
                    elements(vec![
                        elem("city", text("Singapore")),
                        elem("zip", text("018956")),
                    ]),
                ),
                (
                    "items".to_string(),
                    elements(vec![
                        elem(
                            "item1",
                            elements(vec![elem("sku", text("book-001")), elem("qty", text("2"))]),
                        ),
                        elem(
                            "item2",
                            elements(vec![elem("sku", text("pen-007")), elem("qty", text("5"))]),
                        ),
                    ]),
                ),
                (
                    "metadata".to_string(),
                    elements(vec![elem("score", text("42")), elem("rank", text("7"))]),
                ),
                (
                    "duplicate_demo".to_string(),
                    elements(vec![elem("tag", text("a")), elem("tag", text("b"))]),
                ),
                (
                    "closed_object".to_string(),
                    elements(vec![elem("unknown", text("x"))]),
                ),
                (
                    "open_object".to_string(),
                    elements(vec![elem("unknown", text("y"))]),
                ),
                ("payload_text".to_string(), text(r#"{"x":1}"#)),
                ("items_text".to_string(), text("[1,2]")),
                (
                    "unknown_struct".to_string(),
                    elements(vec![
                        elem("a", text("1")),
                        elem("a", text("2")),
                        elem("nil", text("null")),
                    ]),
                ),
            ],
        );

        assert_eq!(
            Value::Object(converted),
            json!({
                "user_id": 42,
                "urgent": true,
                "note": "Please leave at front desk.",
                "nil": "NULL",
                "shipping": {
                    "city": "Singapore",
                    "zip": 18956
                },
                "items": [
                    {
                        "sku": "book-001",
                        "qty": 2
                    },
                    {
                        "sku": "pen-007",
                        "qty": 5
                    }
                ],
                "metadata": {
                    "score": 42,
                    "rank": 7
                },
                "duplicate_demo": {
                    "tag": ["a", "b"]
                },
                "closed_object": {
                    "unknown": "x"
                },
                "open_object": {
                    "unknown": "y"
                },
                "payload_text": {
                    "x": 1
                },
                "items_text": [1, 2],
                "unknown_struct": {
                    "a": ["1", "2"],
                    "nil": null
                }
            })
        );
    }
}
