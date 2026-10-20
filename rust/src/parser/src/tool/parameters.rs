// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde_json::{Map, Number, Value};

use crate::schema::{JsonType, OptionSource, SchemaRoot};
use crate::tool::Tool;

/// Normalized parameter schemas for all tools in one request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ToolSchemas {
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
    /// The tool's parameters schema, read one level at a time as values are
    /// converted, so recursive schemas cost only as deep as the value.
    parameters: Value,
}

/// Parameter input for schema-aware conversion.
///
/// It can be either a raw text string, or a structured input with named child elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParamInput {
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
pub(crate) struct ParamElement {
    pub name: String,
    pub value: ParamInput,
}

/// One level of a parameter's normalized type, used for raw string coercion.
///
/// Object and array types normalize their children only when a structured
/// value reaches them.
#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonParamType<'a> {
    String,
    Integer,
    Number,
    Boolean,
    Object(ObjectType<'a>),
    Array(ArrayType<'a>),
    Null,
    /// A non-string `const` or `enum` value: text spelling it as JSON decodes
    /// to it.
    Literal(&'a Value),
    OneOf(Vec<JsonParamType<'a>>),
}

/// An object type whose field types are normalized on access.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectType<'a> {
    root: SchemaRoot<'a>,
    properties: Option<&'a Map<String, Value>>,
    additional_properties: Option<&'a Value>,
}

/// An array type whose item type is normalized on access.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ArrayType<'a> {
    root: SchemaRoot<'a>,
    items: Option<&'a Value>,
}

impl ToolSchemas {
    /// Normalize OpenAI-style tool parameter JSON schemas for one request.
    pub(crate) fn from_tools(tools: &[Tool]) -> Self {
        let tools = tools
            .iter()
            .map(|tool| {
                (
                    tool.name.clone(),
                    ToolSchema::from_schema(tool.parameters.clone()),
                )
            })
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
    pub(crate) fn convert_param_with_schema<P>(
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
            parameters: Value::Null,
        };
        &EMPTY
    }

    /// Keep an OpenAI-style tool parameters JSON schema.
    fn from_schema(parameters: Value) -> Self {
        Self { parameters }
    }

    /// Convert one parameter value using its normalized schema type.
    ///
    /// If the parameter name is unknown, or we don't have a schema for it, or
    /// the value fails to convert, this falls back to returning the raw
    /// string as a JSON string value, or object-like JSON for structured input.
    fn convert(&self, name: &str, input: ParamInput) -> Value {
        let root = SchemaRoot::new(&self.parameters);
        let param_type = root
            .resolved()
            .get("properties")
            .and_then(|properties| properties.get(name))
            .and_then(|schema| JsonParamType::from_schema(&root, schema));
        convert_with_optional_schema(param_type.as_ref(), &input)
    }
}

impl<'a> JsonParamType<'a> {
    /// Normalize one parameter or nested value schema, or `None` when it does
    /// not constrain the value's type.
    fn from_schema(root: &SchemaRoot<'a>, schema: &'a Value) -> Option<Self> {
        let options = root.options(schema);
        if options.iter().all(|option| matches!(option.source, OptionSource::Any)) {
            return None;
        }
        let mut types = Vec::new();
        for option in &options {
            // String literals stay with the string type, which keeps text
            // verbatim instead of unquoting text that spells one as JSON.
            if let OptionSource::Literal(value) = option.source
                && !value.is_string()
                && !types.contains(&Self::Literal(value))
            {
                types.push(Self::Literal(value));
            }
            let schema = option.schema();
            let param_type = match option.ty {
                JsonType::String => Self::String,
                JsonType::Integer => Self::Integer,
                JsonType::Number => Self::Number,
                JsonType::Boolean => Self::Boolean,
                JsonType::Null => Self::Null,
                JsonType::Object => Self::Object(ObjectType {
                    root: *root,
                    properties: schema
                        .and_then(|schema| schema.get("properties"))
                        .and_then(Value::as_object),
                    additional_properties: schema
                        .and_then(|schema| schema.get("additionalProperties"))
                        .filter(|schema| schema.is_object()),
                }),
                JsonType::Array => Self::Array(ArrayType {
                    root: *root,
                    items: schema.and_then(|schema| schema.get("items")),
                }),
            };
            if !types.contains(&param_type) {
                types.push(param_type);
            }
        }
        // Text spelling a literal decodes to it before any type, a string in
        // particular, accepts the same text.
        types.sort_by_key(|param_type| !matches!(param_type, Self::Literal(_)));
        match types.len() {
            0 => None,
            1 => types.pop(),
            _ => Some(Self::OneOf(types)),
        }
    }
}

impl<'a> ObjectType<'a> {
    /// The type of the field named `name`, from its declared schema or the
    /// additional properties' schema.
    fn field(&self, name: &str) -> Option<JsonParamType<'a>> {
        let schema = self
            .properties
            .and_then(|properties| properties.get(name))
            .or(self.additional_properties)?;
        JsonParamType::from_schema(&self.root, schema)
    }
}

impl<'a> ArrayType<'a> {
    /// The type of the array's items.
    fn items(&self) -> Option<JsonParamType<'a>> {
        JsonParamType::from_schema(&self.root, self.items?)
    }
}

/// Recognize JSON and Python null spellings emitted by chat templates,
/// ignoring surrounding whitespace.
fn is_null_literal(value: &str) -> bool {
    let value = value.trim();
    value.eq_ignore_ascii_case("null") || value.eq_ignore_ascii_case("none")
}

/// Convert one parameter input to a normalized JSON value.
fn convert_with_optional_schema(
    param_type: Option<&JsonParamType<'_>>,
    input: &ParamInput,
) -> Value {
    // Coerce the literal text `null` to JSON null, except for `string`-typed
    // params, where it must stay the string "null": a model emitting the literal
    // text "null" for a string field means the string, not a missing value.
    // Python-style `None` follows the same schema coercion rules.
    if let ParamInput::Text(value) = input
        && is_null_literal(value)
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
            Value::Object(convert_elements_to_object(elements, None))
        }
    }
}

/// Convert one parameter input to a normalized JSON type.
fn try_convert_value(param_type: &JsonParamType<'_>, input: &ParamInput) -> Option<Value> {
    match input {
        ParamInput::Text(value) => try_convert_text_value(param_type, value),
        ParamInput::Elements(elements) => try_convert_elements_value(param_type, elements),
    }
}

/// Convert one raw string value to a normalized JSON type.
///
/// Only `string` keeps the value verbatim; the other types ignore surrounding
/// whitespace.
fn try_convert_text_value(param_type: &JsonParamType<'_>, value: &str) -> Option<Value> {
    match param_type {
        JsonParamType::String => Some(Value::String(value.to_string())),
        JsonParamType::Integer => convert_integer_text(value),
        JsonParamType::Number => convert_number_text(value),
        JsonParamType::Boolean => try_convert_boolean(value),
        JsonParamType::Object(_) if value.trim().is_empty() => Some(Value::Object(Map::new())),
        JsonParamType::Array(_) if value.trim().is_empty() => Some(Value::Array(Vec::new())),
        // For composite types with string input, interpret the string as JSON of that type.
        JsonParamType::Object(_) => serde_json::from_str(value).ok().filter(Value::is_object),
        JsonParamType::Array(_) => serde_json::from_str(value).ok().filter(Value::is_array),
        JsonParamType::Null => is_null_literal(value).then_some(Value::Null),
        JsonParamType::Literal(literal) => serde_json::from_str::<Value>(value.trim())
            .ok()
            .filter(|value| value == *literal),
        JsonParamType::OneOf(types) => {
            types.iter().find_map(|param_type| try_convert_text_value(param_type, value))
        }
    }
}

/// Convert one structured parameter input to a normalized JSON type.
fn try_convert_elements_value(
    param_type: &JsonParamType<'_>,
    elements: &[ParamElement],
) -> Option<Value> {
    match param_type {
        JsonParamType::Object(object) => Some(Value::Object(convert_elements_to_object(
            elements,
            Some(object),
        ))),
        JsonParamType::Array(array) => {
            let items = array.items();
            Some(Value::Array(
                // Collect all child elements into an array, regardless of their names.
                elements
                    .iter()
                    .map(|element| convert_with_optional_schema(items.as_ref(), &element.value))
                    .collect(),
            ))
        }
        JsonParamType::OneOf(types) => types
            .iter()
            .find_map(|param_type| try_convert_elements_value(param_type, elements)),

        // Primitive types can't be converted from structured input.
        JsonParamType::String
        | JsonParamType::Integer
        | JsonParamType::Number
        | JsonParamType::Boolean
        | JsonParamType::Null
        | JsonParamType::Literal(_) => None,
    }
}

/// Convert structured elements to an object, using field schemas when present.
fn convert_elements_to_object(
    elements: &[ParamElement],
    object_type: Option<&ObjectType<'_>>,
) -> Map<String, Value> {
    let mut object = Map::with_capacity(elements.len());
    for element in elements {
        let param_type = object_type.and_then(|object_type| object_type.field(&element.name));
        let value = convert_with_optional_schema(param_type.as_ref(), &element.value);
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

/// Remove single underscores between digits (`1_000`), as Python numeric
/// literals allow; any other underscore makes the value invalid.
fn without_digit_separators(value: &str) -> Option<std::borrow::Cow<'_, str>> {
    if !value.contains('_') {
        return Some(value.into());
    }
    let bytes = value.as_bytes();
    let valid = bytes.iter().enumerate().all(|(index, byte)| {
        *byte != b'_'
            || (index > 0
                && bytes[index - 1].is_ascii_digit()
                && bytes.get(index + 1).is_some_and(u8::is_ascii_digit))
    });
    valid.then(|| value.replace('_', "").into())
}

/// Convert raw text to a JSON integer, as for an `integer` parameter.
pub(crate) fn convert_integer_text(value: &str) -> Option<Value> {
    without_digit_separators(value.trim())?
        .parse::<i64>()
        .ok()
        .map(Number::from)
        .map(Value::Number)
}

/// Convert raw text to a JSON number, as for a `number` parameter.
pub(crate) fn convert_number_text(value: &str) -> Option<Value> {
    try_convert_number(&without_digit_separators(value.trim())?)
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
            defer_loading: None,
        }
    }

    #[test]
    fn invalid_schema_converts_everything_as_string() {
        let params = ToolSchema::from_schema(json!({ "type": "object" }));

        assert_eq!(params.convert("count", text("42")), json!("42"));
        assert_eq!(params.convert("count", text("null")), json!(null));
    }

    #[test]
    fn skips_unknown_property_schema_and_unknown_type() {
        let params = ToolSchema::from_schema(json!({
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
        let params = ToolSchema::from_schema(json!({
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
        let params = ToolSchema::from_schema(json!({
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
    fn non_string_values_ignore_surrounding_whitespace() {
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "count": { "type": "integer" },
                "size": { "type": "number" },
                "enabled": { "type": "boolean" },
                "payload": { "type": "object" },
                "items": { "type": "array" }
            }
        }));

        assert_eq!(params.convert("text", text(" 7 ")), json!(" 7 "));
        assert_eq!(params.convert("count", text(" 7\n")), json!(7));
        assert_eq!(params.convert("size", text("\t2.5 ")), json!(2.5));
        assert_eq!(params.convert("enabled", text(" false ")), json!(false));
        assert_eq!(params.convert("count", text(" None ")), json!(null));
        assert_eq!(params.convert("payload", text(" \n")), json!({}));
        assert_eq!(params.convert("items", text(" [1] ")), json!([1]));
    }

    #[test]
    fn numbers_accept_digit_separators() {
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "count": { "type": "integer" },
                "size": { "type": "number" }
            }
        }));

        assert_eq!(params.convert("count", text("1_000")), json!(1000));
        assert_eq!(params.convert("size", text("1_000.5")), json!(1000.5));
        for invalid in ["_1", "1_", "1__0", "1_.5"] {
            assert_eq!(params.convert("size", text(invalid)), json!(invalid));
        }
    }

    #[test]
    fn nullable_schemas_accept_null() {
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "nullable": true },
                "count": { "type": ["integer", "string"], "nullable": true }
            }
        }));

        // Without `nullable`, a string parameter keeps the literal text "null".
        assert_eq!(params.convert("text", text("null")), json!(null));
        assert_eq!(params.convert("text", text("x")), json!("x"));
        assert_eq!(params.convert("count", text("None")), json!(null));
        assert_eq!(params.convert("count", text("7")), json!(7));
    }

    #[test]
    fn composite_values_must_decode_to_their_type() {
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "payload": { "type": "object" },
                "items": { "type": "array" },
                "either": { "type": ["object", "array"] }
            }
        }));

        assert_eq!(params.convert("payload", text("[1]")), json!("[1]"));
        assert_eq!(params.convert("payload", text("5")), json!("5"));
        assert_eq!(
            params.convert("items", text(r#"{"k":1}"#)),
            json!(r#"{"k":1}"#)
        );
        assert_eq!(params.convert("either", text("[1]")), json!([1]));
    }

    #[test]
    fn converts_upstream_aliases() {
        let params = ToolSchema::from_schema(json!({
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
        let integer_first = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "value": { "type": ["integer", "string"] }
            }
        }));
        let string_first = ToolSchema::from_schema(json!({
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
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "choice": {
                    "anyOf": [
                        { "type": "integer" },
                        { "type": "string" }
                    ]
                },
                "unknown_alternatives": {
                    "oneOf": [
                        { "type": "mystery" }
                    ]
                }
            }
        }));

        assert_eq!(params.convert("choice", text("42")), json!(42));
        // Alternatives of unknown types leave the value unconstrained, like an
        // unknown `type`.
        assert_eq!(
            params.convert("unknown_alternatives", text(r#"{"x":1}"#)),
            json!(r#"{"x":1}"#)
        );
    }

    #[test]
    fn resolves_references_in_pydantic_schemas() {
        let params = ToolSchema::from_schema(json!({
            "$defs": {
                "Place": {
                    "properties": { "city": { "type": "string" }, "days": { "type": "integer" } },
                    "type": "object"
                },
                "Count": { "type": "integer" }
            },
            "type": "object",
            "properties": {
                "place": { "$ref": "#/$defs/Place" },
                "backup": { "anyOf": [{ "$ref": "#/$defs/Place" }, { "type": "null" }] },
                "count": { "$ref": "#/$defs/Count" }
            }
        }));

        assert_eq!(
            params.convert("place", text(r#"{"city": "Paris", "days": 3}"#)),
            json!({ "city": "Paris", "days": 3 })
        );
        assert_eq!(
            params.convert("backup", text(r#"{"city": "Paris"}"#)),
            json!({ "city": "Paris" })
        );
        assert_eq!(params.convert("backup", text("null")), Value::Null);
        assert_eq!(params.convert("count", text("3")), json!(3));
    }

    #[test]
    fn recursive_references_convert_without_unbounded_expansion() {
        let params = ToolSchema::from_schema(json!({
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer" },
                        "children": { "type": "array", "items": { "$ref": "#/$defs/Node" } }
                    }
                }
            },
            "type": "object",
            "properties": { "tree": { "$ref": "#/$defs/Node" } }
        }));

        assert_eq!(
            params.convert(
                "tree",
                text(r#"{"id": 1, "children": [{"id": 2, "children": []}]}"#)
            ),
            json!({ "id": 1, "children": [{ "id": 2, "children": [] }] })
        );
    }

    #[test]
    fn branching_recursive_references_convert_structured_values_by_level() {
        // Two recursive children per node expand to 2^depth types if the schema
        // is walked ahead of the value.
        let params = ToolSchema::from_schema(json!({
            "$defs": {
                "Tree": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "integer" },
                        "left": { "$ref": "#/$defs/Tree" },
                        "right": { "$ref": "#/$defs/Tree" }
                    }
                }
            },
            "type": "object",
            "properties": { "tree": { "$ref": "#/$defs/Tree" } }
        }));

        let leaf = |value: &str| ParamInput::Elements(vec![elem("value", text(value))]);
        let tree = ParamInput::Elements(vec![
            elem("value", text("1")),
            elem(
                "left",
                ParamInput::Elements(vec![elem("value", text("2")), elem("right", leaf("3"))]),
            ),
            elem("right", leaf("4")),
        ]);
        assert_eq!(
            params.convert("tree", tree),
            json!({
                "value": 1,
                "left": { "value": 2, "right": { "value": 3 } },
                "right": { "value": 4 }
            })
        );
    }

    #[test]
    fn mixed_enums_decode_text_spelling_a_non_string_literal_to_it() {
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "mode": { "enum": ["fast", 1] },
                "flag": { "enum": ["auto", true] }
            }
        }));

        assert_eq!(params.convert("mode", text("1")), json!(1));
        assert_eq!(params.convert("mode", text("fast")), json!("fast"));
        // Text matching no literal keeps the schema order, which tries the
        // string first.
        assert_eq!(params.convert("mode", text("2")), json!("2"));
        assert_eq!(params.convert("flag", text("true")), json!(true));
        assert_eq!(params.convert("flag", text("auto")), json!("auto"));
    }

    #[test]
    fn integer_enums_convert_to_numbers() {
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": { "level": { "enum": [1, 2, 3] } }
        }));

        assert_eq!(params.convert("level", text("2")), json!(2));
    }

    #[test]
    fn infers_type_from_schema_shape_without_type() {
        let params = ToolSchema::from_schema(json!({
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
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer" },
                "anything": {}
            }
        }));

        for literal in ["null", "NULL", "None", "none", "NONE"] {
            assert_eq!(params.convert("name", text(literal)), json!(literal));
            // Non-string and schema-less params share the same null coercion.
            assert_eq!(params.convert("count", text(literal)), json!(null));
            assert_eq!(params.convert("anything", text(literal)), json!(null));
        }
    }

    #[test]
    fn nullable_enum_param_coerces_literal_null() {
        // An enum that includes `null` admits a null value, so a literal "null"
        // must coerce to JSON null (matching Python's `extract_types_from_schema`,
        // which infers `null` from the enum values), while a non-null enum keeps
        // "null" as a string.
        let params = ToolSchema::from_schema(json!({
            "type": "object",
            "properties": {
                "mode": { "enum": [null, "auto"] },
                "color": { "enum": ["red", "green"] }
            }
        }));

        for literal in ["null", "NULL", "None", "none", "NONE"] {
            assert_eq!(params.convert("mode", text(literal)), json!(null));
            assert_eq!(params.convert("color", text(literal)), json!(literal));
        }
        assert_eq!(params.convert("mode", text("auto")), json!("auto"));
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
