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
    // Coerce the literal text `null` to JSON null, except when a `string` type is
    // admissible, where it must stay the string "null": a model emitting the
    // literal text "null" for a string field means the string, not a missing
    // value. This covers a bare `string` type and any union containing one,
    // mirroring Python's precedence where a `string` member resolves "null" to
    // the string before the null fallback is reached.
    if let ParamInput::Text(value) = input
        && value.eq_ignore_ascii_case("null")
        && !param_type.is_some_and(admits_string)
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
        JsonParamType::OneOf(types) => one_of_by_precedence(types)
            .into_iter()
            .find_map(|param_type| try_convert_text_value(param_type, value)),
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
        // Structured element trees have no Python counterpart: Python's fixed
        // precedence applies only to raw *string* coercion
        // (`coerce_to_schema_type` takes a `str`), not to these nested inputs.
        // Keep schema declaration order here so an array-or-object union selects
        // the member the schema declares first, rather than always matching the
        // unconditionally-succeeding `Object` branch.
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

/// Whether a `string` value is admissible for this type (a bare `string` or a
/// union containing one). Used to keep the literal text "null" as a string when
/// the schema allows strings, matching Python's coercion precedence.
fn admits_string(param_type: &JsonParamType) -> bool {
    match param_type {
        JsonParamType::String => true,
        JsonParamType::OneOf(types) => types.iter().any(admits_string),
        _ => false,
    }
}

/// Order union member types by Python's fixed coercion precedence.
///
/// Python's `extract_types_from_schema` recursively flattens nested
/// `anyOf`/`oneOf`/`allOf` into a flat set of type names, and
/// `coerce_to_schema_type` then tries them in a fixed priority order
/// (`null > integer > number > boolean > object > array > string`) rather than
/// schema declaration order, returning the first successful coercion. We mirror
/// both steps here: nested unions are flattened before ranking, so a schema like
/// `anyOf: [{"type":"string"}, {"type":["integer","null"]}]` coerces `"42"` to
/// the integer `42` and `"null"` to JSON null (the nested `integer`/`null`
/// members compete at their own precedence instead of losing to the sibling
/// `string`). Ties keep their original relative order via the stable sort.
///
/// Source: `vllm/tool_parsers/utils.py::{extract_types_from_schema,
/// coerce_to_schema_type}`.
fn one_of_by_precedence(types: &[JsonParamType]) -> Vec<&JsonParamType> {
    let mut ordered: Vec<&JsonParamType> = Vec::new();
    flatten_union_members(types, &mut ordered);
    ordered.sort_by_key(|param_type| precedence_rank(param_type));
    ordered
}

/// Collect union members, recursively flattening nested `OneOf` unions so every
/// leaf type is ranked directly (mirrors Python's recursive
/// `extract_types_from_schema`).
fn flatten_union_members<'a>(types: &'a [JsonParamType], out: &mut Vec<&'a JsonParamType>) {
    for param_type in types {
        match param_type {
            JsonParamType::OneOf(nested) => flatten_union_members(nested, out),
            other => out.push(other),
        }
    }
}

/// Rank one union member by Python's fixed coercion precedence (lower wins).
/// Nested unions are removed by [`flatten_union_members`] before ranking, so
/// `OneOf` never reaches this function.
fn precedence_rank(param_type: &JsonParamType) -> u8 {
    match param_type {
        JsonParamType::Null => 0,
        JsonParamType::Integer => 1,
        JsonParamType::Number => 2,
        JsonParamType::Boolean => 3,
        JsonParamType::Object { .. } => 4,
        JsonParamType::Array { .. } => 5,
        JsonParamType::String => 6,
        // Flattened out before ranking; rank last as a defensive fallback so a
        // stray nested union still yields deterministic output.
        JsonParamType::OneOf(_) => 7,
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
    fn union_type_resolution_uses_fixed_precedence() {
        // Union members are tried in Python's fixed precedence order
        // (null > integer > number > boolean > object > array > string),
        // not schema declaration order, so both orderings coerce "42" to the
        // integer 42 (matching `coerce_to_schema_type` in utils.py).
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
        assert_eq!(string_first.convert("value", text("42")), json!(42));
    }

    #[test]
    fn union_type_precedence_prefers_number_over_string_and_bool_over_string() {
        // A value that could be several types resolves to the highest-precedence
        // member present, regardless of declaration order.
        let number_or_string = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": { "value": { "type": ["string", "number"] } }
        }));
        let bool_or_string = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": { "value": { "type": ["string", "boolean"] } }
        }));

        // "1" is integer-parseable -> number wins over string.
        assert_eq!(number_or_string.convert("value", text("1.5")), json!(1.5));
        // "true" is not a number but is a boolean -> boolean wins over string.
        assert_eq!(bool_or_string.convert("value", text("true")), json!(true));
        // A value that matches no higher-precedence member falls back to string.
        assert_eq!(
            bool_or_string.convert("value", text("maybe")),
            json!("maybe")
        );
    }

    #[test]
    fn nested_union_flattens_before_precedence() {
        // A nested union (`anyOf` containing a `["integer","null"]` member next
        // to a sibling `string`) must be flattened before applying precedence,
        // exactly like Python's recursive `extract_types_from_schema`. Without
        // flattening the sibling `string` would swallow "42"/"null" first.
        // Pinned explicitly (not only via the differential oracle) so the
        // semantics survive oracle drift. Live Python `coerce_to_schema_type`
        // on the flattened `{integer,null,string}` set yields:
        //   "42" -> 42, "null" -> null, "3.14" -> "3.14" (no number member).
        let nested = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": ["integer", "null"] }
                    ]
                }
            }
        }));

        assert_eq!(nested.convert("value", text("42")), json!(42));
        assert_eq!(nested.convert("value", text("null")), json!(null));
        // No `number` member is admissible, so a float stays a string.
        assert_eq!(nested.convert("value", text("3.14")), json!("3.14"));
        // A plain word matches no higher-precedence member -> string fallback.
        assert_eq!(nested.convert("value", text("hello")), json!("hello"));
    }

    #[test]
    fn string_or_null_union_coerces_literal_null() {
        // For a flat `["string","null"]` union, `null` (precedence 0) wins over
        // `string` (precedence 6), so the literal text "null" coerces to JSON
        // null, while a value that matches no non-string member (there is no
        // integer member here) stays the string. Matches live Python:
        //   ["string","null"] on "null" -> None, on "42" -> "42".
        let string_or_null = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": { "value": { "type": ["string", "null"] } }
        }));

        assert_eq!(string_or_null.convert("value", text("null")), json!(null));
        assert_eq!(string_or_null.convert("value", text("NULL")), json!(null));
        assert_eq!(string_or_null.convert("value", text("42")), json!("42"));
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
    fn structured_union_input_keeps_declaration_order() {
        // Fixed coercion precedence is a raw-*string* concern (Python's
        // `coerce_to_schema_type` takes a `str`); it must NOT reorder structured
        // element trees. For an `array`-or-`object` union whose `array` member is
        // declared first, structured (nested-XML) input must select the array,
        // not the unconditionally-succeeding object branch that precedence
        // ordering (object=4 < array=5) would otherwise pick first.
        let array_first = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "array", "items": { "type": "integer" } },
                        { "type": "object" }
                    ]
                }
            }
        }));
        let input = elements(vec![elem("a", text("1")), elem("b", text("2"))]);
        assert_eq!(array_first.convert("value", input), json!([1, 2]));

        // Object declared first keeps object shape.
        let object_first = ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "object" },
                        { "type": "array", "items": { "type": "integer" } }
                    ]
                }
            }
        }));
        // The `object` member is untyped (no `properties`), so its children are
        // coerced without a schema and stay strings — the point here is the
        // object *shape* is selected, not the array.
        let input = elements(vec![elem("a", text("1")), elem("b", text("2"))]);
        assert_eq!(
            object_first.convert("value", input),
            json!({ "a": "1", "b": "2" })
        );
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

/// Differential parity tests for union-type coercion against the Python oracle.
///
/// Issue: <https://github.com/vllm-project/vllm/issues/47179> reported that the
/// Rust parser resolved union (`anyOf`/`["a","b"]`) types by schema declaration
/// order, while Python's `coerce_to_schema_type` uses a fixed precedence. Rather
/// than hand-curating cases, this suite ports Python's coercion into a Rust
/// reference oracle and fuzzes `(value, types)` pairs with `proptest`, asserting
/// the Rust schema conversion agrees with the oracle for every generated case.
///
/// Scope: this pins union member *selection* (which precedence wins), the axis
/// the issue is about. It deliberately excludes number-spelling normalization
/// (Python's number branch collapses `"1.0"` to `1`, while Rust preserves the
/// JSON spelling as `1.0` on purpose; see
/// `number_conversion_preserves_json_number_spelling_with_legacy_fallback`).
/// Generated values therefore never have an integer-valued decimal spelling, so
/// the two implementations are expected to agree exactly.
#[cfg(test)]
mod proptest_differential {
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use super::{ParamInput, ToolSchema};

    /// Reference port of `vllm/tool_parsers/utils.py::coerce_to_schema_type`.
    ///
    /// Tries each type in the fixed priority order
    /// (`null > integer > number > boolean > object > array > string`) and
    /// returns the first successful coercion, falling back to a final
    /// best-effort `json` parse and then the raw string. Kept structurally
    /// close to the Python source so divergences are easy to audit.
    fn python_coerce(value: &str, types: &[&str]) -> Value {
        const PRIORITY: [&str; 7] = [
            "null", "integer", "number", "boolean", "object", "array", "string",
        ];

        for candidate in PRIORITY {
            if !types.contains(&candidate) {
                continue;
            }
            match candidate {
                "null" if value.eq_ignore_ascii_case("null") => return Value::Null,
                "string" => return Value::String(value.to_string()),
                "integer" => {
                    if let Some(v) = parse_python_int(value) {
                        return json!(v);
                    }
                }
                "number" => {
                    if let Ok(v) = value.parse::<f64>()
                        && v.is_finite()
                    {
                        // Python: `val if val != int(val) else int(val)`.
                        if v.fract() == 0.0 && v.abs() < i64::MAX as f64 {
                            return json!(v as i64);
                        }
                        return json!(v);
                    }
                }
                "boolean" => {
                    let lower = value.trim().to_ascii_lowercase();
                    if lower == "true" || lower == "1" {
                        return Value::Bool(true);
                    }
                    if lower == "false" || lower == "0" {
                        return Value::Bool(false);
                    }
                }
                "object" | "array" => {
                    if let Ok(parsed) = serde_json::from_str::<Value>(value)
                        && is_json_finite(&parsed)
                    {
                        return parsed;
                    }
                }
                _ => {}
            }
        }

        // Final fallback: best-effort JSON parse, else raw string.
        match serde_json::from_str::<Value>(value) {
            Ok(parsed) if is_json_finite(&parsed) => parsed,
            _ => Value::String(value.to_string()),
        }
    }

    /// Mirror Python's `int(value)`: base-10 integer, no fractional part.
    fn parse_python_int(value: &str) -> Option<i64> {
        value.trim().parse::<i64>().ok()
    }

    /// Mirror Python's `_is_json_finite`: reject non-finite floats anywhere.
    fn is_json_finite(value: &Value) -> bool {
        match value {
            Value::Number(n) => n.as_f64().is_none_or(f64::is_finite),
            Value::Array(items) => items.iter().all(is_json_finite),
            Value::Object(map) => map.values().all(is_json_finite),
            _ => true,
        }
    }

    /// Build a one-property schema whose value is the given union of types.
    fn union_schema(types: &[&str]) -> ToolSchema {
        ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": { "value": { "type": types } }
        }))
    }

    /// Build a one-property schema whose value nests the first two members inside
    /// an `anyOf`, e.g. `["string","integer","null"]` becomes
    /// `anyOf: [{"type":["string","integer"]}, {"type":"null"}]`. Python's
    /// `extract_types_from_schema` flattens this back to the same type set, so
    /// the coercion result must match the flat schema (and the oracle). This
    /// exercises the nested-union flattening the flat generator cannot reach.
    fn nested_union_schema(types: &[&str]) -> ToolSchema {
        let value = if types.len() >= 2 {
            let (head, tail) = types.split_at(2);
            let mut members = vec![json!({ "type": head })];
            members.extend(tail.iter().map(|t| json!({ "type": t })));
            json!({ "anyOf": members })
        } else {
            json!({ "type": types })
        };
        ToolSchema::from_schema(&json!({
            "type": "object",
            "properties": { "value": value }
        }))
    }

    /// The primitive schema types the Rust converter understands and Python
    /// ranks. (`object`/`array` are covered separately via JSON-shaped values.)
    fn arb_type() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("null"),
            Just("integer"),
            Just("number"),
            Just("boolean"),
            Just("string"),
            Just("object"),
            Just("array"),
        ]
    }

    /// A union that always admits `string`, plus 0-3 other randomly ordered
    /// members. Including `string` guarantees Python resolves via a declared
    /// type (string returns the raw value as a last resort) rather than its
    /// unconditional `json.loads` fallback, so generated cases are not rejected
    /// while still exercising whether a higher-precedence member wins over
    /// `string`. The `string` position is randomized to confirm precedence does
    /// not depend on declaration order.
    fn arb_types() -> impl Strategy<Value = Vec<&'static str>> {
        (prop::collection::vec(arb_type(), 0..=3), 0usize..=3).prop_map(|(mut types, at)| {
            let at = at.min(types.len());
            types.insert(at, "string");
            types
        })
    }

    /// Values spanning each coercion branch, plus adversarial tokens. No value
    /// has an integer-valued decimal spelling (see module scope note).
    fn arb_value() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("null".to_string()),
            Just("NULL".to_string()),
            Just("42".to_string()),
            Just("-7".to_string()),
            Just("0".to_string()),
            Just("1".to_string()),
            Just("3.5".to_string()),
            Just("-0.25".to_string()),
            Just("true".to_string()),
            Just("false".to_string()),
            Just("1e999".to_string()),
            Just("[1,2]".to_string()),
            Just(r#"{"k":1}"#.to_string()),
            Just("hello".to_string()),
            // NB: the empty string is excluded — Rust has an `is_empty()`
            // shortcut that coerces `""` to `{}`/`[]` for object/array members
            // with no Python equivalent. That is a separate divergence from the
            // union-ordering fix this suite pins (tracked in #47179).
            "[a-z]{1,8}".prop_map(|s| s),
            "-?[1-9][0-9]{0,5}".prop_map(|s| s),
        ]
    }

    /// Whether Python resolves `(value, types)` via a declared-type branch
    /// rather than its unconditional final `json.loads` fallback.
    ///
    /// Python's `coerce_to_schema_type` ends with a fallback that JSON-parses
    /// the value regardless of the declared types, so e.g. `("[1,2]", ["null"])`
    /// yields `[1, 2]`. Rust instead falls back to the raw string there. That
    /// fallback gap is a *separate* divergence from the union member *selection*
    /// this suite pins, so we only assert parity when a typed branch decides the
    /// result (where both implementations are expected to agree).
    fn decided_by_typed_branch(value: &str, types: &[&str]) -> bool {
        const PRIORITY: [&str; 7] = [
            "null", "integer", "number", "boolean", "object", "array", "string",
        ];
        for candidate in PRIORITY {
            if !types.contains(&candidate) {
                continue;
            }
            let decided = match candidate {
                "null" => value.eq_ignore_ascii_case("null"),
                "string" => true,
                "integer" => parse_python_int(value).is_some(),
                "number" => value.parse::<f64>().is_ok_and(f64::is_finite),
                "boolean" => {
                    let lower = value.trim().to_ascii_lowercase();
                    matches!(lower.as_str(), "true" | "1" | "false" | "0")
                }
                "object" | "array" => serde_json::from_str::<Value>(value)
                    .ok()
                    .is_some_and(|parsed| is_json_finite(&parsed)),
                _ => false,
            };
            if decided {
                return true;
            }
        }
        false
    }

    proptest! {
        /// Rust union coercion must equal the Python oracle for every generated
        /// `(value, types)` pair whose result a declared type decides. This is
        /// the parity guard for the union member-selection fix in issue #47179.
        #[test]
        fn union_coercion_matches_python_oracle(
            value in arb_value(),
            types in arb_types(),
        ) {
            prop_assume!(decided_by_typed_branch(&value, &types));

            let python = python_coerce(&value, &types);

            // Flat schema (`type: [..]`).
            let flat = union_schema(&types).convert("value", ParamInput::Text(value.clone()));
            prop_assert_eq!(
                &flat,
                &python,
                "flat union coercion diverged for value={:?} types={:?}",
                value,
                types
            );

            // Nested schema (`anyOf` wrapping some members). Python flattens
            // nesting, so the result must equal the flat/oracle result — this is
            // the regression guard for the nested-union flattening fix.
            let nested =
                nested_union_schema(&types).convert("value", ParamInput::Text(value.clone()));
            prop_assert_eq!(
                nested,
                python,
                "nested union coercion diverged for value={:?} types={:?}",
                value,
                types
            );
        }
    }

    /// Golden cases copied from Python's actual output, guarding the reference
    /// port itself against drift from `coerce_to_schema_type`.
    #[test]
    fn python_oracle_golden_cases() {
        assert_eq!(python_coerce("42", &["string", "integer"]), json!(42));
        assert_eq!(python_coerce("42", &["integer", "string"]), json!(42));
        assert_eq!(python_coerce("3.5", &["string", "number"]), json!(3.5));
        assert_eq!(python_coerce("true", &["string", "boolean"]), json!(true));
        assert_eq!(python_coerce("1", &["boolean", "integer"]), json!(1));
        assert_eq!(python_coerce("null", &["string", "null"]), json!(null));
        assert_eq!(python_coerce("null", &["string"]), json!("null"));
        assert_eq!(python_coerce("[1,2]", &["string", "array"]), json!([1, 2]));
        assert_eq!(
            python_coerce("maybe", &["boolean", "string"]),
            json!("maybe")
        );
    }
}
