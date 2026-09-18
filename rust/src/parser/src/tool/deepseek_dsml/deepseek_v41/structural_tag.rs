// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Visible-text grammar for DeepSeek V4.1 tool calls.
//!
//! Adapted from xgrammar-structural-tag 0.2.0's DeepSeek V4 builder, with
//! V4.1's spaced DSML tags. Parameter bodies mirror xgrammar's `deepseek_xml`
//! JSON-schema conversion (`cpp/json_schema_converter_ext.cc`): that style
//! hardcodes the V4 markers in C++, so the conversion is reproduced here with
//! generic format primitives. Declared parameter names, required/optional
//! presence, and value schemas are constrained like V4 while the framing
//! keeps V4.1's markers.

use serde_json::{Map, Value};
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{
    StructuralTagBuilder, StructuralTagContext, StructuralTagOptions,
};
use xgrammar_structural_tag::format::{
    Format, JsonSchemaFormat, StructuralTag, TagFormat, TriggeredTagsFormat,
};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam, function_parameters};

use super::DsmlTokens;

const TOKENS: DsmlTokens = DsmlTokens::V41;

pub(super) struct DeepSeekV41StructuralTagBuilder;

impl StructuralTagBuilder for DeepSeekV41StructuralTagBuilder {
    fn build(&self, ctx: StructuralTagContext<'_>) -> Result<StructuralTag> {
        let tags = ctx
            .function_tools
            .iter()
            .map(|tool| call_tag(tool, ctx.options))
            .collect::<Vec<_>>();
        let excludes = if ctx.options.exclude_special_tokens {
            &["<think>", "</think>"][..]
        } else {
            &[]
        };
        let format = match ctx.tool_choice {
            BuilderToolChoice::Auto if tags.is_empty() => Format::any_text_excluding(excludes),
            BuilderToolChoice::Auto => Format::TriggeredTags(
                TriggeredTagsFormat::new(
                    &[TOKENS.tool_calls_start],
                    vec![TagFormat::new(
                        format!("{}\n", TOKENS.tool_calls_start),
                        Format::tags_with_separator(tags, "", true, false),
                        TOKENS.tool_calls_end,
                    )],
                )
                .with_excludes(excludes),
            ),
            BuilderToolChoice::Forced | BuilderToolChoice::Required => Format::sequence(vec![
                Format::const_string(format!("\n\n{}\n", TOKENS.tool_calls_start)),
                Format::tags_with_separator(
                    tags,
                    "",
                    true,
                    ctx.tool_choice == BuilderToolChoice::Forced,
                ),
                Format::const_string(TOKENS.tool_calls_end),
            ]),
        };
        // Reasoning boundaries remain owned by the engine reasoner; serving
        // calls this visible-text builder with reasoning=false.
        Ok(StructuralTag::new(format))
    }
}

fn call_tag(tool: &FunctionToolParam, options: StructuralTagOptions) -> TagFormat {
    TagFormat::new(
        format!("{} name=\"{}\">\n", TOKENS.invoke_start, tool.function.name),
        parameters_content(&function_parameters(&tool.function), options),
        format!("{}\n", TOKENS.invoke_end),
    )
}

/// Constrain one invoke block's parameter list by the tool's JSON schema,
/// mirroring `JSONSchemaConverter::GetPartialRuleForProperties` for the
/// common tool-schema case: declared properties in schema order, required
/// ones mandatory, optional ones skippable, additional properties last when
/// the schema allows them.
///
/// Divergences from the C++ conversion are all in the over-accepting
/// direction so valid model output is never masked out:
/// - `minProperties`/`maxProperties` counts and
///   `patternProperties`/`propertyNames` key rules are not enforced.
/// - Additional property names accept any `[^"]+`; C++ restricts them to
///   identifier characters.
/// - A `false` property schema is treated as unconstrained instead of making
///   the tool uncallable.
/// - The string `format` keyword is ignored, and length bounds use a
///   marker-safe character class instead of C++'s any-character repeat.
/// - Whitespace between parameters is fixed to the trained `\n` framing
///   instead of arbitrary whitespace.
fn parameters_content(parameters: &Value, options: StructuralTagOptions) -> Format {
    let Some(schema) = parameters.as_object() else {
        // A `false` schema accepts no arguments at all; anything else (such
        // as `true` from a strict=false tool) leaves parameters
        // unconstrained.
        return if parameters == &Value::Bool(false) {
            Format::const_string("")
        } else {
            Format::star(permissive_parameter(&Value::Bool(true), options))
        };
    };
    if matches!(schema.get("type"), Some(Value::String(kind)) if kind != "object") {
        // Tool parameters are objects in practice; keep other roots
        // permissive.
        return Format::star(permissive_parameter(&Value::Bool(true), options));
    }
    let extras = additional_properties_schema(schema);
    let Some(properties) = schema
        .get("properties")
        .and_then(Value::as_object)
        .filter(|properties| !properties.is_empty())
    else {
        return extra_parameters_format(extras, options);
    };
    if properties.keys().any(|key| key.contains('"')) {
        // Tag begins are literal strings, so a quoted parameter name cannot
        // be expressed; keep the whole invoke body permissive instead.
        return Format::star(permissive_parameter(&Value::Bool(true), options));
    }

    let root_defs = root_definitions(schema);
    let required = required_properties(schema);

    if options.any_order {
        // Mirror GetAnyOrderRuleForProperties: only the parameter count is
        // constrained, not which keys appear, and duplicates are accepted.
        let mut items = properties
            .iter()
            .map(|(key, schema)| Format::Tag(parameter_tag(key, schema, &root_defs, options)))
            .collect::<Vec<_>>();
        if let Some(extras) = &extras {
            items.push(permissive_parameter(extras, options));
        }
        let item = match items.as_slice() {
            [item] => item.clone(),
            _ => Format::or(items),
        };
        return Format::repeat(item, required.len() as i64, -1);
    }

    let mut elements = Vec::new();
    for (key, schema) in properties {
        let tag = Format::Tag(parameter_tag(key, schema, &root_defs, options));
        elements.push(if required.contains(key.as_str()) {
            tag
        } else {
            Format::optional(tag)
        });
    }
    if let Some(extras) = extras {
        elements.push(Format::star(permissive_parameter(&extras, options)));
    }
    Format::sequence(elements)
}

/// The value schema for undeclared parameters, or None when the schema
/// forbids them. The structural-tag backend compiles schemas with xgrammar
/// `strict_mode=true`, so objects are closed unless `additionalProperties`
/// (or `unevaluatedProperties`) explicitly allows more.
///
/// `patternProperties`/`propertyNames` are treated as allowing additional
/// parameters: constraining them needs xgrammar's C++ key patterns, which
/// the format DSL cannot express.
fn additional_properties_schema(schema: &Map<String, Value>) -> Option<Value> {
    fn as_extra(value: Option<&Value>) -> Option<Option<Value>> {
        match value {
            Some(Value::Bool(false)) => Some(None),
            Some(Value::Bool(true)) => Some(Some(Value::Bool(true))),
            Some(extra @ Value::Object(_)) => Some(Some(extra.clone())),
            Some(_) => Some(None),
            None => None,
        }
    }
    // patternProperties/propertyNames keys cannot be expressed with the
    // format DSL; allow any additional parameter rather than reject a
    // pattern-matched one.
    if schema.contains_key("patternProperties") || schema.contains_key("propertyNames") {
        return Some(Value::Bool(true));
    }
    if let Some(extra) = as_extra(schema.get("additionalProperties")) {
        return extra;
    }
    if let Some(extra) = as_extra(schema.get("unevaluatedProperties")) {
        return extra;
    }
    None
}

fn extra_parameters_format(extras: Option<Value>, options: StructuralTagOptions) -> Format {
    match extras {
        Some(extras) => Format::star(permissive_parameter(&extras, options)),
        None => Format::const_string(""),
    }
}

/// Names from the schema's `required` array. Names missing from `properties`
/// are kept for the any-order count, matching the C++ set semantics.
fn required_properties(schema: &Map<String, Value>) -> std::collections::HashSet<&str> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| required.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// One schema-constrained parameter tag:
/// `<｜DSML｜ parameter name="key" string="true|false">value</｜DSML｜ parameter>`.
fn parameter_tag(
    key: &str,
    schema: &Value,
    root_defs: &Map<String, Value>,
    options: StructuralTagOptions,
) -> TagFormat {
    TagFormat::new(
        // The begin boundary ends at the name's opening quote; the closing
        // quote starts the `string` attribute framing in the content.
        format!("{} name=\"{key}", TOKENS.parameter_start),
        parameter_content(schema, root_defs, options),
        format!("{}\n", TOKENS.parameter_end),
    )
}

/// A parameter tag with an unconstrained name, used for additional
/// properties and unconstrained tools.
fn permissive_parameter(value_schema: &Value, options: StructuralTagOptions) -> Format {
    Format::tag(
        format!("{} name=\"", TOKENS.parameter_start),
        Format::sequence(vec![
            Format::regex(r#"[^"]+"#),
            parameter_content(value_schema, &Map::new(), options),
        ]),
        format!("{}\n", TOKENS.parameter_end),
    )
}

/// The `" string="...">value` framing shared by declared and additional
/// parameters. Like the C++ converter, the `string` flag is accepted in
/// either form for typed parameters and only the value is constrained;
/// untyped parameters keep the raw/JSON split.
fn parameter_content(
    schema: &Value,
    root_defs: &Map<String, Value>,
    options: StructuralTagOptions,
) -> Format {
    match value_kind(schema, root_defs) {
        ValueKind::RawString => string_attr(string_value_format(schema, root_defs)),
        ValueKind::Json => string_attr(json_value_format(schema, root_defs, options)),
        ValueKind::Any => Format::or(vec![
            Format::sequence(vec![
                Format::const_string("\" string=\"true\">"),
                Format::any_text_excluding(&[
                    TOKENS.parameter_end,
                    TOKENS.invoke_end,
                    TOKENS.tool_calls_end,
                ]),
            ]),
            Format::sequence(vec![
                Format::const_string("\" string=\"false\">"),
                json_value_format(&Value::Bool(true), root_defs, options),
            ]),
        ]),
    }
}

/// Frame a typed value behind the `string` attribute, accepting either flag
/// like the C++ converter.
fn string_attr(value: Format) -> Format {
    Format::sequence(vec![
        Format::const_string("\" string=\""),
        Format::or(vec![
            Format::const_string("true"),
            Format::const_string("false"),
        ]),
        Format::const_string("\">"),
        value,
    ])
}

/// How a parameter value is constrained, following the C++ level-1 value
/// rules: string-typed values are raw text, other typed values are JSON, and
/// anything that might be a string falls back to raw text (the C++ raw
/// branch accepts any text anyway).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    RawString,
    Json,
    Any,
}

const KNOWN_JSON_TYPES: &[&str] = &["integer", "number", "boolean", "null", "object", "array"];

fn value_kind(schema: &Value, root_defs: &Map<String, Value>) -> ValueKind {
    let mut schema = schema;
    for _ in 0..8 {
        let Some(map) = schema.as_object() else {
            // `true`, `false`, and non-object schemas stay permissive. The
            // C++ converter rejects a `false` property schema outright;
            // accepting it here keeps a broken tool schema from breaking
            // generation.
            return ValueKind::Any;
        };
        if let Some(kind) = classified_value_kind(map, root_defs) {
            return kind;
        }
        match map.get("$ref") {
            Some(Value::String(reference)) => match resolve_local_ref(reference, root_defs) {
                Some(target) => schema = target,
                None => return ValueKind::Any,
            },
            _ => return ValueKind::Any,
        }
    }
    ValueKind::Any
}

/// Classify an object schema by its declared value constraints, or None when
/// it carries none and a `$ref` should be followed instead.
fn classified_value_kind(
    map: &Map<String, Value>,
    root_defs: &Map<String, Value>,
) -> Option<ValueKind> {
    match map.get("type") {
        Some(Value::String(kind)) => {
            return Some(if kind == "string" {
                ValueKind::RawString
            } else if KNOWN_JSON_TYPES.contains(&kind.as_str()) {
                ValueKind::Json
            } else {
                ValueKind::Any
            });
        }
        Some(Value::Array(kinds)) if !kinds.is_empty() => {
            let mut saw_string = false;
            for kind in kinds.iter().filter_map(Value::as_str) {
                if kind == "string" {
                    saw_string = true;
                } else if !KNOWN_JSON_TYPES.contains(&kind) {
                    return Some(ValueKind::Any);
                }
            }
            return Some(if saw_string {
                ValueKind::Any
            } else {
                ValueKind::Json
            });
        }
        Some(_) => return Some(ValueKind::Any),
        None => {}
    }
    if let Some(value) = map.get("const") {
        return Some(if value.is_string() {
            ValueKind::RawString
        } else {
            ValueKind::Json
        });
    }
    if let Some(values) = map.get("enum").and_then(Value::as_array)
        && !values.is_empty()
    {
        let strings = values.iter().filter(|value| value.is_string()).count();
        return Some(match strings {
            count if count == values.len() => ValueKind::RawString,
            0 => ValueKind::Json,
            // A mixed enum's raw string branch accepts any text anyway.
            _ => ValueKind::Any,
        });
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = map.get(key).and_then(Value::as_array)
            && !branches.is_empty()
        {
            let all_json =
                branches.iter().all(|branch| value_kind(branch, root_defs) == ValueKind::Json);
            return Some(if all_json {
                ValueKind::Json
            } else {
                ValueKind::Any
            });
        }
    }
    None
}

/// Follow local `$ref` chains to the schema's object form.
fn resolve_schema_object<'a>(
    mut schema: &'a Value,
    root_defs: &'a Map<String, Value>,
) -> Option<&'a Map<String, Value>> {
    for _ in 0..8 {
        let map = schema.as_object()?;
        match map.get("$ref") {
            Some(Value::String(reference)) => schema = resolve_local_ref(reference, root_defs)?,
            _ => return Some(map),
        }
    }
    None
}

/// Resolve a local `#/$defs/...` or `#/definitions/...` reference.
fn resolve_local_ref<'a>(reference: &str, root_defs: &'a Map<String, Value>) -> Option<&'a Value> {
    for (prefix, key) in [("#/$defs/", "$defs"), ("#/definitions/", "definitions")] {
        if let Some(name) = reference.strip_prefix(prefix) {
            return root_defs.get(key)?.as_object()?.get(name);
        }
    }
    None
}

/// Constrain a raw string parameter value, mirroring the C++ level-1 string
/// rules: enum/const values become literal choices, `pattern` becomes a
/// regex, length bounds become a bounded repeat, and anything else is free
/// text that cannot swallow a closing marker.
fn string_value_format(schema: &Value, root_defs: &Map<String, Value>) -> Format {
    let free_text = || {
        Format::any_text_excluding(&[
            TOKENS.parameter_end,
            TOKENS.invoke_end,
            TOKENS.tool_calls_end,
        ])
    };
    let Some(schema) = resolve_schema_object(schema, root_defs) else {
        return free_text();
    };
    let values = schema
        .get("enum")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| schema.get("const").cloned().map(|value| vec![value]));
    if let Some(values) = values {
        let strings = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
        // An enum member must not contain a marker prefix, or the literal
        // choice could shadow the parameter's end marker.
        if !values.is_empty()
            && values.len() <= 256
            && strings.len() == values.len()
            && strings.iter().all(|value| !value.contains("<｜"))
        {
            return match strings.as_slice() {
                [value] => Format::const_string(*value),
                _ => Format::or(strings.into_iter().map(Format::const_string).collect()),
            };
        }
        return free_text();
    }
    if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
        return Format::regex(pattern);
    }
    if let Some(regex) = bounded_string_regex(schema) {
        return Format::regex(regex);
    }
    free_text()
}

/// Marker-safe single character for bounded string values: anything but the
/// ambiguous `<｜` marker prefix. A value ending in `<` or containing `<｜`
/// is not expressible, matching the kimi_k3 builder's tradeoff.
const STRING_ATOM: &str = "(?:[^<]|<[^｜])";

/// Length/pattern constraint for the raw string channel, if expressible.
///
/// The DSML string channel emits values raw (not JSON-quoted), so a JSON
/// schema cannot enforce string constraints there; unconstrained free text
/// lets maxLength violations through. xgrammar's regex engine has no
/// lookahead, so the end marker is kept unambiguous by excluding the `<｜`
/// prefix from value characters.
///
/// Returns a regex for the value, or None to keep permissive free text.
fn bounded_string_regex(schema: &Map<String, Value>) -> Option<String> {
    let max_len = schema.get("maxLength")?.as_i64()?;
    if !(0..=4096).contains(&max_len) {
        return None;
    }
    let mut min_len = schema.get("minLength").and_then(Value::as_i64).unwrap_or(0);
    if min_len < 0 || min_len > max_len {
        min_len = 0;
    }
    Some(format!("{STRING_ATOM}{{{min_len},{max_len}}}"))
}

/// Embed a non-string value schema as standard JSON, re-attaching root
/// `$defs`/`definitions` so local `$ref`s inside the slice still resolve.
fn json_value_format(
    schema: &Value,
    root_defs: &Map<String, Value>,
    options: StructuralTagOptions,
) -> Format {
    Format::JsonSchema(
        JsonSchemaFormat::new(attach_root_definitions(schema, root_defs))
            .with_any_order(options.any_order)
            .with_max_whitespace_cnt(options.max_whitespace_cnt),
    )
}

fn root_definitions(schema: &Map<String, Value>) -> Map<String, Value> {
    ["$defs", "definitions"]
        .into_iter()
        .filter_map(|key| schema.get(key).map(|value| (key.to_string(), value.clone())))
        .collect()
}

fn attach_root_definitions(schema: &Value, root_defs: &Map<String, Value>) -> Value {
    let Some(mut schema) = schema.as_object().cloned() else {
        return schema.clone();
    };
    for (key, value) in root_defs {
        schema.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Value::Object(schema)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn kind(schema: &Value, root_defs: &Map<String, Value>) -> ValueKind {
        value_kind(schema, root_defs)
    }

    #[test]
    fn value_kind_classifies_typed_schemas() {
        let empty = Map::new();
        assert_eq!(
            kind(&json!({"type": "string"}), &empty),
            ValueKind::RawString
        );
        assert_eq!(kind(&json!({"type": "integer"}), &empty), ValueKind::Json);
        assert_eq!(kind(&json!({"type": "object"}), &empty), ValueKind::Json);
        assert_eq!(
            kind(&json!({"type": ["integer", "null"]}), &empty),
            ValueKind::Json
        );
        assert_eq!(
            kind(&json!({"type": ["string", "null"]}), &empty),
            ValueKind::Any
        );
        assert_eq!(kind(&json!({"type": []}), &empty), ValueKind::Any);
        assert_eq!(kind(&json!({}), &empty), ValueKind::Any);
        assert_eq!(kind(&json!(true), &empty), ValueKind::Any);
        assert_eq!(kind(&json!(false), &empty), ValueKind::Any);
    }

    #[test]
    fn value_kind_classifies_const_enum_and_unions() {
        let empty = Map::new();
        assert_eq!(kind(&json!({"const": "x"}), &empty), ValueKind::RawString);
        assert_eq!(kind(&json!({"const": 1}), &empty), ValueKind::Json);
        assert_eq!(
            kind(&json!({"enum": ["a", "b"]}), &empty),
            ValueKind::RawString
        );
        assert_eq!(kind(&json!({"enum": [1, 2]}), &empty), ValueKind::Json);
        assert_eq!(kind(&json!({"enum": ["a", 1]}), &empty), ValueKind::Any);
        assert_eq!(kind(&json!({"enum": []}), &empty), ValueKind::Any);
        assert_eq!(
            kind(
                &json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
                &empty
            ),
            ValueKind::Json
        );
        assert_eq!(
            kind(
                &json!({"oneOf": [{"type": "string"}, {"type": "null"}]}),
                &empty
            ),
            ValueKind::Any
        );
        assert_eq!(kind(&json!({"anyOf": []}), &empty), ValueKind::Any);
    }

    #[test]
    fn value_kind_follows_local_refs() {
        let root_defs = root_definitions(
            json!({
                "$defs": {
                    "count": {"type": "integer"},
                    "name": {"type": "string"},
                    "loopy": {"$ref": "#/$defs/loopy"}
                }
            })
            .as_object()
            .unwrap(),
        );
        assert_eq!(
            kind(&json!({"$ref": "#/$defs/count"}), &root_defs),
            ValueKind::Json
        );
        assert_eq!(
            kind(&json!({"$ref": "#/$defs/name"}), &root_defs),
            ValueKind::RawString
        );
        assert_eq!(
            kind(&json!({"$ref": "#/$defs/missing"}), &root_defs),
            ValueKind::Any
        );
        assert_eq!(
            kind(&json!({"$ref": "https://example.com/x"}), &root_defs),
            ValueKind::Any
        );
        assert_eq!(
            kind(&json!({"$ref": "#/$defs/loopy"}), &root_defs),
            ValueKind::Any
        );
        // A sibling `type` wins over the referenced target, matching
        // kimi_k3's fixtures.
        assert_eq!(
            kind(
                &json!({"$ref": "#/$defs/count", "type": "string"}),
                &root_defs
            ),
            ValueKind::RawString
        );
    }

    #[test]
    fn string_value_format_prefers_literal_enum_choices() {
        let empty = Map::new();
        let format = string_value_format(&json!({"enum": ["celsius", "fahrenheit"]}), &empty);
        let value = serde_json::to_value(format).unwrap();
        assert_eq!(value["type"], "or");
        assert_eq!(value["elements"][0]["value"], "celsius");
        assert_eq!(value["elements"][1]["value"], "fahrenheit");

        let format = string_value_format(&json!({"const": "fixed"}), &empty);
        assert_eq!(
            serde_json::to_value(format).unwrap()["type"],
            "const_string"
        );
    }

    #[test]
    fn string_value_format_falls_back_for_marker_shaped_enum() {
        let empty = Map::new();
        let format = string_value_format(&json!({"enum": ["safe", "<｜unsafe"]}), &empty);
        assert_eq!(serde_json::to_value(format).unwrap()["type"], "any_text");

        let format = string_value_format(&json!({"enum": ["a", 1]}), &empty);
        assert_eq!(serde_json::to_value(format).unwrap()["type"], "any_text");
    }

    #[test]
    fn string_value_format_uses_pattern_and_bounded_length() {
        let empty = Map::new();
        let format = string_value_format(&json!({"pattern": "^[a-z]+$"}), &empty);
        assert_eq!(serde_json::to_value(format).unwrap()["pattern"], "^[a-z]+$");

        let format = string_value_format(&json!({"minLength": 2, "maxLength": 5}), &empty);
        assert_eq!(
            serde_json::to_value(format).unwrap()["pattern"],
            "(?:[^<]|<[^｜]){2,5}"
        );

        // Unbounded or unsound bounds stay permissive.
        let format = string_value_format(&json!({"maxLength": -1}), &empty);
        assert_eq!(serde_json::to_value(format).unwrap()["type"], "any_text");
        let format = string_value_format(&json!({"minLength": 2}), &empty);
        assert_eq!(serde_json::to_value(format).unwrap()["type"], "any_text");
    }

    #[test]
    fn parameters_content_enforces_declared_names_and_required() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query"],
            "additionalProperties": false
        });
        let content = parameters_content(&parameters, StructuralTagOptions::default());
        let value = serde_json::to_value(content).unwrap();
        let elements = value["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0]["type"], "tag");
        assert_eq!(elements[0]["begin"], "<｜DSML｜ parameter name=\"query");
        assert_eq!(elements[1]["type"], "optional");
        assert_eq!(
            elements[1]["content"]["begin"],
            "<｜DSML｜ parameter name=\"limit"
        );
    }

    #[test]
    fn parameters_content_appends_additional_properties_last() {
        let parameters = json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": true
        });
        let content = parameters_content(&parameters, StructuralTagOptions::default());
        let value = serde_json::to_value(content).unwrap();
        let elements = value["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[1]["type"], "star");
        assert_eq!(
            elements[1]["content"]["begin"],
            "<｜DSML｜ parameter name=\""
        );

        // Like xgrammar's strict mode, objects are closed unless
        // additionalProperties explicitly allows more.
        let parameters = json!({
            "type": "object",
            "properties": {"query": {"type": "string"}}
        });
        let content = parameters_content(&parameters, StructuralTagOptions::default());
        let value = serde_json::to_value(content).unwrap();
        let elements = value["elements"].as_array().unwrap();
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0]["type"], "optional");
    }

    #[test]
    fn parameters_content_without_properties_matches_schema_emptiness() {
        let open = parameters_content(
            &json!({"type": "object", "additionalProperties": true}),
            StructuralTagOptions::default(),
        );
        assert_eq!(serde_json::to_value(open).unwrap()["type"], "star");

        let closed =
            parameters_content(&json!({"type": "object"}), StructuralTagOptions::default());
        assert_eq!(
            serde_json::to_value(closed).unwrap(),
            json!({"type": "const_string", "value": ""})
        );

        let permissive = parameters_content(&Value::Bool(true), StructuralTagOptions::default());
        assert_eq!(serde_json::to_value(permissive).unwrap()["type"], "star");

        let unsatisfiable =
            parameters_content(&Value::Bool(false), StructuralTagOptions::default());
        assert_eq!(
            serde_json::to_value(unsatisfiable).unwrap(),
            json!({"type": "const_string", "value": ""})
        );
    }

    #[test]
    fn parameters_content_any_order_repeats_the_parameter_choice() {
        let parameters = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["query"]
        });
        let content = parameters_content(
            &parameters,
            StructuralTagOptions::default().with_any_order(true),
        );
        let value = serde_json::to_value(content).unwrap();
        assert_eq!(value["type"], "repeat");
        assert_eq!(value["min"], 1);
        assert_eq!(value["max"], -1);
        // Closed object: only the two declared parameter tags.
        assert_eq!(value["content"]["elements"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn parameters_content_stays_permissive_for_quoted_names() {
        let parameters = json!({
            "type": "object",
            "properties": {"weird\"name": {"type": "string"}},
            "required": ["weird\"name"],
            "additionalProperties": false
        });
        let content = parameters_content(&parameters, StructuralTagOptions::default());
        assert_eq!(serde_json::to_value(content).unwrap()["type"], "star");
    }
}
