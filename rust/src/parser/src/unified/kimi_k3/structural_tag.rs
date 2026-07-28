// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structural-tag grammar for Kimi K3 XTML tool calls.

use serde_json::{Map, Value};
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{
    StructuralTagBuilder, StructuralTagContext, StructuralTagOptions,
};
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat, StructuralTag, TagFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam, function_parameters};

use super::{
    ARG_CLOSE, CALL_CLOSE, END_OF_MSG, JSON_CLOSE, JSON_OPEN, MESSAGE_CLOSE, OPEN, RESPONSE_CLOSE,
    RESPONSE_OPEN, SEP, THINK_CLOSE, TOOLS_CLOSE, TOOLS_OPEN,
};

pub(super) static KIMI_K3_STRUCTURAL_TAG_BUILDER: KimiK3StructuralTagBuilder =
    KimiK3StructuralTagBuilder;

const XTML_TYPES: &[&str] = &["string", "number", "boolean", "null", "object", "array"];

/// Kimi K3 XTML structural-tag builder.
#[derive(Debug, Clone, Copy, Default)]
pub struct KimiK3StructuralTagBuilder;

impl StructuralTagBuilder for KimiK3StructuralTagBuilder {
    fn build(&self, ctx: StructuralTagContext<'_>) -> Result<StructuralTag> {
        let mut elements = response_prefix(ctx.options.reasoning);

        let tools = match ctx.tool_choice {
            // Serving lowering filters empty tools before calling the builder,
            // while direct `build_structural_tag(..., auto, ...)` calls do not.
            BuilderToolChoice::Auto if ctx.function_tools.is_empty() => None,
            BuilderToolChoice::Auto => Some(Format::optional(tools_channel(
                ctx.function_tools,
                ctx.options,
            ))),
            BuilderToolChoice::Forced | BuilderToolChoice::Required => {
                Some(tools_channel(ctx.function_tools, ctx.options))
            }
        };
        if let Some(tools) = tools {
            elements.push(tools);
        }
        elements.push(Format::optional(Format::const_string(MESSAGE_CLOSE)));

        Ok(StructuralTag::new(Format::sequence(elements)))
    }
}

fn response_prefix(reasoning: bool) -> Vec<Format> {
    let mut elements = Vec::new();
    if reasoning {
        elements.push(Format::tag(
            "",
            Format::any_text_excluding(&[THINK_CLOSE, END_OF_MSG]),
            THINK_CLOSE,
        ));
        elements.push(Format::const_string(RESPONSE_OPEN));
    } else {
        elements.push(Format::optional(Format::const_string(RESPONSE_OPEN)));
    }
    elements.push(Format::tag(
        "",
        Format::any_text_excluding(&[RESPONSE_CLOSE, TOOLS_OPEN, MESSAGE_CLOSE, END_OF_MSG]),
        RESPONSE_CLOSE,
    ));
    elements
}

fn tools_channel(tools: &[FunctionToolParam], options: StructuralTagOptions) -> Format {
    let calls = tools.iter().map(|tool| call_tag(tool, options)).collect();
    Format::tag(
        TOOLS_OPEN,
        Format::tags_with_separator(calls, "", true, false),
        TOOLS_CLOSE,
    )
}

fn call_tag(tool: &FunctionToolParam, options: StructuralTagOptions) -> TagFormat {
    let parameters = function_parameters(&tool.function);
    let call_body = Format::or(vec![
        typed_arguments(&parameters, options),
        raw_json_arguments(&parameters, options),
    ]);

    TagFormat::new(
        format!(
            "{OPEN}call tool=\"{}\" index=\"",
            escape_attr_value(&tool.function.name)
        ),
        Format::sequence(vec![
            Format::regex("[1-9][0-9]*"),
            Format::const_string(format!("\"{SEP}")),
            call_body,
        ]),
        CALL_CLOSE,
    )
}

fn typed_arguments(parameters: &Value, options: StructuralTagOptions) -> Format {
    let Some(schema) = parameters.as_object() else {
        return if parameters == &Value::Bool(false) {
            Format::const_string("")
        } else {
            Format::star(permissive_argument())
        };
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return Format::star(permissive_argument());
    };
    if properties.is_empty() {
        return Format::star(permissive_argument());
    }

    let root_defs = root_definitions(schema);
    let arguments = properties
        .iter()
        .flat_map(|(key, schema)| argument_tags(key, schema, &root_defs, options))
        .map(Format::Tag)
        .collect::<Vec<_>>();
    let arguments = match arguments.as_slice() {
        [argument] => argument.clone(),
        _ => Format::or(arguments),
    };
    // Keep typed arguments order-agnostic and non-unique, but do not allow an
    // empty call when the root schema declares required properties.
    if schema
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| !required.is_empty())
    {
        Format::plus(arguments)
    } else {
        Format::star(arguments)
    }
}

fn argument_tags(
    key: &str,
    schema: &Value,
    root_defs: &Map<String, Value>,
    options: StructuralTagOptions,
) -> Vec<TagFormat> {
    let types = schema_types(schema);
    types
        .into_iter()
        .map(|xtml_type| {
            let content = if xtml_type == "string" {
                string_argument_content(schema)
            } else {
                json_schema(
                    attach_root_definitions(&narrow_schema_type(schema, xtml_type), root_defs),
                    options,
                )
            };
            TagFormat::new(
                format!(
                    "{OPEN}argument key=\"{}\" type=\"{xtml_type}\"{SEP}",
                    escape_attr_value(key)
                ),
                content,
                ARG_CLOSE,
            )
        })
        .collect()
}

fn schema_types(schema: &Value) -> Vec<&'static str> {
    let Some(schema) = schema.as_object() else {
        return XTML_TYPES.to_vec();
    };
    let mut types = Vec::new();
    match schema.get("type") {
        Some(Value::String(value)) => push_schema_type(&mut types, value),
        Some(Value::Array(values)) => {
            for value in values.iter().filter_map(Value::as_str) {
                push_schema_type(&mut types, value);
            }
        }
        _ => {}
    }
    if types.is_empty()
        && let Some(value) = schema.get("const")
    {
        push_value_type(&mut types, value);
    }
    if types.is_empty()
        && let Some(values) = schema.get("enum").and_then(Value::as_array)
    {
        for value in values {
            push_value_type(&mut types, value);
        }
    }
    if types.is_empty() {
        XTML_TYPES.to_vec()
    } else {
        types
    }
}

fn push_value_type(types: &mut Vec<&'static str>, value: &Value) {
    let xtml_type = match value {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    };
    if !types.contains(&xtml_type) {
        types.push(xtml_type);
    }
}

fn narrow_schema_type(schema: &Value, xtml_type: &str) -> Value {
    let Some(mut schema) = schema.as_object().cloned() else {
        return schema.clone();
    };
    let json_type = if xtml_type == "number" && explicitly_integer_only(&schema) {
        "integer"
    } else {
        xtml_type
    };
    schema.insert("type".to_string(), Value::String(json_type.to_string()));
    Value::Object(schema)
}

fn explicitly_integer_only(schema: &Map<String, Value>) -> bool {
    match schema.get("type") {
        Some(Value::String(value)) => value == "integer",
        Some(Value::Array(values)) => {
            let values = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
            values.contains(&"integer") && !values.contains(&"number")
        }
        _ => false,
    }
}

fn push_schema_type(types: &mut Vec<&'static str>, json_type: &str) {
    let xtml_type = match json_type {
        "string" => Some("string"),
        "integer" | "number" => Some("number"),
        "boolean" => Some("boolean"),
        "null" => Some("null"),
        "object" => Some("object"),
        "array" => Some("array"),
        _ => None,
    };
    if let Some(xtml_type) = xtml_type
        && !types.contains(&xtml_type)
    {
        types.push(xtml_type);
    }
}

fn string_argument_content(schema: &Value) -> Format {
    let Some(schema) = schema.as_object() else {
        return Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    };
    let values = schema
        .get("enum")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| schema.get("const").cloned().map(|value| vec![value]));
    let Some(values) = values else {
        return Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    };
    if values.is_empty()
        || values.len() > 256
        || values
            .iter()
            .any(|value| value.as_str().is_none_or(|value| value.contains("<|")))
    {
        return Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE]);
    }
    let values = values.iter().filter_map(Value::as_str).collect::<Vec<_>>();
    match values.as_slice() {
        [value] => Format::const_string(*value),
        _ => Format::or(values.into_iter().map(Format::const_string).collect()),
    }
}

fn raw_json_arguments(parameters: &Value, options: StructuralTagOptions) -> Format {
    Format::tag(
        format!("{JSON_OPEN} type=\"object\"{SEP}"),
        json_schema(parameters.clone(), options),
        JSON_CLOSE,
    )
}

fn permissive_argument() -> Format {
    let key = Format::regex(r#"(?:[^<\"&]|&(?:amp|quot);|<[^|])*"#);
    let alternatives = XTML_TYPES
        .iter()
        .map(|xtml_type| {
            Format::sequence(vec![
                key.clone(),
                Format::const_string(format!("\" type=\"{xtml_type}\"{SEP}")),
                if *xtml_type == "string" {
                    Format::any_text_excluding(&[ARG_CLOSE, CALL_CLOSE])
                } else {
                    Format::json_schema(Value::Bool(true))
                },
            ])
        })
        .collect();
    Format::tag(
        format!("{OPEN}argument key=\""),
        Format::or(alternatives),
        ARG_CLOSE,
    )
}

fn json_schema(schema: Value, options: StructuralTagOptions) -> Format {
    Format::JsonSchema(
        JsonSchemaFormat::new(schema)
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

fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;
    use xgrammar_structural_tag::builders::StructuralTagOptions;
    use xgrammar_structural_tag::{
        FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
    };

    use super::KimiK3StructuralTagBuilder;

    fn tool(name: &str, parameters: serde_json::Value) -> ToolParam {
        ToolParam::Function(FunctionToolParam::new(
            FunctionDefinition::new(name).with_parameters(parameters),
        ))
    }

    #[test]
    fn required_structural_tag_matches_xtml_channels() {
        let tools = vec![tool(
            "get_weather",
            json!({
                "$defs": {
                    "place": { "type": "object", "properties": { "city": { "type": "string" } } }
                },
                "type": "object",
                "properties": {
                    "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
                    "place": { "$ref": "#/$defs/place", "type": "object" }
                },
                "required": ["place"]
            }),
        )];
        let tag = build_structural_tag(
            KimiK3StructuralTagBuilder,
            &tools,
            ToolChoice::required(),
            StructuralTagOptions::default().with_reasoning(false),
        )
        .unwrap();

        expect![[r##"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|open|>response<|sep|>"}},{"type":"tag","begin":"","content":{"type":"any_text","excludes":["<|close|>response<|sep|>","<|open|>tools<|sep|>","<|close|>message<|sep|>","<|end_of_msg|>"]},"end":"<|close|>response<|sep|>"},{"type":"tag","begin":"<|open|>tools<|sep|>","content":{"type":"tags_with_separator","tags":[{"begin":"<|open|>call tool=\"get_weather\" index=\"","content":{"type":"sequence","elements":[{"type":"regex","pattern":"[1-9][0-9]*"},{"type":"const_string","value":"\"<|sep|>"},{"type":"or","elements":[{"type":"plus","content":{"type":"or","elements":[{"type":"tag","begin":"<|open|>argument key=\"unit\" type=\"string\"<|sep|>","content":{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]},"end":"<|close|>argument<|sep|>"},{"type":"tag","begin":"<|open|>argument key=\"place\" type=\"object\"<|sep|>","content":{"type":"json_schema","json_schema":{"$ref":"#/$defs/place","type":"object","$defs":{"place":{"type":"object","properties":{"city":{"type":"string"}}}}},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"<|close|>argument<|sep|>"}]}},{"type":"tag","begin":"<|open|>json type=\"object\"<|sep|>","content":{"type":"json_schema","json_schema":{"$defs":{"place":{"type":"object","properties":{"city":{"type":"string"}}}},"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]},"place":{"$ref":"#/$defs/place","type":"object"}},"required":["place"]},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"<|close|>json<|sep|>"}]}]},"end":"<|close|>call<|sep|>"}],"separator":"","at_least_one":true,"stop_after_first":false},"end":"<|close|>tools<|sep|>"},{"type":"optional","content":{"type":"const_string","value":"<|close|>message<|sep|>"}}]}}"##]].assert_eq(&tag.to_json_string().unwrap());
    }

    #[test]
    fn typed_arguments_require_one_tag_only_for_nonempty_required() {
        let required = super::typed_arguments(
            &json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
            StructuralTagOptions::default(),
        );
        let optional = super::typed_arguments(
            &json!({
                "type": "object",
                "properties": { "query": { "type": "string" } }
            }),
            StructuralTagOptions::default(),
        );
        let empty_required = super::typed_arguments(
            &json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": []
            }),
            StructuralTagOptions::default(),
        );

        assert_eq!(serde_json::to_value(required).unwrap()["type"], "plus");
        assert_eq!(serde_json::to_value(optional).unwrap()["type"], "star");
        assert_eq!(
            serde_json::to_value(empty_required).unwrap()["type"],
            "star"
        );
    }

    #[test]
    fn reasoning_grammar_starts_inside_prefilled_think_channel() {
        let tools = vec![tool("ping", json!({ "type": "object", "properties": {} }))];
        let tag = build_structural_tag(
            KimiK3StructuralTagBuilder,
            &tools,
            ToolChoice::auto(),
            StructuralTagOptions::default().with_reasoning(true),
        )
        .unwrap();
        let value = serde_json::to_value(tag).unwrap();

        assert_eq!(
            value["format"]["elements"][0]["end"],
            "<|close|>think<|sep|>"
        );
        assert_eq!(
            value["format"]["elements"][1]["value"],
            "<|open|>response<|sep|>"
        );
        assert_eq!(value["format"]["elements"][3]["type"], "optional");
    }

    #[test]
    fn forced_choice_keeps_only_the_named_tool() {
        let tools = vec![
            tool("search", json!({ "type": "object" })),
            tool("lookup", json!({ "type": "object" })),
        ];
        let tag = build_structural_tag(
            KimiK3StructuralTagBuilder,
            &tools,
            ToolChoice::function("lookup"),
            StructuralTagOptions::default().with_reasoning(false),
        )
        .unwrap()
        .to_json_string()
        .unwrap();

        assert!(tag.contains("lookup"));
        assert!(!tag.contains("search"));
    }

    #[test]
    fn union_argument_content_matches_its_xtml_type() {
        let tools = vec![tool(
            "set_count",
            json!({
                "type": "object",
                "properties": {
                    "count": { "type": ["integer", "null"] }
                }
            }),
        )];
        let tag = build_structural_tag(
            KimiK3StructuralTagBuilder,
            &tools,
            ToolChoice::required(),
            StructuralTagOptions::default(),
        )
        .unwrap()
        .to_json_string()
        .unwrap();

        assert!(tag.contains(r#"type=\"number\""#));
        assert!(tag.contains(r#""json_schema":{"type":"integer"}"#), "{tag}");
        assert!(tag.contains(r#"type=\"null\""#));
        assert!(tag.contains(r#""json_schema":{"type":"null"}"#), "{tag}");
    }

    #[test]
    fn unsafe_string_enum_falls_back_as_a_whole() {
        let format = super::string_argument_content(&json!({
            "type": "string",
            "enum": ["safe", "<|unsafe"]
        }));

        assert_eq!(serde_json::to_value(format).unwrap()["type"], "any_text");
    }
}
