// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structural-tag grammar for Kimi K3 XTML tool calls.

use serde_json::Value;
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{
    ReasoningMode, StructuralTagBuilder, StructuralTagContext, StructuralTagOptions,
};
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat, StructuralTag, TagFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam, function_parameters};

use crate::output_grammar::arguments::{
    self, ArgumentContext, ArgumentOptions, ArgumentSyntax, JsonType, ParameterKey, RawString,
    ValueOption, group_by, one_of,
};

use super::{
    ARG_CLOSE, ARG_OPEN, CALL_CLOSE, CLOSE, END_OF_MSG, JSON_CLOSE, JSON_OPEN, MESSAGE_CLOSE, OPEN,
    RESPONSE_CLOSE, RESPONSE_OPEN, SEP, THINK_CLOSE, THINK_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
};

pub(super) static KIMI_K3_STRUCTURAL_TAG_BUILDER: KimiK3StructuralTagBuilder =
    KimiK3StructuralTagBuilder;

/// XTML arguments:
/// `<|open|>argument key="KEY" type="TYPE"<|sep|>VALUE<|close|>argument<|sep|>`,
/// adjacent to each other.
///
/// The parser decodes each value by its `type` attribute, so every value option
/// sits under the attribute of its own JSON type. String values are raw text
/// and every other value is JSON.
struct XtmlArguments;

impl ArgumentSyntax for XtmlArguments {
    fn separator(&self, _cx: &ArgumentContext<'_>) -> Option<Format> {
        None
    }

    fn parameter(
        &self,
        _cx: &ArgumentContext<'_>,
        key: ParameterKey<'_>,
        options: &[ValueOption<'_>],
    ) -> Option<Format> {
        let escaped;
        let key = match key {
            ParameterKey::Declared(key) => {
                escaped = escape_attr_value(key);
                ParameterKey::Declared(&escaped)
            }
            ParameterKey::Free => ParameterKey::Free,
        };
        let arguments = group_by(options, |option| option.ty)
            .into_iter()
            .filter_map(|(ty, options)| {
                let values = options
                    .into_iter()
                    .filter_map(|option| match ty {
                        // The parser ends the call at its close marker, so a
                        // string value may contain neither close marker.
                        JsonType::String => {
                            option.raw_string(&[ARG_CLOSE, CALL_CLOSE]).map(RawString::into_format)
                        }
                        _ => Some(option.json()),
                    })
                    .collect::<Vec<_>>();
                (!values.is_empty()).then(|| {
                    let suffix = format!("\" type=\"{}\"{SEP}", ty.name());
                    key.tag(
                        &format!("{ARG_OPEN} key=\""),
                        &suffix,
                        one_of(values),
                        ARG_CLOSE,
                    )
                })
            })
            .collect::<Vec<_>>();
        (!arguments.is_empty()).then(|| one_of(arguments))
    }
}

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

fn response_prefix(reasoning: ReasoningMode) -> Vec<Format> {
    let mut elements = Vec::new();
    if reasoning != ReasoningMode::Disabled {
        let prefix = Format::tag(
            if reasoning == ReasoningMode::Auto {
                THINK_OPEN
            } else {
                ""
            },
            Format::any_text_excluding(&[THINK_CLOSE, END_OF_MSG]),
            THINK_CLOSE,
        );
        elements.push(if reasoning == ReasoningMode::Auto {
            Format::optional(prefix)
        } else {
            prefix
        });
        elements.push(Format::const_string(RESPONSE_OPEN));
    } else {
        elements.push(Format::optional(Format::const_string(RESPONSE_OPEN)));
    }
    elements.push(Format::tag(
        "",
        // Keep marker-looking prefixes out of response text so the grammar
        // must resolve them through a valid channel boundary.
        Format::any_text_excluding(&[OPEN, CLOSE, END_OF_MSG]),
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
    let typed_arguments = arguments::arguments(
        &parameters,
        &XtmlArguments,
        &ArgumentOptions {
            any_order: options.any_order,
            max_whitespace_cnt: options.max_whitespace_cnt,
            excludes: vec![],
        },
    );
    let call_body = Format::or(vec![
        typed_arguments,
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

fn raw_json_arguments(parameters: &Value, options: StructuralTagOptions) -> Format {
    Format::tag(
        format!("{JSON_OPEN} type=\"object\"{SEP}"),
        json_schema(parameters.clone(), options),
        JSON_CLOSE,
    )
}

fn json_schema(schema: Value, options: StructuralTagOptions) -> Format {
    Format::JsonSchema(
        JsonSchemaFormat::new(schema)
            .with_any_order(options.any_order)
            .with_max_whitespace_cnt(options.max_whitespace_cnt),
    )
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
    use crate::output_grammar::test_utils::outline;

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

        expect![[r#"
            sequence
              optional `<|open|>response<|sep|>`
              tag `` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
              tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                tags_with_separator `` at_least_one
                  tag `<|open|>call tool="get_weather" index="` .. `<|close|>call<|sep|>`
                    sequence
                      /[1-9][0-9]*/
                      `"<|sep|>`
                      or
                        sequence
                          optional
                            tag `<|open|>argument key="unit" type="string"<|sep|>` .. `<|close|>argument<|sep|>`
                              or
                                `celsius`
                                `fahrenheit`
                          tag `<|open|>argument key="place" type="object"<|sep|>` json({ city?: string } where place = { city?: string }) `<|close|>argument<|sep|>`
                        tag `<|open|>json type="object"<|sep|>` json({ unit?: "celsius" | "fahrenheit", place: place & object } where place = { city?: string }) `<|close|>json<|sep|>`
              optional `<|close|>message<|sep|>`
        "#]].assert_eq(&outline(&tag.format));
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
}
