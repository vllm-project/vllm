// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structural-tag grammar for Kimi K3 XTML tool calls.

use serde_json::Value;
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{
    ReasoningMode, StructuralTagBuilder, StructuralTagContext, StructuralTagOptions,
};
use xgrammar_structural_tag::format::{
    Format, JsonSchemaFormat, JsonSchemaStyle, StructuralTag, TagFormat,
};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam, function_parameters};

use super::{
    CALL_CLOSE, CLOSE, END_OF_MSG, JSON_CLOSE, JSON_OPEN, MESSAGE_CLOSE, OPEN, RESPONSE_CLOSE,
    RESPONSE_OPEN, SEP, THINK_CLOSE, THINK_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
};

pub(super) static KIMI_K3_STRUCTURAL_TAG_BUILDER: KimiK3StructuralTagBuilder =
    KimiK3StructuralTagBuilder;

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
    let call_body = Format::or(vec![
        json_schema(parameters.clone(), JsonSchemaStyle::KimiK3Xml, options),
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
        json_schema(parameters.clone(), JsonSchemaStyle::Json, options),
        JSON_CLOSE,
    )
}

fn json_schema(schema: Value, style: JsonSchemaStyle, options: StructuralTagOptions) -> Format {
    Format::JsonSchema(
        JsonSchemaFormat::new(schema)
            .with_style(style)
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

        expect![[r##"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|open|>response<|sep|>"}},{"type":"tag","begin":"","content":{"type":"any_text","excludes":["<|open|>","<|close|>","<|end_of_msg|>"],"max_tokens":null,"max_chars":null},"end":"<|close|>response<|sep|>"},{"type":"tag","begin":"<|open|>tools<|sep|>","content":{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<|open|>call tool=\"get_weather\" index=\"","content":{"type":"sequence","elements":[{"type":"regex","pattern":"[1-9][0-9]*"},{"type":"const_string","value":"\"<|sep|>"},{"type":"or","elements":[{"type":"json_schema","json_schema":{"$defs":{"place":{"type":"object","properties":{"city":{"type":"string"}}}},"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]},"place":{"$ref":"#/$defs/place","type":"object"}},"required":["place"]},"style":"kimi_k3_xml","any_order":false,"max_whitespace_cnt":null},{"type":"tag","begin":"<|open|>json type=\"object\"<|sep|>","content":{"type":"json_schema","json_schema":{"$defs":{"place":{"type":"object","properties":{"city":{"type":"string"}}}},"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]},"place":{"$ref":"#/$defs/place","type":"object"}},"required":["place"]},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"<|close|>json<|sep|>"}]}]},"end":"<|close|>call<|sep|>"}],"separator":"","at_least_one":true,"stop_after_first":false},"end":"<|close|>tools<|sep|>"},{"type":"optional","content":{"type":"const_string","value":"<|close|>message<|sep|>"}}]}}"##]].assert_eq(&tag.to_json_string().unwrap());
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
}
