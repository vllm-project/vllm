// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Visible-text grammar for DeepSeek V4.1 tool calls.
//!
//! Adapted from xgrammar-structural-tag 0.2.0's DeepSeek V4 builder, with
//! V4.1's spaced DSML tags and schema-unconstrained parameter values.

use serde_json::Value;
use xgrammar_structural_tag::Result;
use xgrammar_structural_tag::builders::{
    StructuralTagBuilder, StructuralTagContext, StructuralTagOptions,
};
use xgrammar_structural_tag::format::{
    Format, JsonSchemaFormat, StructuralTag, TagFormat, TriggeredTagsFormat,
};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam};

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
    // TODO: enforce tool.parameters and strict schema semantics once V4.1
    // argument conversion is available. This builder intentionally ignores
    // parameters and strict: it constrains tool names and DSML/value syntax
    // only. Parameter names, presence, uniqueness and value schemas remain
    // unconstrained, including for strict=true. Serving still uses strict to
    // decide whether auto tool choice activates a grammar.
    TagFormat::new(
        format!("{} name=\"{}\">\n", TOKENS.invoke_start, tool.function.name),
        Format::star(parameter(options)),
        format!("{}\n", TOKENS.invoke_end),
    )
}

fn parameter(options: StructuralTagOptions) -> Format {
    Format::tag(
        format!("{} name=\"", TOKENS.parameter_start),
        Format::sequence(vec![
            Format::regex(r#"[^"]+"#),
            Format::const_string("\" string=\""),
            Format::or(vec![
                Format::sequence(vec![
                    Format::const_string("true\">"),
                    Format::any_text_excluding(&[
                        TOKENS.parameter_end,
                        TOKENS.invoke_end,
                        TOKENS.tool_calls_end,
                    ]),
                ]),
                Format::sequence(vec![
                    Format::const_string("false\">"),
                    Format::JsonSchema(
                        JsonSchemaFormat::new(Value::Bool(true))
                            .with_any_order(options.any_order)
                            .with_max_whitespace_cnt(options.max_whitespace_cnt),
                    ),
                ]),
            ]),
        ]),
        format!("{}\n", TOKENS.parameter_end),
    )
}
