// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Structural-tag grammar for MiniMax M3 namespace-delimited tool calls.

use std::cell::Cell;

use xgrammar_structural_tag::builders::{
    StructuralTagBuilder, StructuralTagContext, StructuralTagOptions,
};
use xgrammar_structural_tag::format::{Format, StructuralTag, TagFormat, TriggeredTagsFormat};
use xgrammar_structural_tag::tool::{BuilderToolChoice, FunctionToolParam, function_parameters};
use xgrammar_structural_tag::{Error, Result};

use super::{
    ELEMENT_END_START, ELEMENT_START, INVOKE_END, INVOKE_START, NAMESPACE, TOOL_CALL_END,
    TOOL_CALL_START,
};
use crate::output_grammar::arguments::{
    self, ArgumentOptions, ArgumentSyntax, JsonType, ParameterKey, ValueOption, one_of,
};
use crate::reasoning::{M3_THINK_END, M3_THINK_START};

pub(super) static MINIMAX_M3_STRUCTURAL_TAG_BUILDER: MinimaxM3StructuralTagBuilder =
    MinimaxM3StructuralTagBuilder;

/// MiniMax M3 arguments: `]<]minimax[>[<KEY>VALUE]<]minimax[>[</KEY>`, adjacent
/// to each other.
///
/// Values nest as the chat template renders them: an object is its fields as
/// child elements, an array one `<item>` element per item, a string raw text,
/// and any other value JSON. A closing tag repeats its key, so a grammar can
/// only spell declared keys; a value that admits undeclared keys or takes any
/// shape is free text up to its closing tag, which the parser decodes without
/// the schema constraining it. That text ends early only if the value nests an
/// element named like its own key. A value that nests in itself always does,
/// so it is recorded and its whole call becomes free text instead.
#[derive(Default)]
struct M3Arguments {
    /// Whether a value nests in itself.
    recursive: Cell<bool>,
}

impl ArgumentSyntax for M3Arguments {
    fn separator(&self) -> Option<Format> {
        None
    }

    fn parameter(&self, key: ParameterKey<'_>, options: &[ValueOption<'_>]) -> Option<Format> {
        let ParameterKey::Declared(key) = key else {
            return None;
        };
        // The parser reads a key up to `>` and rejects closing-tag spellings.
        if key.trim().is_empty() || key.contains('>') || key.starts_with('/') {
            return None;
        }
        let close = format!("{ELEMENT_END_START}{key}>");
        let values = self.values(options, &close);
        (!values.is_empty())
            .then(|| Format::tag(format!("{ELEMENT_START}{key}>"), one_of(values), close))
    }
}

impl M3Arguments {
    /// The grammars of a value taking one of `options` that ends at `close`.
    fn values(&self, options: &[ValueOption<'_>], close: &str) -> Vec<Format> {
        options
            .iter()
            .filter_map(|option| match option.ty {
                // The parser reads a string up to the next namespace marker.
                JsonType::String => option.raw_string(&[NAMESPACE]),
                JsonType::Object => {
                    Some(option.fields(self).unwrap_or_else(|| self.free(option, close)))
                }
                JsonType::Array => Some(
                    option
                        .items()
                        .map_or_else(|| self.free(option, close), |items| self.list(&items)),
                ),
                _ => Some(option.json()),
            })
            .collect()
    }

    /// One `<item>` element per item of an array whose items take `options`.
    fn list(&self, options: &[ValueOption<'_>]) -> Format {
        let close = format!("{ELEMENT_END_START}item>");
        let values = self.values(options, &close);
        if values.is_empty() {
            return Format::const_string("");
        }
        Format::star(Format::tag(
            format!("{ELEMENT_START}item>"),
            one_of(values),
            close,
        ))
    }

    /// Text of `option` up to `close` that the schema does not constrain.
    fn free(&self, option: &ValueOption<'_>, close: &str) -> Format {
        if option.recurses() {
            self.recursive.set(true);
        }
        Format::any_text_excluding(&[close, INVOKE_END])
    }
}

/// MiniMax M3 structural-tag builder for the output after reasoning.
///
/// It describes only the response text and tool calls; the reasoning parser
/// wraps it with the reasoning phase, so `options.reasoning` is ignored.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct MinimaxM3StructuralTagBuilder;

impl StructuralTagBuilder for MinimaxM3StructuralTagBuilder {
    fn build(&self, ctx: StructuralTagContext<'_>) -> Result<StructuralTag> {
        if !ctx.builtin_tools.is_empty() {
            return Err(Error::UnsupportedBuiltinTools {
                model: "minimax_m3",
            });
        }
        let options = ctx.options;
        let mut invokes = ctx
            .function_tools
            .iter()
            .map(|tool| invoke_tag(tool, options))
            .collect::<Vec<_>>();
        // Keep a second reasoning block and stray call markers out of text.
        let excludes: &[&str] = if options.exclude_special_tokens {
            &[
                M3_THINK_START,
                M3_THINK_END,
                TOOL_CALL_END,
                INVOKE_START,
                INVOKE_END,
            ]
        } else {
            &[]
        };
        let block = |invokes: Vec<TagFormat>| {
            TagFormat::new(
                format!("{TOOL_CALL_START}\n"),
                Format::tags_with_separator(invokes, "", true, !options.parallel_tool_calls),
                TOOL_CALL_END,
            )
        };
        let calls = |invokes: Vec<TagFormat>| {
            TriggeredTagsFormat::new(&[TOOL_CALL_START], vec![block(invokes)])
                .with_excludes(excludes)
                .with_stop_after_first(!options.parallel_tool_calls)
        };

        let format = match ctx.tool_choice {
            BuilderToolChoice::Auto if invokes.is_empty() => {
                Format::any_text_excluding(&[&[TOOL_CALL_START], excludes].concat())
            }
            BuilderToolChoice::Auto => Format::TriggeredTags(calls(invokes)),
            BuilderToolChoice::Required => {
                Format::TriggeredTags(calls(invokes).require_at_least_one())
            }
            // Normalization leaves exactly one tool for a forced choice.
            BuilderToolChoice::Forced => Format::tag(
                format!("{TOOL_CALL_START}\n"),
                Format::Tag(invokes.remove(0)),
                TOOL_CALL_END,
            ),
        };
        Ok(StructuralTag::new(format))
    }
}

/// One `<invoke>` of `tool`, with its arguments as keyed elements.
fn invoke_tag(tool: &FunctionToolParam, options: StructuralTagOptions) -> TagFormat {
    let syntax = M3Arguments::default();
    let arguments = arguments::arguments(
        &function_parameters(&tool.function),
        &syntax,
        &ArgumentOptions {
            any_order: options.any_order,
            max_whitespace_cnt: options.max_whitespace_cnt,
            excludes: vec![],
        },
    )
    .filter(|_| !syntax.recursive.get())
    // The parser ends the call at the first `</invoke>`, so no value holds it.
    .unwrap_or_else(|| Format::any_text_excluding(&[INVOKE_END]));
    TagFormat::new(
        format!("{INVOKE_START} name=\"{}\">", tool.function.name),
        arguments,
        format!("{INVOKE_END}\n"),
    )
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use serde_json::{Value, json};
    use xgrammar_structural_tag::builders::StructuralTagOptions;
    use xgrammar_structural_tag::{
        FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
    };

    use super::MinimaxM3StructuralTagBuilder;
    use crate::output_grammar::test_utils::outline;

    fn tool(definition: FunctionDefinition) -> ToolParam {
        ToolParam::Function(FunctionToolParam::new(definition))
    }

    fn check(tools: &[ToolParam], tool_choice: ToolChoice, parallel: bool, expected: Expect) {
        let tag = build_structural_tag(
            MinimaxM3StructuralTagBuilder,
            tools,
            tool_choice,
            StructuralTagOptions::default()
                .with_reasoning(false)
                .with_parallel_tool_calls(parallel),
        )
        .unwrap();
        expected.assert_eq(&outline(&tag.format));
    }

    fn order_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "user_id": { "type": "integer" },
                "note": { "type": "string" },
                "shipping": {
                    "type": "object",
                    "properties": { "city": { "type": "string" }, "zip": { "type": "integer" } },
                    "required": ["city"]
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": { "sku": { "type": "string" }, "qty": { "type": "integer" } },
                        "required": ["sku", "qty"]
                    }
                },
                "priority": { "anyOf": [{ "enum": ["low", "high"] }, { "type": "integer" }] }
            },
            "required": ["user_id", "items"]
        })
    }

    #[test]
    fn required_arguments_nest_as_keyed_elements() {
        check(
            &[tool(
                FunctionDefinition::new("create_order").with_parameters(order_schema()),
            )],
            ToolChoice::required(),
            true,
            expect![[r#"
                triggered_tags [`]<]minimax[>[<tool_call>`] excluding [`<mm:think>`, `</mm:think>`, `]<]minimax[>[</tool_call>`, `]<]minimax[>[<invoke`, `]<]minimax[>[</invoke>`] at_least_one
                  tag `]<]minimax[>[<tool_call>\n` .. `]<]minimax[>[</tool_call>`
                    tags_with_separator `` at_least_one
                      tag `]<]minimax[>[<invoke name="create_order">` .. `]<]minimax[>[</invoke>\n`
                        sequence
                          tag `]<]minimax[>[<user_id>` json(integer) `]<]minimax[>[</user_id>`
                          optional tag `]<]minimax[>[<note>` text excluding [`]<]minimax[>[`] `]<]minimax[>[</note>`
                          optional
                            tag `]<]minimax[>[<shipping>` .. `]<]minimax[>[</shipping>`
                              sequence
                                tag `]<]minimax[>[<city>` text excluding [`]<]minimax[>[`] `]<]minimax[>[</city>`
                                optional tag `]<]minimax[>[<zip>` json(integer) `]<]minimax[>[</zip>`
                          tag `]<]minimax[>[<items>` .. `]<]minimax[>[</items>`
                            star
                              tag `]<]minimax[>[<item>` .. `]<]minimax[>[</item>`
                                sequence
                                  tag `]<]minimax[>[<sku>` text excluding [`]<]minimax[>[`] `]<]minimax[>[</sku>`
                                  tag `]<]minimax[>[<qty>` json(integer) `]<]minimax[>[</qty>`
                          optional
                            tag `]<]minimax[>[<priority>` .. `]<]minimax[>[</priority>`
                              or
                                `low`
                                `high`
                                json(integer)
            "#]],
        );
    }

    #[test]
    fn undeclared_keys_fall_back_to_free_text() {
        let metadata = json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "labels": { "type": "object", "additionalProperties": { "type": "string" } }
            },
            "required": ["path"]
        });
        check(
            &[
                tool(FunctionDefinition::new("tag").with_parameters(metadata)),
                tool(
                    FunctionDefinition::new("search")
                        .with_parameters(json!({
                            "type": "object",
                            "properties": { "query": { "type": "string" } }
                        }))
                        .with_strict(false),
                ),
            ],
            ToolChoice::auto(),
            false,
            expect![[r#"
                triggered_tags [`]<]minimax[>[<tool_call>`] excluding [`<mm:think>`, `</mm:think>`, `]<]minimax[>[</tool_call>`, `]<]minimax[>[<invoke`, `]<]minimax[>[</invoke>`] stop_after_first
                  tag `]<]minimax[>[<tool_call>\n` .. `]<]minimax[>[</tool_call>`
                    tags_with_separator `` at_least_one stop_after_first
                      tag `]<]minimax[>[<invoke name="tag">` .. `]<]minimax[>[</invoke>\n`
                        sequence
                          tag `]<]minimax[>[<path>` text excluding [`]<]minimax[>[`] `]<]minimax[>[</path>`
                          optional tag `]<]minimax[>[<labels>` text excluding [`]<]minimax[>[</labels>`, `]<]minimax[>[</invoke>`] `]<]minimax[>[</labels>`
                      tag `]<]minimax[>[<invoke name="search">` text excluding [`]<]minimax[>[</invoke>`] `]<]minimax[>[</invoke>\n`
            "#]],
        );
    }

    #[test]
    fn values_nesting_in_themselves_make_the_call_free_text() {
        let tree = json!({
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "children": { "type": "array", "items": { "$ref": "#/$defs/node" } }
                    }
                }
            },
            "type": "object",
            "properties": { "root": { "$ref": "#/$defs/node" } },
            "required": ["root"]
        });
        check(
            &[tool(FunctionDefinition::new("plant").with_parameters(tree))],
            ToolChoice::required(),
            true,
            expect![[r#"
                triggered_tags [`]<]minimax[>[<tool_call>`] excluding [`<mm:think>`, `</mm:think>`, `]<]minimax[>[</tool_call>`, `]<]minimax[>[<invoke`, `]<]minimax[>[</invoke>`] at_least_one
                  tag `]<]minimax[>[<tool_call>\n` .. `]<]minimax[>[</tool_call>`
                    tags_with_separator `` at_least_one
                      tag `]<]minimax[>[<invoke name="plant">` text excluding [`]<]minimax[>[</invoke>`] `]<]minimax[>[</invoke>\n`
            "#]],
        );
    }

    #[test]
    fn forced_choice_wraps_only_the_named_tool() {
        let schema = json!({ "type": "object", "properties": { "q": { "type": "string" } } });
        check(
            &[
                tool(FunctionDefinition::new("search").with_parameters(schema.clone())),
                tool(FunctionDefinition::new("lookup").with_parameters(schema)),
            ],
            ToolChoice::function("lookup"),
            true,
            expect![[r#"
                tag `]<]minimax[>[<tool_call>\n` .. `]<]minimax[>[</tool_call>`
                  tag `]<]minimax[>[<invoke name="lookup">` .. `]<]minimax[>[</invoke>\n`
                    optional tag `]<]minimax[>[<q>` text excluding [`]<]minimax[>[`] `]<]minimax[>[</q>`
            "#]],
        );
    }
}
