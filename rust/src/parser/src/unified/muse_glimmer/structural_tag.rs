// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Whole-generation structural-tag grammar for Muse Glimmer.
//!
//! The tag covers the channel framing itself (the generation prompt ends with
//! `<|start|>assistant`, so the grammar starts at the first bare
//! ` to=<recipient><|message|>` header) and scopes any caller-provided JSON
//! schema to the `to=user` answer channel, so it can neither suppress the
//! framing nor leak into an ATEM tool channel.
//!
//! The turn grammar relies on `TagsWithSeparator` semantics verified against
//! xgrammar's `StructuralTagGrammarConverter::VisitSub` for
//! `TagsWithSeparatorFormat` (`cpp/structural_tag.cc` @ dd729e7, the pinned
//! xgrammar 0.2.4 revision): the format compiles to
//! `tags_rule (separator tags_rule)*` where every position is a fresh choice
//! over ALL tags, so one tag may repeat (multiple reasoning blocks, repeated
//! calls to one tool) and tags may appear in any order. `at_least_one`
//! controls whether the empty string is accepted; `stop_after_first` caps the
//! match at a single tag.
//!
//! ATEM value patterns use plain capturing groups only: xgrammar's regex
//! engine follows JavaScript syntax and rejects `(?...)` constructs.

use serde_json::{Map, Value, json};
use xgrammar_structural_tag::builders::{
    StructuralTagBuilder, StructuralTagContext, StructuralTagOptions,
};
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat, StructuralTag, TagFormat};
use xgrammar_structural_tag::tool::BuilderToolChoice;
use xgrammar_structural_tag::{Error as XgrammarError, Result as XgrammarResult};

use super::super::{ScopedStructuralTagBuilder, ScopedToolChoice};
use crate::tool::Tool;

pub(super) static MUSE_GLIMMER_STRUCTURAL_TAG_BUILDER: MuseGlimmerStructuralTagBuilder =
    MuseGlimmerStructuralTagBuilder;

// Channel framing markers, redefined locally: the parser core
// (`unified/muse_glimmer.rs`) owns its own copies and the two must not be
// coupled while both evolve.
const CHANNEL_SEPARATOR: &str = "<|start|>assistant";
const CHANNEL_START: &str = "<|start|>";
const END_OF_MESSAGE: &str = "<|eom|>";
const END_OF_TURN: &str = "<|eot|>";
const REASONING_BEGIN: &str = " to=self<|message|>";
const ANSWER_BEGIN: &str = " to=user<|message|>";
const INVOKE_CLOSE: &str = "</atem:invoke>";
const PARAMETER_CLOSE: &str = "</atem:parameter>";

const INTEGER_PATTERN: &str = "-?(0|[1-9][0-9]*)";
const NUMBER_PATTERN: &str = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?";
const BOOLEAN_PATTERN: &str = "true|false";

/// Muse Glimmer structural-tag builder.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MuseGlimmerStructuralTagBuilder;

impl ScopedStructuralTagBuilder for MuseGlimmerStructuralTagBuilder {
    fn build_scoped(
        &self,
        tools: &[Tool],
        tool_choice: Option<ScopedToolChoice>,
        caller_schema: Option<&serde_json::Value>,
        options: &StructuralTagOptions,
    ) -> XgrammarResult<StructuralTag> {
        match tool_choice {
            // Reasoning, answer, and tool channels are all optional, in any
            // order; an empty generation stays valid.
            None | Some(ScopedToolChoice::Auto) => {
                let mut tags = vec![reasoning_tag()];
                for tool in tools {
                    tags.extend(tool_tags(tool));
                }
                Ok(StructuralTag::new(Format::sequence(vec![
                    Format::tags_with_separator(tags, CHANNEL_SEPARATOR, false, false),
                    // The answer channel, when present, is always LAST.
                    // `<|eot|>` is a pure stop token (never grammar content),
                    // and the model was never trained to emit it right after
                    // `<|eom|>` — if the answer could be followed by more
                    // channels, the model keeps re-opening answer channels
                    // forever instead of stopping.
                    Format::optional(Format::sequence(vec![
                        Format::optional(Format::const_string(CHANNEL_SEPARATOR)),
                        Format::Tag(answer_tag(caller_schema, options)),
                    ])),
                ])))
            }
            Some(ScopedToolChoice::Required) => required_turn(tools),
            Some(ScopedToolChoice::Function(name)) => {
                let tool = tools
                    .iter()
                    .find(|tool| tool.name == *name)
                    .ok_or_else(|| XgrammarError::ToolNotFound { name: name.clone() })?;
                required_turn(std::slice::from_ref(tool))
            }
        }
    }
}

impl StructuralTagBuilder for MuseGlimmerStructuralTagBuilder {
    fn build(&self, ctx: StructuralTagContext<'_>) -> XgrammarResult<StructuralTag> {
        // Muse Glimmer has no builtin tools; only function tools are scoped in.
        let tools = ctx
            .function_tools
            .iter()
            .map(|tool| Tool {
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone().unwrap_or_else(|| json!(true)),
                strict: tool.function.strict,
            })
            .collect::<Vec<_>>();
        let tool_choice = match ctx.tool_choice {
            BuilderToolChoice::Auto => Some(ScopedToolChoice::Auto),
            BuilderToolChoice::Required => Some(ScopedToolChoice::Required),
            BuilderToolChoice::Forced => Some(ScopedToolChoice::Function(
                ctx.function_tools
                    .first()
                    .map(|tool| tool.function.name.clone())
                    .ok_or(XgrammarError::ForcedToolChoiceInvalid { count: 0 })?,
            )),
        };
        self.build_scoped(&tools, tool_choice, None, &ctx.options)
    }
}

/// `reasoning* tool+`: at least one tool call, with any reasoning before it.
fn required_turn(tools: &[Tool]) -> XgrammarResult<StructuralTag> {
    if tools.is_empty() {
        return Err(XgrammarError::RequiredWithoutTools);
    }
    let tool_tags = tools.iter().flat_map(tool_tags).collect();
    Ok(StructuralTag::new(Format::sequence(vec![
        Format::star(Format::sequence(vec![
            Format::Tag(reasoning_tag()),
            Format::const_string(CHANNEL_SEPARATOR),
        ])),
        Format::tags_with_separator(tool_tags, CHANNEL_SEPARATOR, true, false),
    ])))
}

/// ` to=self<|message|>...<|eom|>`. Reasoning is always permitted (zero or
/// more blocks per turn), so `StructuralTagOptions::reasoning` is not
/// consulted.
fn reasoning_tag() -> TagFormat {
    TagFormat::new(
        REASONING_BEGIN,
        Format::any_text_excluding(&[END_OF_MESSAGE, END_OF_TURN, CHANNEL_START]),
        END_OF_MESSAGE,
    )
}

/// ` to=user<|message|>...<|eom|>`. A caller schema constrains this body only.
///
/// ALL channels end with `<|eom|>` in the grammar, never `<|eot|>`: vLLM runs
/// the matcher with `override_stop_tokens`, which intercepts stop tokens
/// anywhere and rejects them when the matcher cannot terminate — and a
/// channel-sequence grammar only reaches a terminable state after a tag's end
/// boundary, so an `<|eot|>` consumed as a tag end would 500 the request.
/// With `<|eom|>`-only ends, the model closes the final channel with
/// `<|eom|>` and then emits `<|eot|>` purely as the stop token.
fn answer_tag(caller_schema: Option<&Value>, options: &StructuralTagOptions) -> TagFormat {
    let content = match caller_schema {
        Some(schema) => Format::JsonSchema(
            JsonSchemaFormat::new(schema.clone())
                .with_any_order(options.any_order)
                .with_max_whitespace_cnt(options.max_whitespace_cnt),
        ),
        None => Format::any_text_excluding(&[END_OF_TURN, END_OF_MESSAGE, CHANNEL_START]),
    };
    TagFormat::new(ANSWER_BEGIN, content, END_OF_MESSAGE)
}

/// Tool channel tags: one begin variant per model-known recipient spelling.
/// The recipient is the registered tool name verbatim, but for a dot-less
/// name `ns` the model is also known to emit the doubled `ns.ns`.
fn tool_tags(tool: &Tool) -> Vec<TagFormat> {
    let content = tool_channel_content(tool);
    let mut begins = vec![format!(" to={}<|message|>", tool.name)];
    if !tool.name.contains('.') {
        begins.push(format!(" to={0}.{0}<|message|>", tool.name));
    }
    begins
        .into_iter()
        // `<|eom|>`-only ends: see `answer_tag` for why `<|eot|>` must stay
        // out of the grammar.
        .map(|begin| TagFormat::new(begin, content.clone(), END_OF_MESSAGE))
        .collect()
}

/// The ATEM body of a tool channel, whitespace-exact as the chat template
/// renders it. A tool with no declared parameters or `strict: false` keeps
/// the channel and invoke framing but leaves the invoke body free-form.
fn tool_channel_content(tool: &Tool) -> Format {
    if tool.strict != Some(false)
        && let Some(properties) = tool.parameters.get("properties").and_then(Value::as_object)
        && !properties.is_empty()
    {
        return typed_invokes(&tool.name, properties, required_names(&tool.parameters));
    }
    Format::sequence(vec![
        Format::const_string(format!(
            "<atem:function_calls>\n<atem:invoke name=\"{}\">\n",
            tool.name
        )),
        Format::any_text_excluding(&[INVOKE_CLOSE]),
        Format::const_string("\n</atem:invoke>\n</atem:function_calls>"),
    ])
}

/// One or more typed invokes: `invoke (\n invoke)*`, newline-separated.
fn typed_invokes(name: &str, properties: &Map<String, Value>, required: Vec<&str>) -> Format {
    let invoke = || typed_invoke(name, properties, &required);
    Format::sequence(vec![
        Format::const_string("<atem:function_calls>\n"),
        invoke(),
        Format::star(Format::sequence(vec![Format::const_string("\n"), invoke()])),
        Format::const_string("\n</atem:function_calls>"),
    ])
}

/// One `<atem:invoke>` block. Required properties come first in schema order;
/// each parameter carries its trailing newline, so a call with no arguments
/// stays `<atem:invoke name="N">\n</atem:invoke>`.
fn typed_invoke(name: &str, properties: &Map<String, Value>, required: &[&str]) -> Format {
    let mut elements = vec![Format::const_string(format!(
        "<atem:invoke name=\"{name}\">\n"
    ))];
    let (required_props, optional_props): (Vec<_>, Vec<_>) =
        properties.iter().partition(|(key, _)| required.contains(&key.as_str()));
    for (key, schema) in required_props {
        elements.push(parameter_line(key, schema));
    }
    for (key, schema) in optional_props {
        elements.push(Format::optional(parameter_line(key, schema)));
    }
    elements.push(Format::const_string(INVOKE_CLOSE));
    Format::sequence(elements)
}

/// One parameter plus the newline the template emits after it.
fn parameter_line(key: &str, schema: &Value) -> Format {
    Format::sequence(vec![parameter(key, schema), Format::const_string("\n")])
}

/// `<atem:parameter name="KEY">VALUE</atem:parameter>` with a typed value.
fn parameter(key: &str, schema: &Value) -> Format {
    Format::sequence(vec![
        Format::const_string(format!("<atem:parameter name=\"{key}\">")),
        parameter_value(schema),
        Format::const_string(PARAMETER_CLOSE),
    ])
}

/// Value grammar for one parameter, by JSON-schema type. Strings, objects,
/// arrays, and unknown schemas stay free-form (the template renders objects
/// and arrays as JSON text); scalars get exact patterns.
fn parameter_value(schema: &Value) -> Format {
    if let Some(pattern) = scalar_enum_pattern(schema) {
        return Format::regex(pattern);
    }
    let text = || Format::any_text_excluding(&[PARAMETER_CLOSE]);
    let Some(json_type) = schema.get("type").and_then(Value::as_str) else {
        return text();
    };
    match json_type {
        "integer" => Format::regex(INTEGER_PATTERN),
        "number" => Format::regex(NUMBER_PATTERN),
        "boolean" => Format::regex(BOOLEAN_PATTERN),
        "null" => Format::const_string("null"),
        _ => text(),
    }
}

/// A scalar `enum` becomes a regex alternation of its escaped literals.
/// Non-scalar values, markers, and huge lists stay free-form.
fn scalar_enum_pattern(schema: &Value) -> Option<String> {
    let values = schema.get("enum").and_then(Value::as_array)?;
    if values.is_empty() || values.len() > 256 {
        return None;
    }
    let mut literals = Vec::with_capacity(values.len());
    for value in values {
        let literal = match value {
            Value::String(string) => string.clone(),
            Value::Number(number) => number.to_string(),
            Value::Bool(boolean) => boolean.to_string(),
            _ => return None,
        };
        if literal.contains(PARAMETER_CLOSE) || literal.contains(CHANNEL_START) {
            return None;
        }
        literals.push(escape_regex(&literal));
    }
    match literals.as_slice() {
        [literal] => Some(literal.clone()),
        _ => Some(format!("({})", literals.join("|"))),
    }
}

/// Escape the ASCII characters that are special in xgrammar's
/// JavaScript-flavored regex syntax.
fn escape_regex(literal: &str) -> String {
    let mut escaped = String::with_capacity(literal.len());
    for ch in literal.chars() {
        if "\\^$.|?*+()[]{}".contains(ch) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

fn required_names(parameters: &Value) -> Vec<&str> {
    parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;
    use xgrammar_structural_tag::builders::StructuralTagOptions;
    use xgrammar_structural_tag::{
        FunctionDefinition, FunctionToolParam, ToolChoice, ToolParam, build_structural_tag,
    };

    use super::{
        MuseGlimmerStructuralTagBuilder, ScopedStructuralTagBuilder, ScopedToolChoice, Tool,
    };

    fn tool(name: &str, parameters: serde_json::Value) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters,
            strict: None,
        }
    }

    fn loose_tool(name: &str) -> Tool {
        Tool {
            strict: Some(false),
            ..tool(name, json!({"type": "object"}))
        }
    }

    #[test]
    fn caller_schema_scopes_to_answer_channel() {
        let tag = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &[],
                None,
                Some(&json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"]
                })),
                &StructuralTagOptions::default(),
            )
            .unwrap();
        let json = tag.to_json_string().unwrap();

        assert!(json.contains(r#""begin":" to=user<|message|>""#));
        assert!(json.contains(r#""begin":" to=self<|message|>""#));
        assert!(json.contains(r#""type":"json_schema""#));
        assert!(!json.contains("atem:invoke"));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"tags_with_separator","tags":[{"begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":false,"stop_after_first":false},{"type":"optional","content":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|start|>assistant"}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"<|eom|>"}]}}]}}"#]].assert_eq(&json);
    }

    #[test]
    fn no_tools_no_schema_allows_empty_and_bare_channels() {
        let tag = MuseGlimmerStructuralTagBuilder
            .build_scoped(&[], None, None, &StructuralTagOptions::default())
            .unwrap();
        let json = tag.to_json_string().unwrap();

        assert!(json.contains(r#""begin":" to=self<|message|>""#));
        assert!(json.contains(r#""at_least_one":false"#));
        assert!(!json.contains("json_schema"));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"tags_with_separator","tags":[{"begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":false,"stop_after_first":false},{"type":"optional","content":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|start|>assistant"}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"any_text","excludes":["<|eot|>","<|eom|>","<|start|>"]},"end":"<|eom|>"}]}}]}}"#]].assert_eq(&json);
    }

    #[test]
    fn auto_with_typed_and_loose_tools_matches_atem_bytes() {
        let tools = vec![
            tool(
                "get_weather",
                json!({
                    "type": "object",
                    "properties": {
                        "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
                        "city": { "type": "string" },
                        "days": { "type": "integer" }
                    },
                    "required": ["city"]
                }),
            ),
            loose_tool("loose"),
        ];
        let tag = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Auto),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap();
        let json = tag.to_json_string().unwrap();

        // Typed tool: exact wrapper bytes with the newline inside each const.
        assert!(json.contains(r#""value":"<atem:function_calls>\n""#));
        assert!(json.contains(r#""value":"<atem:invoke name=\"get_weather\">\n""#));
        // Loose tool: permissive body keeps channel + invoke framing.
        assert!(json.contains(r#"<atem:function_calls>\n<atem:invoke name=\"loose\">\n"#));
        assert!(json.contains(r#""begin":" to=get_weather.get_weather<|message|>""#));
        assert!(json.contains(r#""begin":" to=loose.loose<|message|>""#));
        assert!(json.contains(r#"(celsius|fahrenheit)"#));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"tags_with_separator","tags":[{"begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"},{"begin":" to=get_weather<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"get_weather\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"regex","pattern":"(celsius|fahrenheit)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"regex","pattern":"-?(0|[1-9][0-9]*)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"const_string","value":"</atem:invoke>"}]},{"type":"star","content":{"type":"sequence","elements":[{"type":"const_string","value":"\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"get_weather\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"regex","pattern":"(celsius|fahrenheit)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"regex","pattern":"-?(0|[1-9][0-9]*)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"const_string","value":"</atem:invoke>"}]}]}},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=get_weather.get_weather<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"get_weather\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"regex","pattern":"(celsius|fahrenheit)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"regex","pattern":"-?(0|[1-9][0-9]*)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"const_string","value":"</atem:invoke>"}]},{"type":"star","content":{"type":"sequence","elements":[{"type":"const_string","value":"\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"get_weather\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"regex","pattern":"(celsius|fahrenheit)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"regex","pattern":"-?(0|[1-9][0-9]*)"},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"const_string","value":"</atem:invoke>"}]}]}},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=loose<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n<atem:invoke name=\"loose\">\n"},{"type":"any_text","excludes":["</atem:invoke>"]},{"type":"const_string","value":"\n</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=loose.loose<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n<atem:invoke name=\"loose\">\n"},{"type":"any_text","excludes":["</atem:invoke>"]},{"type":"const_string","value":"\n</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":false,"stop_after_first":false},{"type":"optional","content":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|start|>assistant"}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"any_text","excludes":["<|eot|>","<|eom|>","<|start|>"]},"end":"<|eom|>"}]}}]}}"#]].assert_eq(&json);
    }

    #[test]
    fn required_turn_demands_at_least_one_tool_call() {
        let tools = vec![tool(
            "search",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        )];
        let tag = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Required),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap();
        let json = tag.to_json_string().unwrap();

        assert!(json.contains(r#""at_least_one":true"#));
        assert!(json.contains(r#""separator":"<|start|>assistant""#));
        assert!(!json.contains(" to=user<|message|>"));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"star","content":{"type":"sequence","elements":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"},{"type":"const_string","value":"<|start|>assistant"}]}},{"type":"tags_with_separator","tags":[{"begin":" to=search<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"search\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]},{"type":"star","content":{"type":"sequence","elements":[{"type":"const_string","value":"\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"search\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]}]}},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=search.search<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"search\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]},{"type":"star","content":{"type":"sequence","elements":[{"type":"const_string","value":"\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"search\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]}]}},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":true,"stop_after_first":false}]}}"#]].assert_eq(&json);
    }

    #[test]
    fn function_choice_keeps_only_the_named_tool() {
        let tools = vec![
            tool("search", json!({"type": "object"})),
            tool("lookup", json!({"type": "object"})),
            tool("my.ns", json!({"type": "object"})),
        ];
        let options = StructuralTagOptions::default();

        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Function("lookup".to_string())),
                None,
                &options,
            )
            .unwrap()
            .to_json_string()
            .unwrap();
        assert!(json.contains(r#""begin":" to=lookup.lookup<|message|>""#));
        assert!(!json.contains("search"));
        assert!(!json.contains("my.ns"));

        // A dotted name has exactly the verbatim begin variant.
        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Function("my.ns".to_string())),
                None,
                &options,
            )
            .unwrap()
            .to_json_string()
            .unwrap();
        assert!(json.contains(r#""begin":" to=my.ns<|message|>""#));
        assert!(!json.contains("my.ns.my.ns"));

        let error = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Function("missing".to_string())),
                None,
                &options,
            )
            .unwrap_err();
        assert!(
            matches!(error, xgrammar_structural_tag::Error::ToolNotFound { name } if name == "missing")
        );
    }

    #[test]
    fn legacy_build_maps_choices_and_tools() {
        let tools = vec![ToolParam::Function(FunctionToolParam::new(
            FunctionDefinition::new("ping").with_parameters(json!({
                "type": "object",
                "properties": { "host": { "type": "string" } },
                "required": ["host"]
            })),
        ))];
        let tag = build_structural_tag(
            MuseGlimmerStructuralTagBuilder,
            &tools,
            ToolChoice::required(),
            StructuralTagOptions::default(),
        )
        .unwrap();
        let json = tag.to_json_string().unwrap();

        assert!(json.contains(r#""begin":" to=ping.ping<|message|>""#));
        assert!(json.contains(r#""at_least_one":true"#));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"star","content":{"type":"sequence","elements":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"},{"type":"const_string","value":"<|start|>assistant"}]}},{"type":"tags_with_separator","tags":[{"begin":" to=ping<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"ping\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"host\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]},{"type":"star","content":{"type":"sequence","elements":[{"type":"const_string","value":"\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"ping\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"host\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]}]}},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=ping.ping<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"ping\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"host\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]},{"type":"star","content":{"type":"sequence","elements":[{"type":"const_string","value":"\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\"ping\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"host\">"},{"type":"any_text","excludes":["</atem:parameter>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"const_string","value":"</atem:invoke>"}]}]}},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":true,"stop_after_first":false}]}}"#]].assert_eq(&json);
    }
}
