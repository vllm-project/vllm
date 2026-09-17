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
//! The turn grammar and the invoke repetition inside a typed tool channel
//! rely on `TagsWithSeparator` semantics verified against
//! xgrammar's `StructuralTagGrammarConverter::VisitSub` for
//! `TagsWithSeparatorFormat` (`cpp/structural_tag.cc` @ dd729e7, the pinned
//! xgrammar 0.2.4 revision): the format compiles to
//! `tags_rule (separator tags_rule)*` where every position is a fresh choice
//! over ALL tags, so one tag may repeat (multiple reasoning blocks, repeated
//! calls to one tool) and tags may appear in any order. `at_least_one`
//! controls whether the empty string is accepted; `stop_after_first` caps the
//! match at a single tag.

use serde_json::{Map, Value};
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat, StructuralTag, TagFormat};
use xgrammar_structural_tag::{Error as XgrammarError, Result as XgrammarResult};

use super::super::{ScopedStructuralTagBuilder, ScopedToolChoice};
use super::{EOM, EOT, INVOKE_CLOSE, PARAMETER_CLOSE, START};
use crate::tool::Tool;

pub(super) static MUSE_GLIMMER_STRUCTURAL_TAG_BUILDER: MuseGlimmerStructuralTagBuilder =
    MuseGlimmerStructuralTagBuilder;

const CHANNEL_SEPARATOR: &str = "<|start|>assistant";
const REASONING_BEGIN: &str = " to=self<|message|>";
const ANSWER_BEGIN: &str = " to=user<|message|>";

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
            // Tools absent or disabled (`"none"`): the grammar must not
            // sanction tool channels the request forbade.
            None => auto_turn(&[], caller_schema, options),
            Some(ScopedToolChoice::Auto) => auto_turn(tools, caller_schema, options),
            Some(ScopedToolChoice::Required) => required_turn(tools, options),
            Some(ScopedToolChoice::Function(name)) => {
                let tool = tools
                    .iter()
                    .find(|tool| tool.name == *name)
                    .ok_or_else(|| XgrammarError::ToolNotFound { name: name.clone() })?;
                required_turn(std::slice::from_ref(tool), options)
            }
        }
    }
}

/// Reasoning, answer, and tool channels in any order; an empty generation
/// stays valid unless a caller schema must be honored.
fn auto_turn(
    tools: &[Tool],
    caller_schema: Option<&Value>,
    options: &StructuralTagOptions,
) -> XgrammarResult<StructuralTag> {
    validate_tool_names(tools)?;
    let mut tags = vec![reasoning_tag()];
    for tool in tools {
        tags.extend(tool_tags(tool, options));
    }
    // The answer channel, when present, is always LAST. `<|eot|>` is a pure
    // stop token (never grammar content), and the model was never trained to
    // emit it right after `<|eom|>` — if the answer could be followed by more
    // channels, the model keeps re-opening answer channels forever instead of
    // stopping.
    let answer = Format::sequence(vec![
        Format::optional(Format::const_string(CHANNEL_SEPARATOR)),
        Format::Tag(answer_tag(caller_schema, options)),
    ]);
    // Without tool channels, a caller schema makes the answer mandatory:
    // `response_format` must guarantee schema-conforming output. With tool
    // channels it stays optional — a turn may end on a tool call instead.
    let answer = if caller_schema.is_some() && tools.is_empty() {
        answer
    } else {
        Format::optional(answer)
    };
    Ok(StructuralTag::new(Format::sequence(vec![
        Format::tags_with_separator(tags, CHANNEL_SEPARATOR, false, false),
        answer,
    ])))
}

/// Reject tool names the streaming parser could not round-trip as channel
/// recipients (`[A-Za-z0-9_.\-]+`): the grammar interpolates them verbatim
/// into channel begins and invoke wrappers.
fn validate_tool_names(tools: &[Tool]) -> XgrammarResult<()> {
    for tool in tools {
        let valid = !tool.name.is_empty()
            && tool
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
        if !valid {
            return Err(XgrammarError::Custom(
                format!(
                    "tool name {:?} cannot be a Muse Glimmer channel recipient",
                    tool.name
                )
                .into(),
            ));
        }
    }
    Ok(())
}

/// `reasoning* tool+`: at least one tool call, with any reasoning before it.
fn required_turn(tools: &[Tool], options: &StructuralTagOptions) -> XgrammarResult<StructuralTag> {
    if tools.is_empty() {
        return Err(XgrammarError::RequiredWithoutTools);
    }
    validate_tool_names(tools)?;
    let tool_tags = tools.iter().flat_map(|tool| tool_tags(tool, options)).collect();
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
        Format::any_text_excluding(&[EOM, EOT, START]),
        EOM,
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
        Some(schema) => json_schema(schema.clone(), options),
        None => Format::any_text_excluding(&[EOT, EOM, START]),
    };
    TagFormat::new(ANSWER_BEGIN, content, EOM)
}

/// Tool channel tags: one begin variant per model-known recipient spelling.
/// The recipient is the registered tool name verbatim, but for a dot-less
/// name `ns` the model is also known to emit the doubled `ns.ns`.
fn tool_tags(tool: &Tool, options: &StructuralTagOptions) -> Vec<TagFormat> {
    let content = tool_channel_content(tool, options);
    let mut begins = vec![format!(" to={}<|message|>", tool.name)];
    if !tool.name.contains('.') {
        begins.push(format!(" to={0}.{0}<|message|>", tool.name));
    }
    begins
        .into_iter()
        // `<|eom|>`-only ends: see `answer_tag` for why `<|eot|>` must stay
        // out of the grammar.
        .map(|begin| TagFormat::new(begin, content.clone(), EOM))
        .collect()
}

/// The ATEM body of a tool channel, whitespace-exact as the chat template
/// renders it. A tool with no declared parameters, `strict: false`, or a
/// schema the fixed-order typed encoding cannot express faithfully keeps the
/// channel and invoke framing but leaves the invoke body free-form.
fn tool_channel_content(tool: &Tool, options: &StructuralTagOptions) -> Format {
    if tool.strict != Some(false)
        && let Some(properties) = tool.parameters.get("properties").and_then(Value::as_object)
        && !properties.is_empty()
        && typed_encoding_is_faithful(&tool.parameters, properties)
    {
        return typed_invokes(
            &tool.name,
            properties,
            required_names(&tool.parameters),
            options,
        );
    }
    Format::sequence(vec![
        Format::const_string(format!(
            "<atem:function_calls>\n<atem:invoke name=\"{}\">\n",
            tool.name
        )),
        Format::any_text_excluding(&[INVOKE_CLOSE, EOM, EOT, START]),
        Format::const_string("\n</atem:invoke>\n</atem:function_calls>"),
    ])
}

/// Whether the typed encoding can express every schema-valid call: no
/// extra properties admitted (an explicit `additionalProperties`/
/// `patternProperties` allowance cannot be rendered), every `required` name
/// declared, and every key safe inside `<atem:parameter name="…">`. Anything
/// else stays free-form rather than silently narrowing the schema.
fn typed_encoding_is_faithful(parameters: &Value, properties: &Map<String, Value>) -> bool {
    let extra_properties_allowed = match parameters.get("additionalProperties") {
        None | Some(Value::Bool(false)) => false,
        Some(_) => true,
    };
    !extra_properties_allowed
        && parameters.get("patternProperties").is_none()
        && required_names(parameters).iter().all(|name| properties.contains_key(*name))
        && properties.keys().all(|key| !key.contains(['"', '<']))
}

/// One or more typed invokes, newline-separated. The repetition is expressed
/// with `TagsWithSeparator` (`invoke (\n invoke)*`) so the invoke tree is
/// serialized once per channel rather than cloned into a `star`.
fn typed_invokes(
    name: &str,
    properties: &Map<String, Value>,
    required: Vec<&str>,
    options: &StructuralTagOptions,
) -> Format {
    Format::sequence(vec![
        Format::const_string("<atem:function_calls>\n"),
        Format::tags_with_separator(
            vec![typed_invoke(name, properties, &required, options)],
            "\n",
            true,
            false,
        ),
        Format::const_string("\n</atem:function_calls>"),
    ])
}

/// One `<atem:invoke>` block. Required properties come first in schema order;
/// each parameter carries its trailing newline, so a call with no arguments
/// stays `<atem:invoke name="N">\n</atem:invoke>`.
fn typed_invoke(
    name: &str,
    properties: &Map<String, Value>,
    required: &[&str],
    options: &StructuralTagOptions,
) -> TagFormat {
    let (required_props, optional_props): (Vec<_>, Vec<_>) =
        properties.iter().partition(|(key, _)| required.contains(&key.as_str()));
    let mut lines = Vec::with_capacity(properties.len());
    for (key, schema) in required_props {
        lines.push(parameter_line(key, schema, options));
    }
    for (key, schema) in optional_props {
        lines.push(Format::optional(parameter_line(key, schema, options)));
    }
    TagFormat::new(
        format!("<atem:invoke name=\"{name}\">\n"),
        Format::sequence(lines),
        INVOKE_CLOSE,
    )
}

/// One parameter plus the newline the template emits after it.
fn parameter_line(key: &str, schema: &Value, options: &StructuralTagOptions) -> Format {
    Format::sequence(vec![
        parameter(key, schema, options),
        Format::const_string("\n"),
    ])
}

/// `<atem:parameter name="KEY">VALUE</atem:parameter>` with a typed value.
fn parameter(key: &str, schema: &Value, options: &StructuralTagOptions) -> Format {
    Format::sequence(vec![
        Format::const_string(format!("<atem:parameter name=\"{key}\">")),
        parameter_value(schema, options),
        Format::const_string(PARAMETER_CLOSE),
    ])
}

/// Value grammar for one parameter, by JSON-schema type. Strings, objects,
/// arrays, and unknown schemas stay free-form (the template renders objects
/// and arrays as JSON text); the other scalars reuse xgrammar's JSON grammar
/// over the parameter schema, whose single `type` is already the narrowing,
/// so facets such as `minimum` keep applying.
fn parameter_value(schema: &Value, options: &StructuralTagOptions) -> Format {
    if let Some(alternation) = scalar_enum(schema) {
        return alternation;
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("integer" | "number" | "boolean" | "null") => json_schema(schema.clone(), options),
        // Framing markers and the invoke close stay excluded: the streaming
        // parser cuts the invoke body at the first `</atem:invoke>` and treats
        // quoted framing as a channel boundary, so the grammar must never
        // force bytes the parser cannot round-trip.
        _ => Format::any_text_excluding(&[PARAMETER_CLOSE, INVOKE_CLOSE, EOM, EOT, START]),
    }
}

/// A scalar `enum` becomes an alternation of its literals as const strings.
/// Non-scalar values, markers, and huge lists stay free-form.
fn scalar_enum(schema: &Value) -> Option<Format> {
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
        if [PARAMETER_CLOSE, INVOKE_CLOSE, EOM, EOT, START]
            .iter()
            .any(|marker| literal.contains(marker))
        {
            return None;
        }
        literals.push(literal);
    }
    Some(match literals.as_slice() {
        [literal] => Format::const_string(literal.clone()),
        _ => Format::or(literals.into_iter().map(Format::const_string).collect()),
    })
}

/// A JSON body honoring the request's key-order and whitespace options.
fn json_schema(schema: Value, options: &StructuralTagOptions) -> Format {
    Format::JsonSchema(
        JsonSchemaFormat::new(schema)
            .with_any_order(options.any_order)
            .with_max_whitespace_cnt(options.max_whitespace_cnt),
    )
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

    use super::super::{ASSISTANT, MESSAGE, START};
    use super::{
        ANSWER_BEGIN, CHANNEL_SEPARATOR, MuseGlimmerStructuralTagBuilder, REASONING_BEGIN,
        ScopedStructuralTagBuilder, ScopedToolChoice, Tool,
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
    fn composite_markers_are_built_from_shared_parts() {
        assert_eq!(CHANNEL_SEPARATOR, format!("{START}{ASSISTANT}"));
        assert_eq!(REASONING_BEGIN, format!(" to=self{MESSAGE}"));
        assert_eq!(ANSWER_BEGIN, format!(" to=user{MESSAGE}"));
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
        // With no tool channels the schema-bearing answer is mandatory.
        assert!(json.contains(r#""stop_after_first":false},{"type":"sequence""#));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"tags_with_separator","tags":[{"begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":false,"stop_after_first":false},{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|start|>assistant"}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"<|eom|>"}]}]}}"#]].assert_eq(&json);
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

        // Typed tool: exact wrapper bytes with the newline inside each const,
        // and the invoke tree serialized once per channel.
        assert!(json.contains(r#""value":"<atem:function_calls>\n""#));
        assert!(json.contains(r#""begin":"<atem:invoke name=\"get_weather\">\n""#));
        assert_eq!(json.matches(r#"<atem:parameter name=\"city\">"#).count(), 2);
        // Loose tool: permissive body keeps channel + invoke framing.
        assert!(json.contains(r#"<atem:function_calls>\n<atem:invoke name=\"loose\">\n"#));
        assert!(json.contains(r#""begin":" to=get_weather.get_weather<|message|>""#));
        assert!(json.contains(r#""begin":" to=loose.loose<|message|>""#));
        assert!(json.contains(
            r#"{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]}"#
        ));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"tags_with_separator","tags":[{"begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"},{"begin":" to=get_weather<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"begin":"<atem:invoke name=\"get_weather\">\n","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"json_schema","json_schema":{"type":"integer"},"style":"json","any_order":false,"max_whitespace_cnt":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=get_weather.get_weather<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"begin":"<atem:invoke name=\"get_weather\">\n","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"json_schema","json_schema":{"type":"integer"},"style":"json","any_order":false,"max_whitespace_cnt":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=loose<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n<atem:invoke name=\"loose\">\n"},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>"]},{"type":"const_string","value":"\n</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=loose.loose<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n<atem:invoke name=\"loose\">\n"},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>"]},{"type":"const_string","value":"\n</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":false,"stop_after_first":false},{"type":"optional","content":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|start|>assistant"}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"any_text","excludes":["<|eot|>","<|eom|>","<|start|>"]},"end":"<|eom|>"}]}}]}}"#]].assert_eq(&json);
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
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"star","content":{"type":"sequence","elements":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>"]},"end":"<|eom|>"},{"type":"const_string","value":"<|start|>assistant"}]}},{"type":"tags_with_separator","tags":[{"begin":" to=search<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"begin":"<atem:invoke name=\"search\">\n","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"begin":" to=search.search<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"begin":"<atem:invoke name=\"search\">\n","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>"]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":true,"stop_after_first":false}]}}"#]].assert_eq(&json);
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
    fn none_tool_choice_generates_no_tool_channels() {
        let tools = vec![tool("search", json!({"type": "object"}))];
        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                None,
                Some(&json!({"type": "object"})),
                &StructuralTagOptions::default(),
            )
            .unwrap()
            .to_json_string()
            .unwrap();

        assert!(!json.contains(" to=search"));
        assert!(json.contains(r#""begin":" to=user<|message|>""#));
    }

    #[test]
    fn unfaithful_schemas_keep_invoke_body_free_form() {
        let extra_properties = tool(
            "open",
            json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "additionalProperties": true
            }),
        );
        let undeclared_required = tool(
            "lookup",
            json!({
                "type": "object",
                "properties": { "a": { "type": "string" } },
                "required": ["a", "b"]
            }),
        );

        for tool in [extra_properties, undeclared_required] {
            let json = MuseGlimmerStructuralTagBuilder
                .build_scoped(
                    std::slice::from_ref(&tool),
                    Some(ScopedToolChoice::Required),
                    None,
                    &StructuralTagOptions::default(),
                )
                .unwrap()
                .to_json_string()
                .unwrap();
            assert!(!json.contains("<atem:parameter"), "{}", tool.name);
        }
    }

    #[test]
    fn tool_name_outside_recipient_charset_is_rejected() {
        let tools = vec![tool("my tool", json!({"type": "object"}))];
        let error = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Auto),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap_err();

        assert!(matches!(error, xgrammar_structural_tag::Error::Custom(_)));
    }

    #[test]
    fn scalar_enum_members_become_const_string_alternation() {
        let format = super::parameter_value(
            &json!({"enum": ["a.b", 1, true]}),
            &StructuralTagOptions::default(),
        );

        assert_eq!(
            serde_json::to_value(format).unwrap(),
            json!({"type": "or", "elements": [
                {"type": "const_string", "value": "a.b"},
                {"type": "const_string", "value": "1"},
                {"type": "const_string", "value": "true"}
            ]})
        );
    }

    #[test]
    fn single_value_enum_becomes_const_string() {
        let format = super::parameter_value(
            &json!({"type": "string", "enum": ["only"]}),
            &StructuralTagOptions::default(),
        );

        assert_eq!(
            serde_json::to_value(format).unwrap(),
            json!({"type": "const_string", "value": "only"})
        );
    }

    #[test]
    fn typed_scalar_parameter_keeps_schema_facets() {
        let format = super::parameter_value(
            &json!({"type": "integer", "minimum": 1}),
            &StructuralTagOptions::default(),
        );

        assert_eq!(
            serde_json::to_value(format).unwrap(),
            json!({
                "type": "json_schema",
                "json_schema": {"type": "integer", "minimum": 1},
                "style": "json",
                "any_order": false,
                "max_whitespace_cnt": null
            })
        );
    }
}
