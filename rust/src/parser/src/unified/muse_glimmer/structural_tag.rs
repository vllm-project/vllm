// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Whole-generation structural-tag grammar for Muse Glimmer.
//!
//! The tag covers the channel framing itself (the generation prompt ends with
//! `<|start|>assistant`, so the grammar starts at the first bare
//! ` to=<recipient><|message|>` header) and scopes any caller-provided
//! constraint to the `to=user` answer channel, so it can neither suppress
//! the framing nor leak into an ATEM tool channel.
//!
//! The turn grammar and the invoke repetition inside a typed tool channel
//! rely on `TagsWithSeparator` semantics verified against
//! xgrammar's `StructuralTagGrammarConverter::VisitSub` for
//! `TagsWithSeparatorFormat` (`cpp/structural_tag.cc` @ 82505d0, the pinned
//! xgrammar 0.2.7 revision): the format compiles to
//! `tags_rule (separator tags_rule)*` where every position is a fresh choice
//! over ALL tags, so one tag may repeat (multiple reasoning blocks, repeated
//! calls to one tool) and tags may appear in any order. `at_least_one`
//! controls whether the empty string is accepted; `stop_after_first` caps the
//! match at a single tag.

use serde_json::{Map, Value};
use xgrammar_structural_tag::builders::StructuralTagOptions;
use xgrammar_structural_tag::format::{
    Format, GrammarFormat, JsonSchemaFormat, StructuralTag, TagFormat,
};
use xgrammar_structural_tag::{Error as XgrammarError, Result as XgrammarResult};

use super::super::{ScopedCallerConstraint, ScopedStructuralTagBuilder, ScopedToolChoice};
use super::{
    ChannelKind, EOM, EOT, FRAMING_MARKERS, FUNCTION_CALLS_CLOSE, FUNCTION_CALLS_OPEN,
    INVOKE_CLOSE, INVOKE_OPEN, MAX_CANDIDATE_LEN, MESSAGE, PARAMETER_CLOSE, PARAMETER_OPEN, START,
    classify_recipient, is_recipient_char,
};
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
        caller: Option<ScopedCallerConstraint<'_>>,
        options: &StructuralTagOptions,
    ) -> XgrammarResult<StructuralTag> {
        match tool_choice {
            // Tools absent or disabled (`"none"`): the grammar must not
            // sanction tool channels the request forbade.
            None => auto_turn(&[], caller, options),
            Some(ScopedToolChoice::Auto) => auto_turn(tools, caller, options),
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

/// Reasoning and tool channels in any order, with the answer channel last
/// when present. Without a caller constraint an empty generation stays
/// valid; with one, the turn must end in the constrained answer or (with
/// tools) a tool call, so a reasoning-only turn cannot slip past
/// `response_format`.
fn auto_turn(
    tools: &[Tool],
    caller: Option<ScopedCallerConstraint<'_>>,
    options: &StructuralTagOptions,
) -> XgrammarResult<StructuralTag> {
    validate_tool_names(tools)?;
    let tool_tags: Vec<TagFormat> =
        tools.iter().flat_map(|tool| tool_tags(tool, options)).collect();
    // The answer channel, when present, is always LAST. `<|eot|>` is a pure
    // stop token (never grammar content), and the model was never trained to
    // emit it right after `<|eom|>` — if the answer could be followed by more
    // channels, the model keeps re-opening answer channels forever instead of
    // stopping.
    let answer = Format::Tag(answer_tag(caller, options));
    let format = if caller.is_none() {
        let mut tags = vec![reasoning_tag()];
        tags.extend(tool_tags);
        Format::sequence(vec![
            Format::tags_with_separator(tags, CHANNEL_SEPARATOR, false, false),
            Format::optional(Format::sequence(vec![
                Format::optional(Format::const_string(CHANNEL_SEPARATOR)),
                answer,
            ])),
        ])
    } else {
        // `(channel <|start|>assistant)* (tool | answer)`: the tool tags are
        // serialized twice, once as a repeatable channel and once as a
        // terminal one.
        let channels = std::iter::once(reasoning_tag())
            .chain(tool_tags.iter().cloned())
            .map(Format::Tag)
            .collect();
        let terminal = one_of(tool_tags.into_iter().map(Format::Tag).chain([answer]).collect());
        Format::sequence(vec![
            Format::star(Format::sequence(vec![
                one_of(channels),
                Format::const_string(CHANNEL_SEPARATOR),
            ])),
            terminal,
        ])
    };
    Ok(StructuralTag::new(format))
}

/// An alternation, or the single alternative itself.
fn one_of(mut alternatives: Vec<Format>) -> Format {
    if alternatives.len() == 1 {
        alternatives.pop().expect("one alternative")
    } else {
        Format::or(alternatives)
    }
}

/// Reject tool names the streaming parser could not round-trip: the grammar
/// interpolates them verbatim into channel begins and invoke openers, so a
/// name must be a channel recipient (`[A-Za-z0-9_.\-]+`) whose longest
/// grammar spelling — doubled for a dot-less name (`ns.ns`, see
/// [`tool_tags`]) and quoted inside the invoke opener's attribute run —
/// stays within the parser's `MAX_CANDIDATE_LEN` cap, past which it
/// definitively rejects a run. `self` and `user` are reserved: the parser
/// classifies those recipients as reasoning/content channels, so a tool
/// channel with that begin would never be parsed as a tool call.
fn validate_tool_names(tools: &[Tool]) -> XgrammarResult<()> {
    for tool in tools {
        let spelled_len = if tool.name.contains('.') {
            tool.name.len()
        } else {
            2 * tool.name.len() + 1
        };
        let valid = !tool.name.is_empty()
            && classify_recipient(Some(&tool.name)) == ChannelKind::Tool
            && spelled_len + INVOKE_NAME_ATTR_OVERHEAD <= MAX_CANDIDATE_LEN
            && tool.name.chars().all(is_recipient_char);
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

/// Bytes the invoke opener's attribute run adds around the name: ` name=""`.
const INVOKE_NAME_ATTR_OVERHEAD: usize = " name=\"\"".len();

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
        Format::any_text_excluding(FRAMING_MARKERS),
        EOM,
    )
}

/// ` to=user<|message|>...<|eom|>`. A caller constraint constrains this body
/// only.
///
/// ALL channels end with `<|eom|>` in the grammar, never `<|eot|>`: vLLM runs
/// the matcher with `override_stop_tokens`, so a stop token is masked (and
/// rejected if it ever reaches the matcher) wherever the grammar cannot
/// terminate — a channel-sequence grammar only reaches a terminable state
/// after a tag's end boundary, so a channel ending in `<|eot|>` could never
/// close. With `<|eom|>`-only ends, the model closes the final channel with
/// `<|eom|>` and then emits `<|eot|>` purely as the stop token.
///
/// A JSON schema body cannot exclude the framing markers (xgrammar's JSON
/// string grammar admits them), so the streaming parser, which treats
/// framing as authoritative anywhere, stays the last line of defense there.
fn answer_tag(
    caller: Option<ScopedCallerConstraint<'_>>,
    options: &StructuralTagOptions,
) -> TagFormat {
    let content = match caller {
        None => Format::any_text_excluding(FRAMING_MARKERS),
        Some(ScopedCallerConstraint::JsonSchema(schema)) => json_schema(schema.clone(), options),
        Some(ScopedCallerConstraint::Regex(pattern)) => Format::regex(pattern),
        Some(ScopedCallerConstraint::Choice(choices)) => {
            one_of(choices.iter().map(Format::const_string).collect())
        }
        Some(ScopedCallerConstraint::Grammar(grammar)) => Format::Grammar(GrammarFormat {
            grammar: grammar.to_string(),
        }),
    };
    TagFormat::new(ANSWER_BEGIN, content, EOM)
}

/// Tool channel tags: one begin variant per model-known recipient spelling.
/// The recipient is the registered tool name verbatim, but for a dot-less
/// name `ns` the model is also known to emit the doubled `ns.ns`.
fn tool_tags(tool: &Tool, options: &StructuralTagOptions) -> Vec<TagFormat> {
    let content = tool_channel_content(tool, options);
    let mut begins = vec![format!(" to={}{MESSAGE}", tool.name)];
    if !tool.name.contains('.') {
        begins.push(format!(" to={0}.{0}{MESSAGE}", tool.name));
    }
    begins
        .into_iter()
        // `<|eom|>`-only ends: see `answer_tag` for why `<|eot|>` must stay
        // out of the grammar.
        .map(|begin| TagFormat::new(begin, content.clone(), EOM))
        .collect()
}

/// The ATEM body of a tool channel, whitespace-exact as the chat template
/// renders it. A tool that is not `strict: true`, has no declared
/// parameters, or has a schema the typed encoding cannot express faithfully
/// keeps the channel and invoke framing but leaves the invoke body free-form.
fn tool_channel_content(tool: &Tool, options: &StructuralTagOptions) -> Format {
    if tool.strict == Some(true)
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
        Format::const_string(format!("{FUNCTION_CALLS_OPEN}\n")),
        invoke_begin(&tool.name),
        // The body absorbs the newline before the invoke close, so an
        // empty-args call keeps the canonical `<atem:invoke name="N">\n`
        // `</atem:invoke>` shape the template renders.
        Format::any_text_excluding(&[INVOKE_CLOSE, EOM, EOT, START, MESSAGE]),
        Format::const_string(format!("{INVOKE_CLOSE}\n{FUNCTION_CALLS_CLOSE}")),
    ])
}

/// Whether the typed encoding can express every schema-valid call: no
/// extra properties admitted (an explicit `additionalProperties`/
/// `patternProperties` allowance cannot be rendered), every `required` name
/// declared, and every key round-trippable through
/// `<atem:parameter name="…">` — non-empty and free of `"`, `<`, and `>`.
/// The parser reads attributes up to the first `>` and skips empty names,
/// so the grammar must not force bytes it would drop. Anything else stays
/// free-form rather than silently narrowing the schema.
fn typed_encoding_is_faithful(parameters: &Value, properties: &Map<String, Value>) -> bool {
    let extra_properties_allowed = match parameters.get("additionalProperties") {
        None | Some(Value::Bool(false)) => false,
        Some(_) => true,
    };
    !extra_properties_allowed
        && parameters.get("patternProperties").is_none()
        && required_names(parameters).iter().all(|name| properties.contains_key(*name))
        && properties.keys().all(|key| !key.is_empty() && !key.contains(['"', '<', '>']))
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
        Format::const_string(format!("{FUNCTION_CALLS_OPEN}\n")),
        Format::tags_with_separator(
            vec![typed_invoke(name, properties, &required, options)],
            "\n",
            true,
            false,
        ),
        Format::const_string(format!("\n{FUNCTION_CALLS_CLOSE}")),
    ])
}

/// One `<atem:invoke>` block. Parameters follow the schema's property order
/// (the order the prompt shows the model), optional ones optionally; each
/// carries its trailing newline, so a call with no arguments stays
/// `<atem:invoke name="N">\n</atem:invoke>`.
fn typed_invoke(
    name: &str,
    properties: &Map<String, Value>,
    required: &[&str],
    options: &StructuralTagOptions,
) -> TagFormat {
    let mut elements = vec![invoke_name(name), Format::const_string("\">\n")];
    for (key, schema) in properties {
        let line = parameter_line(key, schema, options);
        elements.push(if required.contains(&key.as_str()) {
            line
        } else {
            Format::optional(line)
        });
    }
    TagFormat::new(
        invoke_name_begin(),
        Format::sequence(elements),
        INVOKE_CLOSE,
    )
}

/// The invoke opener up to its name: `<atem:invoke name="`.
fn invoke_name_begin() -> String {
    format!("{INVOKE_OPEN} name=\"")
}

/// `<atem:invoke name="NAME">` plus the newline the template emits after it.
fn invoke_begin(name: &str) -> Format {
    Format::sequence(vec![
        Format::const_string(invoke_name_begin()),
        invoke_name(name),
        Format::const_string("\">\n"),
    ])
}

/// The invoke name: the registered tool name verbatim, and for a dot-less
/// name also its doubled form (the model spells the doubled channel
/// recipient of [`tool_tags`] into the invoke as well; the parser collapses
/// it).
fn invoke_name(name: &str) -> Format {
    if name.contains('.') {
        Format::const_string(name)
    } else {
        Format::or(vec![
            Format::const_string(name),
            Format::const_string(format!("{name}.{name}")),
        ])
    }
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
        Format::const_string(format!("{PARAMETER_OPEN} name=\"{key}\">")),
        parameter_value(schema, options),
        Format::const_string(PARAMETER_CLOSE),
    ])
}

/// Value grammar for one parameter, by JSON-schema type. Strings, objects,
/// arrays, and unknown schemas stay free-form (the template renders objects
/// and arrays as JSON text); the other scalars reuse xgrammar's JSON grammar
/// over the compilable subset of the parameter schema, so facets such as
/// `minimum` keep applying.
fn parameter_value(schema: &Value, options: &StructuralTagOptions) -> Format {
    if let Some(alternation) = scalar_enum(schema) {
        return alternation;
    }
    match schema.get("type").and_then(Value::as_str) {
        Some(ty @ ("integer" | "number" | "boolean" | "null")) => {
            json_schema(scalar_schema(schema, ty), options)
        }
        // Framing markers and the invoke close stay excluded: the streaming
        // parser cuts the invoke body at the first `</atem:invoke>` and treats
        // quoted framing as a channel boundary, so the grammar must never
        // force bytes the parser cannot round-trip.
        _ => Format::any_text_excluding(&[PARAMETER_CLOSE, INVOKE_CLOSE, EOM, EOT, START, MESSAGE]),
    }
}

/// A scalar `enum` (or a scalar `const`) becomes an alternation of its
/// literals as const strings. Non-scalar values, markers, and huge lists stay
/// free-form.
fn scalar_enum(schema: &Value) -> Option<Format> {
    let values: Vec<&Value> = match (
        schema.get("enum").and_then(Value::as_array),
        schema.get("const"),
    ) {
        (Some(values), _) => values.iter().collect(),
        (None, Some(constant)) => vec![constant],
        (None, None) => return None,
    };
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
        if [PARAMETER_CLOSE, INVOKE_CLOSE, EOM, EOT, START, MESSAGE]
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

/// The facets of a scalar parameter schema that xgrammar 0.2.x compiles:
/// `type`, a scalar `const`, and for numeric types the bounds as numbers
/// (whole and within i64 for `integer`). Anything else (draft-4 boolean
/// exclusives, bounds beyond i64, `$ref`, `multipleOf`, an empty `enum`) is
/// dropped, widening the value to its bare type instead of failing grammar
/// compilation in the engine after the request was accepted.
fn scalar_schema(schema: &Value, ty: &str) -> Value {
    let mut narrowed = Map::new();
    narrowed.insert("type".to_string(), Value::String(ty.to_string()));
    if let Some(constant) = schema
        .get("const")
        .filter(|constant| matches!(constant, Value::Number(_) | Value::Bool(_) | Value::Null))
    {
        narrowed.insert("const".to_string(), constant.clone());
    }
    if matches!(ty, "integer" | "number") {
        for facet in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            if let Some(bound) =
                schema.get(facet).filter(|bound| compilable_bound(bound, ty == "integer"))
            {
                narrowed.insert(facet.to_string(), bound.clone());
            }
        }
    }
    Value::Object(narrowed)
}

/// Whether xgrammar compiles `bound`: a number, and for `integer` a whole one
/// strictly inside i64 when written as a float (a float literal at the i64
/// boundary itself is rejected for precision loss).
fn compilable_bound(bound: &Value, integer: bool) -> bool {
    let Value::Number(number) = bound else {
        return false;
    };
    !integer
        || number.is_i64()
        || number.as_f64().is_some_and(|float| {
            float.fract() == 0.0 && float > i64::MIN as f64 && float < i64::MAX as f64
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

    use super::super::{ASSISTANT, FRAMING_MARKERS, MAX_CANDIDATE_LEN};
    use super::{
        ANSWER_BEGIN, CHANNEL_SEPARATOR, MESSAGE, MuseGlimmerStructuralTagBuilder, REASONING_BEGIN,
        START, ScopedCallerConstraint, ScopedStructuralTagBuilder, ScopedToolChoice, Tool,
    };

    /// A strict tool: its arguments are grammar-pinned.
    fn tool(name: &str, parameters: serde_json::Value) -> Tool {
        Tool {
            name: name.to_string(),
            description: None,
            parameters,
            strict: Some(true),
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
                Some(ScopedCallerConstraint::JsonSchema(&json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"]
                }))),
                &StructuralTagOptions::default(),
            )
            .unwrap();
        let json = tag.to_json_string().unwrap();

        assert!(json.contains(r#""begin":" to=user<|message|>""#));
        assert!(json.contains(r#""begin":" to=self<|message|>""#));
        assert!(json.contains(r#""type":"json_schema""#));
        assert!(!json.contains("atem:invoke"));
        // With no tool channels the schema-bearing answer is mandatory.
        // With a schema the turn must END in the answer: `(reasoning sep)*`
        // then the mandatory answer tag.
        assert!(json.contains(r#"{"type":"star","content":{"type":"sequence""#));
        assert!(json.ends_with(r#""end":"<|eom|>"}]}}"#));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"star","content":{"type":"sequence","elements":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},"end":"<|eom|>"},{"type":"const_string","value":"<|start|>assistant"}]}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"<|eom|>"}]}}"#]].assert_eq(&json);
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
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"tags_with_separator","tags":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":false,"stop_after_first":false},{"type":"optional","content":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|start|>assistant"}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},"end":"<|eom|>"}]}}]}}"#]].assert_eq(&json);
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
        // the invoke tree serialized once per channel, and the dot-less name
        // admitted doubled in the invoke opener as well as the recipient.
        assert!(json.contains(r#""value":"<atem:function_calls>\n""#));
        assert!(json.contains(r#""begin":"<atem:invoke name=\"""#));
        assert!(json.contains(
            r#"{"type":"or","elements":[{"type":"const_string","value":"get_weather"},{"type":"const_string","value":"get_weather.get_weather"}]}"#
        ));
        assert_eq!(json.matches(r#"<atem:parameter name=\"city\">"#).count(), 2);
        // Parameters keep the schema's property order: the optional `unit`
        // precedes the required `city`.
        assert!(json.find(r#"name=\"unit\""#) < json.find(r#"name=\"city\""#));
        // Loose tool: permissive body keeps channel + invoke framing.
        assert!(json.contains(
            r#"{"type":"or","elements":[{"type":"const_string","value":"loose"},{"type":"const_string","value":"loose.loose"}]}"#
        ));
        assert!(json.contains(r#""begin":" to=get_weather.get_weather<|message|>""#));
        assert!(json.contains(r#""begin":" to=loose.loose<|message|>""#));
        assert!(json.contains(
            r#"{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]}"#
        ));
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"tags_with_separator","tags":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},"end":"<|eom|>"},{"type":"tag","begin":" to=get_weather<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<atem:invoke name=\"","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"get_weather"},{"type":"const_string","value":"get_weather.get_weather"}]},{"type":"const_string","value":"\">\n"},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"json_schema","json_schema":{"type":"integer"},"style":"json","any_order":false,"max_whitespace_cnt":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"type":"tag","begin":" to=get_weather.get_weather<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<atem:invoke name=\"","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"get_weather"},{"type":"const_string","value":"get_weather.get_weather"}]},{"type":"const_string","value":"\">\n"},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"unit\">"},{"type":"or","elements":[{"type":"const_string","value":"celsius"},{"type":"const_string","value":"fahrenheit"}]},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"city\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]},{"type":"optional","content":{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"days\">"},{"type":"json_schema","json_schema":{"type":"integer"},"style":"json","any_order":false,"max_whitespace_cnt":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"type":"tag","begin":" to=loose<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\""},{"type":"or","elements":[{"type":"const_string","value":"loose"},{"type":"const_string","value":"loose.loose"}]},{"type":"const_string","value":"\">\n"}]},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"},{"type":"tag","begin":" to=loose.loose<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\""},{"type":"or","elements":[{"type":"const_string","value":"loose"},{"type":"const_string","value":"loose.loose"}]},{"type":"const_string","value":"\">\n"}]},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":false,"stop_after_first":false},{"type":"optional","content":{"type":"sequence","elements":[{"type":"optional","content":{"type":"const_string","value":"<|start|>assistant"}},{"type":"tag","begin":" to=user<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},"end":"<|eom|>"}]}}]}}"#]].assert_eq(&json);
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
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"star","content":{"type":"sequence","elements":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},"end":"<|eom|>"},{"type":"const_string","value":"<|start|>assistant"}]}},{"type":"tags_with_separator","tags":[{"type":"tag","begin":" to=search<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<atem:invoke name=\"","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"search"},{"type":"const_string","value":"search.search"}]},{"type":"const_string","value":"\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"},{"type":"tag","begin":" to=search.search<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"tags_with_separator","tags":[{"type":"tag","begin":"<atem:invoke name=\"","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"search"},{"type":"const_string","value":"search.search"}]},{"type":"const_string","value":"\">\n"},{"type":"sequence","elements":[{"type":"sequence","elements":[{"type":"const_string","value":"<atem:parameter name=\"query\">"},{"type":"any_text","excludes":["</atem:parameter>","</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:parameter>"}]},{"type":"const_string","value":"\n"}]}]},"end":"</atem:invoke>"}],"separator":"\n","at_least_one":true,"stop_after_first":false},{"type":"const_string","value":"\n</atem:function_calls>"}]},"end":"<|eom|>"}],"separator":"<|start|>assistant","at_least_one":true,"stop_after_first":false}]}}"#]].assert_eq(&json);
    }

    #[test]
    fn function_choice_keeps_only_the_named_tool() {
        let tools = vec![
            tool("search", json!({"type": "object"})),
            tool("lookup", json!({"type": "object"})),
        ];
        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Function("lookup".to_string())),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap()
            .to_json_string()
            .unwrap();

        assert!(json.contains(r#""begin":" to=lookup.lookup<|message|>""#));
        assert!(!json.contains("search"));
    }

    #[test]
    fn function_choice_on_dotted_name_keeps_only_the_verbatim_begin() {
        let tools = vec![tool("my.ns", json!({"type": "object"}))];
        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Function("my.ns".to_string())),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap()
            .to_json_string()
            .unwrap();

        assert!(json.contains(r#""begin":" to=my.ns<|message|>""#));
        assert!(!json.contains("my.ns.my.ns"));
    }

    #[test]
    fn function_choice_on_unknown_name_fails_with_tool_not_found() {
        let tools = vec![tool("search", json!({"type": "object"}))];
        let error = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Function("missing".to_string())),
                None,
                &StructuralTagOptions::default(),
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
                Some(ScopedCallerConstraint::JsonSchema(
                    &json!({"type": "object"}),
                )),
                &StructuralTagOptions::default(),
            )
            .unwrap()
            .to_json_string()
            .unwrap();

        assert!(!json.contains(" to=search"));
        assert!(json.contains(r#""begin":" to=user<|message|>""#));
    }

    #[test]
    fn additional_properties_allowance_keeps_invoke_body_free_form() {
        let tools = vec![tool(
            "open",
            json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "additionalProperties": true
            }),
        )];
        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Required),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap()
            .to_json_string()
            .unwrap();

        assert!(!json.contains("<atem:parameter"));
    }

    #[test]
    fn undeclared_required_name_keeps_invoke_body_free_form() {
        let tools = vec![tool(
            "lookup",
            json!({
                "type": "object",
                "properties": { "a": { "type": "string" } },
                "required": ["a", "b"]
            }),
        )];
        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Required),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap()
            .to_json_string()
            .unwrap();

        assert!(!json.contains("<atem:parameter"));
    }

    #[test]
    fn unroundtrippable_parameter_keys_keep_invoke_body_free_form() {
        // The parser reads parameter attributes up to the first `>` and skips
        // empty names, so keys with `>` or empty keys must not be baked into
        // the typed encoding the grammar would force but the parser drops.
        for properties in [
            json!({ "a>b": { "type": "string" } }),
            json!({ "": { "type": "string" } }),
        ] {
            let tools = vec![tool(
                "calc",
                json!({ "type": "object", "properties": properties }),
            )];
            let json = MuseGlimmerStructuralTagBuilder
                .build_scoped(
                    &tools,
                    Some(ScopedToolChoice::Required),
                    None,
                    &StructuralTagOptions::default(),
                )
                .unwrap()
                .to_json_string()
                .unwrap();

            assert!(
                !json.contains("<atem:parameter"),
                "properties {properties} should stay free-form"
            );
        }
    }

    #[test]
    fn free_form_empty_invoke_keeps_the_canonical_shape() {
        // The minimal accepted invoke body is empty: a call with no arguments
        // stays `<atem:invoke name="N">\n</atem:invoke>`, with no blank line.
        let tools = vec![loose_tool("ping")];
        let json: serde_json::Value = serde_json::from_str(
            &MuseGlimmerStructuralTagBuilder
                .build_scoped(
                    &tools,
                    Some(ScopedToolChoice::Required),
                    None,
                    &StructuralTagOptions::default(),
                )
                .unwrap()
                .to_json_string()
                .unwrap(),
        )
        .unwrap();

        let content = &json["format"]["elements"][1]["tags"][0]["content"];
        let elements = content["elements"].as_array().unwrap();
        assert_eq!(elements[2]["type"], "any_text");
        let opener = elements[1]["elements"].as_array().unwrap();
        let minimal = format!(
            "{}{}{}{}{}",
            elements[0]["value"].as_str().unwrap(),
            opener[0]["value"].as_str().unwrap(),
            opener[1]["elements"][0]["value"].as_str().unwrap(),
            opener[2]["value"].as_str().unwrap(),
            elements[3]["value"].as_str().unwrap()
        );
        assert_eq!(
            minimal,
            "<atem:function_calls>\n<atem:invoke name=\"ping\">\n</atem:invoke>\n</atem:function_calls>"
        );
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
    fn reserved_channel_recipients_are_rejected_as_tool_names() {
        // The parser reserves `self` (reasoning) and `user` (content): a tool
        // with one of those names would build a grammar whose tool channel is
        // never parsed as a tool call.
        for reserved in ["self", "user"] {
            let tools = vec![tool(reserved, json!({"type": "object"}))];
            let error = MuseGlimmerStructuralTagBuilder
                .build_scoped(
                    &tools,
                    Some(ScopedToolChoice::Required),
                    None,
                    &StructuralTagOptions::default(),
                )
                .unwrap_err();

            assert!(
                matches!(&error, xgrammar_structural_tag::Error::Custom(_)),
                "{reserved}: {error:?}"
            );
        }
    }

    #[test]
    fn overlong_tool_names_are_rejected_as_unmatchable_recipients() {
        // The parser definitively rejects recipient runs over
        // MAX_CANDIDATE_LEN bytes, so the grammar must not bake one in.
        let tools = vec![tool(
            &"a".repeat(MAX_CANDIDATE_LEN + 1),
            json!({"type": "object"}),
        )];
        let error = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Required),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap_err();

        assert!(matches!(&error, xgrammar_structural_tag::Error::Custom(_)));
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

    #[test]
    fn boolean_exclusive_bound_is_dropped_from_scalar_schema() {
        let format = super::parameter_value(
            &json!({"type": "integer", "minimum": 0, "exclusiveMinimum": true}),
            &StructuralTagOptions::default(),
        );

        assert_eq!(
            serde_json::to_value(format).unwrap()["json_schema"],
            json!({"type": "integer", "minimum": 0})
        );
    }

    #[test]
    fn bound_beyond_i64_is_dropped_from_scalar_schema() {
        let format = super::parameter_value(
            &json!({"type": "integer", "minimum": 0, "maximum": 18446744073709551615u64}),
            &StructuralTagOptions::default(),
        );

        assert_eq!(
            serde_json::to_value(format).unwrap()["json_schema"],
            json!({"type": "integer", "minimum": 0})
        );
    }

    #[test]
    fn auto_with_tools_and_caller_schema_requires_a_terminal_channel() {
        // `response_format` with tools under `auto`: the turn may end on a
        // tool call, but never after reasoning alone or empty.
        let tools = vec![loose_tool("lookup")];
        let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
        let tag = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Auto),
                Some(ScopedCallerConstraint::JsonSchema(&schema)),
                &StructuralTagOptions::default(),
            )
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&tag.to_json_string().unwrap()).unwrap();

        let elements = json["format"]["elements"].as_array().unwrap();
        assert_eq!(elements[0]["type"], "star");
        let terminal = elements[1]["elements"].as_array().unwrap();
        assert_eq!(terminal.last().unwrap()["begin"], ANSWER_BEGIN);
        assert_eq!(terminal.last().unwrap()["content"]["type"], "json_schema");
        assert!(
            terminal[..terminal.len() - 1]
                .iter()
                .all(|tag| tag["begin"].as_str().unwrap().starts_with(" to=lookup"))
        );
        expect![[r#"{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"star","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"tag","begin":" to=self<|message|>","content":{"type":"any_text","excludes":["<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},"end":"<|eom|>"},{"type":"tag","begin":" to=lookup<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\""},{"type":"or","elements":[{"type":"const_string","value":"lookup"},{"type":"const_string","value":"lookup.lookup"}]},{"type":"const_string","value":"\">\n"}]},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"},{"type":"tag","begin":" to=lookup.lookup<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\""},{"type":"or","elements":[{"type":"const_string","value":"lookup"},{"type":"const_string","value":"lookup.lookup"}]},{"type":"const_string","value":"\">\n"}]},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"}]},{"type":"const_string","value":"<|start|>assistant"}]}},{"type":"or","elements":[{"type":"tag","begin":" to=lookup<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\""},{"type":"or","elements":[{"type":"const_string","value":"lookup"},{"type":"const_string","value":"lookup.lookup"}]},{"type":"const_string","value":"\">\n"}]},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"},{"type":"tag","begin":" to=lookup.lookup<|message|>","content":{"type":"sequence","elements":[{"type":"const_string","value":"<atem:function_calls>\n"},{"type":"sequence","elements":[{"type":"const_string","value":"<atem:invoke name=\""},{"type":"or","elements":[{"type":"const_string","value":"lookup"},{"type":"const_string","value":"lookup.lookup"}]},{"type":"const_string","value":"\">\n"}]},{"type":"any_text","excludes":["</atem:invoke>","<|eom|>","<|eot|>","<|start|>","<|message|>"],"max_tokens":null,"max_chars":null},{"type":"const_string","value":"</atem:invoke>\n</atem:function_calls>"}]},"end":"<|eom|>"},{"type":"tag","begin":" to=user<|message|>","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"answer":{"type":"string"}}},"style":"json","any_order":false,"max_whitespace_cnt":null},"end":"<|eom|>"}]}]}}"#]].assert_eq(&tag.to_json_string().unwrap());
    }

    #[test]
    fn every_free_text_region_excludes_all_framing_markers() {
        fn check(value: &serde_json::Value) {
            if let Some(object) = value.as_object() {
                if object.get("type").and_then(serde_json::Value::as_str) == Some("any_text") {
                    let excludes = object["excludes"].as_array().unwrap();
                    for marker in FRAMING_MARKERS {
                        assert!(
                            excludes.contains(&json!(marker)),
                            "{marker} missing in {value}"
                        );
                    }
                }
                object.values().for_each(check);
            } else if let Some(array) = value.as_array() {
                array.iter().for_each(check);
            }
        }
        let tools = vec![
            tool(
                "get_weather",
                json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            ),
            loose_tool("loose"),
        ];
        for tool_choice in [ScopedToolChoice::Auto, ScopedToolChoice::Required] {
            let tag = MuseGlimmerStructuralTagBuilder
                .build_scoped(
                    &tools,
                    Some(tool_choice),
                    None,
                    &StructuralTagOptions::default(),
                )
                .unwrap();
            check(&serde_json::from_str(&tag.to_json_string().unwrap()).unwrap());
        }
    }

    #[test]
    fn dotless_tool_name_bound_covers_its_doubled_spelling() {
        // `ns.ns` inside ` name=""` must fit the parser's candidate cap.
        let longest = (MAX_CANDIDATE_LEN - " name=\"\"".len() - 1) / 2;
        for (len, ok) in [(longest, true), (longest + 1, false)] {
            let tools = vec![tool(&"a".repeat(len), json!({"type": "object"}))];
            let result = MuseGlimmerStructuralTagBuilder.build_scoped(
                &tools,
                Some(ScopedToolChoice::Required),
                None,
                &StructuralTagOptions::default(),
            );
            assert_eq!(result.is_ok(), ok, "dot-less name of {len} bytes");
        }
        // A dotted name is spelled verbatim, so it may be longer.
        let dotted = format!("{}.{}", "a".repeat(longest), "b".repeat(longest));
        let tools = vec![tool(&dotted, json!({"type": "object"}))];
        MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Required),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap();
    }

    #[test]
    fn string_const_becomes_const_string() {
        let format = super::parameter_value(
            &json!({"type": "string", "const": "only"}),
            &StructuralTagOptions::default(),
        );

        assert_eq!(
            serde_json::to_value(format).unwrap(),
            json!({"type": "const_string", "value": "only"})
        );
    }

    #[test]
    fn absent_strict_keeps_invoke_body_free_form() {
        // Only `strict: true` pins arguments; the caller resolves the server
        // strictness floor into that flag.
        let tools = vec![Tool {
            strict: None,
            ..tool(
                "search",
                json!({"type": "object", "properties": {"q": {"type": "string"}}}),
            )
        }];
        let json = MuseGlimmerStructuralTagBuilder
            .build_scoped(
                &tools,
                Some(ScopedToolChoice::Required),
                None,
                &StructuralTagOptions::default(),
            )
            .unwrap()
            .to_json_string()
            .unwrap();

        assert!(!json.contains("<atem:parameter"));
        assert!(json.contains(r#""type":"any_text""#));
    }

    #[test]
    fn regex_choice_and_grammar_constraints_scope_to_answer_channel() {
        let cases = [
            (
                ScopedCallerConstraint::Regex("^[a-z]+$"),
                r#"{"type":"regex","pattern":"^[a-z]+$"}"#,
            ),
            (
                ScopedCallerConstraint::Choice(&["yes".to_string(), "no".to_string()]),
                r#"{"type":"or","elements":[{"type":"const_string","value":"yes"},{"type":"const_string","value":"no"}]}"#,
            ),
            (
                ScopedCallerConstraint::Grammar("root ::= \"ok\""),
                r#"{"type":"grammar","grammar":"root ::= \"ok\""}"#,
            ),
        ];
        for (caller, content) in cases {
            let json = MuseGlimmerStructuralTagBuilder
                .build_scoped(&[], None, Some(caller), &StructuralTagOptions::default())
                .unwrap()
                .to_json_string()
                .unwrap();
            let answer = format!(r#""begin":" to=user<|message|>","content":{content}"#);
            assert!(json.contains(&answer), "{caller:?}: {json}");
            assert!(json.ends_with(r#""end":"<|eom|>"}]}}"#), "{caller:?}");
        }
    }
}
