// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use serde_json::{Value, json};
use vllm_tokenizer::test_utils::TestTokenizer;
use vllm_tokenizer::{DecodedText, Tokenizer as _};
use xgrammar_structural_tag::ToolChoice;
use xgrammar_structural_tag::format::Format;

use super::{BeamUnifiedParser, HfPosition, build_output_grammar, other_names};
use crate::output_grammar::test_utils::outline;
use crate::output_grammar::{GrammarCoverage, OutputGrammarContext, ToolStrictLevel};
use crate::tool::Tool;
use crate::unified::{ResponseTemplate, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

/// The `tag_pattern` of the checkpoint of [`template`].
const TAG_PATTERN: &str =
    r#"<rfl:parameter name="(?P<key>[^"]+)">\n(?P<value>.*?)\n</rfl:parameter>"#;
/// [`TAG_PATTERN`], but a value may also end where the next parameter opens.
const LENIENT_TAG_PATTERN: &str = r#"<rfl:parameter name="(?P<key>[^"]+)">\n(?P<value>.*?)\n(?:</rfl:parameter>|(?=<rfl:parameter name="))"#;

/// The Beam-1.0 `response_template`, with `tag_pattern`.
fn template(tag_pattern: &str) -> Arc<ResponseTemplate> {
    template_with(tag_pattern, json!("<|end|><|start|>assistant"))
}

/// [`template`] with `thinking_close` as the reasoning closer.
fn template_with(tag_pattern: &str, thinking_close: Value) -> Arc<ResponseTemplate> {
    let template = json!({
        "defaults": {"role": "assistant"},
        "start_anchor_pattern": r"<\|start\|>assistant",
        "fields": {
            "thinking": {
                "open": "<|channel|>analysis<|message|>",
                "close": thinking_close,
                "content": "text"
            },
            "content": {
                "open": "<|channel|>final<|message|>",
                "close": ["<|return|>", "<|end|>", "<|call|>"],
                "content": "text"
            },
            "tool_calls": {
                "open_pattern": r" to=(?P<name>[^\s<]+)<\|channel\|>analysis <\|constrain\|>xml<\|message\|>",
                "close": ["<|end|><|start|>assistant", "<|call|>"],
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": tag_pattern,
                    "value_parser": {"name": "text", "args": {"strip": false}}
                },
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"}
                }
            }
        }
    });
    Arc::new(ResponseTemplate::from_json(&template).unwrap())
}

fn tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {"enum": ["c", "f"]}
                },
                "required": ["location"],
                "additionalProperties": false
            }),
            strict: Some(true),
            defer_loading: None,
        },
        Tool {
            name: "get_time".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {"zone": {"type": "string"}}
            }),
            strict: None,
            defer_loading: None,
        },
    ]
}

/// A parser initialized from `prompt`, the rendered conversation.
fn parser(prompt: &str) -> BeamUnifiedParser {
    parser_with(TAG_PATTERN, prompt)
}

fn parser_with(tag_pattern: &str, prompt: &str) -> BeamUnifiedParser {
    let tokenizer = Arc::new(TestTokenizer::new());
    let prompt_ids = tokenizer.encode(prompt, false).unwrap();
    let mut parser = BeamUnifiedParser::new(template(tag_pattern), &tools(), tokenizer);
    parser.initialize(&prompt_ids).unwrap();
    parser
}

const PROMPT: &str = "<|start|>user<|message|>Weather in Paris?<|end|><|start|>assistant";

fn grammar(position: HfPosition, tool_choice: ToolChoice, parallel: bool) -> Option<String> {
    grammar_with(position, &tools(), tool_choice, parallel, false, false)
}

fn grammar_with(
    position: HfPosition,
    tools: &[Tool],
    tool_choice: ToolChoice,
    parallel: bool,
    always_constrain: bool,
    lenient: bool,
) -> Option<String> {
    let ctx = OutputGrammarContext {
        tools,
        tool_choice: &tool_choice,
        tool_strict_level: ToolStrictLevel::Auto,
        parallel_tool_calls: parallel,
        answer: None,
    };
    grammar_in(position, &ctx, always_constrain, lenient)
}

fn grammar_in(
    position: HfPosition,
    ctx: &OutputGrammarContext<'_>,
    always_constrain: bool,
    lenient: bool,
) -> Option<String> {
    let grammar = build_output_grammar(
        position,
        ctx,
        always_constrain,
        lenient,
        template(TAG_PATTERN).reasoning_closers(),
    )
    .unwrap()?;
    assert_eq!(grammar.coverage, GrammarCoverage::FromTokenZero);
    Some(outline(&grammar.format))
}

fn constrained_grammar(
    position: HfPosition,
    tool_choice: ToolChoice,
    answer: Format,
    always_constrain: bool,
) -> Option<String> {
    let tools = tools();
    let ctx = OutputGrammarContext {
        tools: &tools,
        tool_choice: &tool_choice,
        tool_strict_level: ToolStrictLevel::Auto,
        parallel_tool_calls: true,
        answer: Some(&answer),
    };
    grammar_in(position, &ctx, always_constrain, false)
}

#[test]
fn position_follows_the_prompt_and_then_the_output() {
    let position = |suffix: &str| parser(&format!("{PROMPT}{suffix}")).inner.position();

    assert_eq!(position(""), HfPosition::ResponseStart);
    assert_eq!(
        position("<|channel|>analysis<|message|>Let me"),
        HfPosition::Reasoning
    );
    // The reasoning closer ends with the anchor, so the next message starts.
    assert_eq!(
        position("<|channel|>analysis<|message|>Done.<|end|><|start|>assistant"),
        HfPosition::ResponseStart
    );
    // A partial closer stays buffered.
    assert_eq!(
        position("<|channel|>analysis<|message|>Done.<|end|>"),
        HfPosition::Other
    );
    assert_eq!(
        position("<|channel|>final<|message|>It is"),
        HfPosition::Content
    );
    // A partial boundary stays buffered.
    assert_eq!(position("<|channel|>ana"), HfPosition::Other);
    // Without the anchor, the parser parses the output from the start.
    assert_eq!(
        parser("no anchor").inner.position(),
        HfPosition::ResponseStart
    );

    let mut parser = parser(PROMPT);
    parser
        .parse_into(
            DecodedText::unattributed("<|channel|>analysis<|message|>Hmm"),
            &mut UnifiedParserOutput::default(),
        )
        .unwrap();
    assert_eq!(parser.inner.position(), HfPosition::Reasoning);
}

#[test]
fn required_calls_any_tool_after_reasoning() {
    expect_test::expect![[r#"
        sequence
          tag `<|channel|>analysis<|message|>` .. `<|end|><|start|>assistant`
            sequence
              /[^<]/
              text
          sequence
            or
              sequence
                ` to=get_weather<|channel|>analysis <|constrain|>xml<|message|>`
                or
                  sequence
                    sequence
                      optional `\n`
                      tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                    optional
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                    optional
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                          or
                            `c`
                            `f`
                  sequence
                    optional
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                    sequence
                      optional `\n`
                      tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                    optional
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                          or
                            `c`
                            `f`
              sequence
                ` to=get_time<|channel|>analysis <|constrain|>xml<|message|>`
                sequence
                  optional
                    sequence
                      optional `\n`
                      tag `<rfl:parameter name="zone">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
            star
              sequence
                `<|end|><|start|>assistant`
                or
                  sequence
                    ` to=get_weather<|channel|>analysis <|constrain|>xml<|message|>`
                    or
                      sequence
                        sequence
                          optional `\n`
                          tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                        optional
                          sequence
                            optional `\n`
                            tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                        optional
                          sequence
                            optional `\n`
                            tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                              or
                                `c`
                                `f`
                      sequence
                        optional
                          sequence
                            optional `\n`
                            tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                        sequence
                          optional `\n`
                          tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                        optional
                          sequence
                            optional `\n`
                            tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                              or
                                `c`
                                `f`
                  sequence
                    ` to=get_time<|channel|>analysis <|constrain|>xml<|message|>`
                    sequence
                      optional
                        sequence
                          optional `\n`
                          tag `<rfl:parameter name="zone">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
            optional `<|call|>`
    "#]].assert_eq(
        &grammar(HfPosition::ResponseStart, ToolChoice::required(), true).unwrap(),
    );
}

#[test]
fn auto_with_a_strict_tool_also_allows_an_answer() {
    expect_test::expect![[r#"
        sequence
          tag `<|channel|>analysis<|message|>` .. `<|end|><|start|>assistant`
            sequence
              /[^<]/
              text
          or
            sequence
              or
                sequence
                  ` to=get_weather<|channel|>analysis <|constrain|>xml<|message|>`
                  or
                    sequence
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                      optional
                        sequence
                          optional `\n`
                          tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                      optional
                        sequence
                          optional `\n`
                          tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                            or
                              `c`
                              `f`
                    sequence
                      optional
                        sequence
                          optional `\n`
                          tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                      optional
                        sequence
                          optional `\n`
                          tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                            or
                              `c`
                              `f`
                sequence
                  ` to=get_time<|channel|>analysis <|constrain|>xml<|message|>`
                  sequence
                    optional
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="zone">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                sequence
                  sequence
                    ` to=`
                    /(?:[^\s<g][^\s<]*|g(?:[^\s<e][^\s<]*|e(?:[^\s<t][^\s<]*|t(?:[^\s<_][^\s<]*|_(?:[^\s<tw][^\s<]*|t(?:[^\s<i][^\s<]*|i(?:[^\s<m][^\s<]*|m(?:[^\s<e][^\s<]*|e(?:[^\s<][^\s<]*))?)?)?|w(?:[^\s<e][^\s<]*|e(?:[^\s<a][^\s<]*|a(?:[^\s<t][^\s<]*|t(?:[^\s<h][^\s<]*|h(?:[^\s<e][^\s<]*|e(?:[^\s<r][^\s<]*|r(?:[^\s<][^\s<]*))?)?)?)?)?)?)?)?)?)?)/
                    `<|channel|>analysis <|constrain|>xml<|message|>`
                  sequence
                    star
                      sequence
                        optional `\n`
                        sequence
                          `<rfl:parameter name="`
                          /[a-zA-Z_][a-zA-Z0-9_]*/
                          `">\n`
                          or
                            text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`]
                            json(number) excluding [`<|end|>`, `<|call|>`]
                            json(boolean) excluding [`<|end|>`, `<|call|>`]
                            json(null) excluding [`<|end|>`, `<|call|>`]
                            json({ ... }) excluding [`<|end|>`, `<|call|>`]
                            json(any[]) excluding [`<|end|>`, `<|call|>`]
                          `\n</rfl:parameter>`
              star
                sequence
                  `<|end|><|start|>assistant`
                  or
                    sequence
                      ` to=get_weather<|channel|>analysis <|constrain|>xml<|message|>`
                      or
                        sequence
                          sequence
                            optional `\n`
                            tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                          optional
                            sequence
                              optional `\n`
                              tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                          optional
                            sequence
                              optional `\n`
                              tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                                or
                                  `c`
                                  `f`
                        sequence
                          optional
                            sequence
                              optional `\n`
                              tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                          sequence
                            optional `\n`
                            tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                          optional
                            sequence
                              optional `\n`
                              tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                                or
                                  `c`
                                  `f`
                    sequence
                      ` to=get_time<|channel|>analysis <|constrain|>xml<|message|>`
                      sequence
                        optional
                          sequence
                            optional `\n`
                            tag `<rfl:parameter name="zone">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                    sequence
                      sequence
                        ` to=`
                        /(?:[^\s<g][^\s<]*|g(?:[^\s<e][^\s<]*|e(?:[^\s<t][^\s<]*|t(?:[^\s<_][^\s<]*|_(?:[^\s<tw][^\s<]*|t(?:[^\s<i][^\s<]*|i(?:[^\s<m][^\s<]*|m(?:[^\s<e][^\s<]*|e(?:[^\s<][^\s<]*))?)?)?|w(?:[^\s<e][^\s<]*|e(?:[^\s<a][^\s<]*|a(?:[^\s<t][^\s<]*|t(?:[^\s<h][^\s<]*|h(?:[^\s<e][^\s<]*|e(?:[^\s<r][^\s<]*|r(?:[^\s<][^\s<]*))?)?)?)?)?)?)?)?)?)?)/
                        `<|channel|>analysis <|constrain|>xml<|message|>`
                      sequence
                        star
                          sequence
                            optional `\n`
                            sequence
                              `<rfl:parameter name="`
                              /[a-zA-Z_][a-zA-Z0-9_]*/
                              `">\n`
                              or
                                text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`]
                                json(number) excluding [`<|end|>`, `<|call|>`]
                                json(boolean) excluding [`<|end|>`, `<|call|>`]
                                json(null) excluding [`<|end|>`, `<|call|>`]
                                json({ ... }) excluding [`<|end|>`, `<|call|>`]
                                json(any[]) excluding [`<|end|>`, `<|call|>`]
                              `\n</rfl:parameter>`
              optional `<|call|>`
            sequence
              `<|channel|>final<|message|>`
              text excluding [`<|end|>`, `<|return|>`]
              optional `<|return|>`
    "#]]
        .assert_eq(&grammar(HfPosition::ResponseStart, ToolChoice::auto(), true).unwrap());
}

#[test]
fn named_choice_calls_that_tool_once_after_prompted_reasoning() {
    expect_test::expect![[r#"
        sequence
          tag `` text `<|end|><|start|>assistant`
          sequence
            sequence
              ` to=get_weather<|channel|>analysis <|constrain|>xml<|message|>`
              or
                sequence
                  sequence
                    optional `\n`
                    tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                  optional
                    sequence
                      optional `\n`
                      tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                  optional
                    sequence
                      optional `\n`
                      tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                        or
                          `c`
                          `f`
                sequence
                  optional
                    sequence
                      optional `\n`
                      tag `<rfl:parameter name="days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                  sequence
                    optional `\n`
                    tag `<rfl:parameter name="location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                  optional
                    sequence
                      optional `\n`
                      tag `<rfl:parameter name="unit">\n` .. `\n</rfl:parameter>`
                        or
                          `c`
                          `f`
            optional `<|call|>`
    "#]].assert_eq(
        &grammar(HfPosition::Reasoning, ToolChoice::function("get_weather"), true).unwrap(),
    );
}

#[test]
fn no_grammar_without_a_constraining_request_or_known_position() {
    let non_strict = vec![tools().pop().unwrap()];
    for tools in [&[][..], &non_strict] {
        assert!(
            grammar_with(
                HfPosition::ResponseStart,
                tools,
                ToolChoice::auto(),
                true,
                false,
                false
            )
            .is_none()
        );
    }
    assert!(
        grammar_with(
            HfPosition::Other,
            &tools(),
            ToolChoice::required(),
            true,
            true,
            false
        )
        .is_none()
    );
}

#[test]
fn none_with_tools_only_answers() {
    expect_test::expect![[r#"
        sequence
          tag `<|channel|>analysis<|message|>` .. `<|end|><|start|>assistant`
            sequence
              /[^<]/
              text
          sequence
            `<|channel|>final<|message|>`
            text excluding [`<|end|>`, `<|return|>`]
            optional `<|return|>`
    "#]]
    .assert_eq(&grammar(HfPosition::ResponseStart, ToolChoice::none(), true).unwrap());
}

#[test]
fn always_constrain_answers_without_tools() {
    expect_test::expect![[r#"
        sequence
          tag `` text `<|end|><|start|>assistant`
          sequence
            `<|channel|>final<|message|>`
            text excluding [`<|end|>`, `<|return|>`]
            optional `<|return|>`
    "#]]
    .assert_eq(
        &grammar_with(
            HfPosition::Reasoning,
            &[],
            ToolChoice::auto(),
            true,
            true,
            false,
        )
        .unwrap(),
    );
}

#[test]
fn always_constrain_allows_non_strict_calls_or_an_answer() {
    let non_strict = vec![tools().pop().unwrap()];
    expect_test::expect![[r#"
        sequence
          tag `<|channel|>analysis<|message|>` .. `<|end|><|start|>assistant`
            sequence
              /[^<]/
              text
          or
            sequence
              or
                sequence
                  ` to=get_time<|channel|>analysis <|constrain|>xml<|message|>`
                  sequence
                    optional
                      sequence
                        optional `\n`
                        tag `<rfl:parameter name="zone">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`] `\n</rfl:parameter>`
                sequence
                  sequence
                    ` to=`
                    /(?:[^\s<g][^\s<]*|g(?:[^\s<e][^\s<]*|e(?:[^\s<t][^\s<]*|t(?:[^\s<_][^\s<]*|_(?:[^\s<t][^\s<]*|t(?:[^\s<i][^\s<]*|i(?:[^\s<m][^\s<]*|m(?:[^\s<e][^\s<]*|e(?:[^\s<][^\s<]*))?)?)?)?)?)?)?)/
                    `<|channel|>analysis <|constrain|>xml<|message|>`
                  sequence
                    star
                      sequence
                        optional `\n`
                        sequence
                          `<rfl:parameter name="`
                          /[a-zA-Z_][a-zA-Z0-9_]*/
                          `">\n`
                          or
                            text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`]
                            json(number) excluding [`<|end|>`, `<|call|>`]
                            json(boolean) excluding [`<|end|>`, `<|call|>`]
                            json(null) excluding [`<|end|>`, `<|call|>`]
                            json({ ... }) excluding [`<|end|>`, `<|call|>`]
                            json(any[]) excluding [`<|end|>`, `<|call|>`]
                          `\n</rfl:parameter>`
              optional `<|call|>`
            sequence
              `<|channel|>final<|message|>`
              text excluding [`<|end|>`, `<|return|>`]
              optional `<|return|>`
    "#]]
    .assert_eq(
        &grammar_with(
            HfPosition::ResponseStart,
            &non_strict,
            ToolChoice::auto(),
            false,
            true,
            false,
        )
        .unwrap(),
    );
}

#[test]
fn always_constrain_finishes_an_open_answer() {
    expect_test::expect![[r#"
        sequence
          text excluding [`<|end|>`, `<|return|>`]
          optional `<|return|>`
    "#]]
    .assert_eq(
        &grammar_with(
            HfPosition::Content,
            &tools(),
            ToolChoice::auto(),
            true,
            true,
            false,
        )
        .unwrap(),
    );
    assert!(
        grammar_with(
            HfPosition::Content,
            &tools(),
            ToolChoice::auto(),
            true,
            false,
            false
        )
        .is_none()
    );
    assert!(
        grammar_with(
            HfPosition::Content,
            &tools(),
            ToolChoice::required(),
            true,
            true,
            false
        )
        .is_none()
    );
}

#[test]
fn answer_holds_the_requested_format() {
    let json = || Format::json_schema(json!({"type": "object"}));
    let answer = constrained_grammar(HfPosition::ResponseStart, ToolChoice::none(), json(), false);
    expect_test::expect![[r#"
        sequence
          tag `<|channel|>analysis<|message|>` .. `<|end|><|start|>assistant`
            sequence
              /[^<]/
              text
          sequence
            `<|channel|>final<|message|>`
            json(object)
            optional `<|return|>`
    "#]]
    .assert_eq(&answer.unwrap());

    // Under `auto`, the answer that may replace calls holds it too.
    let auto =
        constrained_grammar(HfPosition::ResponseStart, ToolChoice::auto(), json(), false).unwrap();
    assert!(auto.ends_with(
        "`<|channel|>final<|message|>`\n      json(object)\n      optional `<|return|>`\n"
    ));

    // A forced call leaves no answer to constrain.
    assert_eq!(
        constrained_grammar(
            HfPosition::ResponseStart,
            ToolChoice::required(),
            json(),
            false
        ),
        grammar(HfPosition::ResponseStart, ToolChoice::required(), true)
    );
}

#[test]
fn request_constraint_applies_alone_in_an_open_answer() {
    let json = Format::json_schema(json!({"type": "object"}));
    assert!(constrained_grammar(HfPosition::Content, ToolChoice::auto(), json, true).is_none());
}

/// Parse `output` after the prompt under `tag_pattern`, returning the calls as
/// `(name, arguments)`.
fn calls(tag_pattern: &str, output: &str) -> Vec<(String, Value)> {
    let mut parser = parser_with(tag_pattern, PROMPT);
    let mut events = UnifiedParserOutput::default();
    parser.parse_into(DecodedText::unattributed(output), &mut events).unwrap();
    events.append(parser.finish().unwrap());
    let mut calls: Vec<(String, String)> = Vec::new();
    for event in events.events {
        if let UnifiedParserEvent::ToolCall(delta) = event {
            if let Some(name) = delta.name {
                calls.push((name, String::new()));
            }
            calls.last_mut().unwrap().1.push_str(&delta.arguments);
        }
    }
    calls
        .into_iter()
        .map(|(name, arguments)| (name, serde_json::from_str(&arguments).unwrap()))
        .collect()
}

/// Output in the grammar's canonical spelling parses back to the values the
/// grammar admitted.
#[test]
fn parser_reads_back_the_grammar_spelling() {
    let parameter = |key: &str, value: &str| {
        format!("<rfl:parameter name=\"{key}\">\n{value}\n</rfl:parameter>")
    };
    let call = |name: &str, parameters: &[String]| {
        format!(
            " to={name}<|channel|>analysis <|constrain|>xml<|message|>{}",
            parameters.join("\n")
        )
    };
    let output = format!(
        "<|channel|>analysis<|message|>Two tools.<|end|><|start|>assistant{}<|end|><|start|>assistant{}",
        call(
            "get_weather",
            &[
                parameter("location", "Paris"),
                parameter("days", "2"),
                parameter("unit", "c")
            ]
        ),
        call("get_time", &[parameter("zone", "multi\nline")]),
    );

    for tag_pattern in [TAG_PATTERN, LENIENT_TAG_PATTERN] {
        assert_eq!(
            calls(tag_pattern, &output),
            vec![
                (
                    "get_weather".to_string(),
                    json!({"location": "Paris", "days": 2, "unit": "c"})
                ),
                ("get_time".to_string(), json!({"zone": "multi\nline"})),
            ]
        );
    }
}

#[test]
fn values_end_before_the_next_parameter_where_the_template_looks_ahead() {
    let output = " to=get_weather<|channel|>analysis <|constrain|>xml<|message|>\
        <rfl:parameter name=\"days\">\n2\n\
        <rfl:parameter name=\"location\">\nParis\n</rfl:parameter>";

    assert!(parser_with(LENIENT_TAG_PATTERN, PROMPT).lenient);
    assert_eq!(
        calls(LENIENT_TAG_PATTERN, output),
        vec![(
            "get_weather".to_string(),
            json!({"days": 2, "location": "Paris"})
        )]
    );
    assert!(!parser(PROMPT).lenient);
}

#[test]
fn reasoning_ends_at_every_closer_of_the_template() {
    let closers = [
        "<|end|><|start|>assistant",
        "<|constrain|><|start|>assistant",
        "<|start|>assistant",
    ];
    let tools = tools();
    let tokenizer = Arc::new(TestTokenizer::new());
    let mut parser = BeamUnifiedParser::new(
        template_with(TAG_PATTERN, json!(closers)),
        &tools,
        tokenizer.clone(),
    );
    parser.initialize(&tokenizer.encode(PROMPT, false).unwrap()).unwrap();
    let grammar = parser
        .build_output_grammar(&OutputGrammarContext {
            tools: &tools,
            tool_choice: &ToolChoice::required(),
            tool_strict_level: ToolStrictLevel::Auto,
            parallel_tool_calls: false,
            answer: None,
        })
        .unwrap()
        .unwrap();

    assert_eq!(
        outline(&grammar.format).lines().nth(1).unwrap(),
        "  tag `<|channel|>analysis<|message|>` .. `<|constrain|><|start|>assistant` | \
         `<|end|><|start|>assistant` | `<|start|>assistant`"
    );
}

#[test]
fn lenient_arguments_come_in_declared_or_sorted_order() {
    expect_test::expect![[r#"
        sequence
          tag `` text `<|end|><|start|>assistant`
          sequence
            sequence
              ` to=get_weather<|channel|>analysis <|constrain|>xml<|message|>`
              sequence
                optional `\n`
                `<rfl:parameter name="`
                or
                  sequence
                    or
                      tag `location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`, `\n<rfl:parameter name="`] `\n</rfl:parameter>`
                      sequence
                        tag `location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`, `\n<rfl:parameter name="`] `\n</rfl:parameter><rfl:parameter name="` | `\n</rfl:parameter>\n<rfl:parameter name="` | `\n<rfl:parameter name="`
                        or
                          tag `days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter>`
                          sequence
                            optional tag `days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter><rfl:parameter name="` | `\n</rfl:parameter>\n<rfl:parameter name="` | `\n<rfl:parameter name="`
                            tag `unit">\n` .. `\n</rfl:parameter>`
                              or
                                `c`
                                `f`
                  sequence
                    optional tag `days">\n` json(integer) excluding [`<|end|>`, `<|call|>`] `\n</rfl:parameter><rfl:parameter name="` | `\n</rfl:parameter>\n<rfl:parameter name="` | `\n<rfl:parameter name="`
                    or
                      tag `location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`, `\n<rfl:parameter name="`] `\n</rfl:parameter>`
                      sequence
                        tag `location">\n` text excluding [`<|end|>`, `<|call|>`, `</rfl:parameter>`, `\n<rfl:parameter name="`] `\n</rfl:parameter><rfl:parameter name="` | `\n</rfl:parameter>\n<rfl:parameter name="` | `\n<rfl:parameter name="`
                        tag `unit">\n` .. `\n</rfl:parameter>`
                          or
                            `c`
                            `f`
            optional `<|call|>`
    "#]].assert_eq(
        &grammar_with(
            HfPosition::Reasoning,
            &tools(),
            ToolChoice::function("get_weather"),
            true,
            false,
            true,
        )
        .unwrap(),
    );
}

#[test]
fn other_names_match_every_name_but_the_given_ones() {
    let regex = regex_automata::meta::Regex::new(&format!(
        "^(?:{})$",
        other_names(&["get", "get_time", "a.b"])
    ))
    .unwrap();

    for name in [
        "g",
        "ge",
        "gets",
        "get_",
        "get_t",
        "get_times",
        "x",
        "a",
        "a.",
        "axb",
        "a.bc",
    ] {
        assert!(regex.is_match(name), "{name}");
    }
    for name in ["get", "get_time", "a.b", "", "get time", "get<"] {
        assert!(!regex.is_match(name), "{name}");
    }
}
