// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Full-output grammar for Kimi K3 XTML generations.
//!
//! The grammar starts at the channel the prompt left open, which is the
//! parser's initialized [`KimiK3Mode`], and follows the same channel
//! transitions as the parser. A prompt that opened no channel may first emit a
//! whole `think` channel.
//!
//! | Tool choice | After `think` closes           | After `response` closes |
//! | ----------- | ------------------------------ | ----------------------- |
//! | `auto`      | `response [tools]`, or `tools` | `[tools]`               |
//! | `required`  | `[response] tools`             | `tools`                 |
//! | named       | `[response] tools`, one call   | `tools`, one call       |
//!
//! Every path may end with the message close, and a stop token is legal only
//! after the last channel closes.
//!
//! The user's answer constraint, if any, replaces the free text of the
//! `response` channel. Without a callable tool, the `response` channel then
//! follows `think` alone.

mod arguments;

use xgrammar_structural_tag::NormalizedToolChoice;
use xgrammar_structural_tag::format::Format;
use xgrammar_structural_tag::tool::BuilderToolChoice;

use self::arguments::call_tag;
use super::{
    CLOSE, END_OF_MSG, KimiK3Mode, MESSAGE_CLOSE, OPEN, RESPONSE_CLOSE, RESPONSE_OPEN, THINK_CLOSE,
    THINK_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
};
use crate::output_grammar::{
    self, BuiltOutputGrammar, OutputGrammarContext, normalize_tool_choice,
};

/// Build the grammar of everything generated after a prompt that left the
/// parser in `mode`, or `None` when the request asks for no tool grammar and
/// has no answer constraint.
pub(super) fn build_output_grammar(
    mode: &KimiK3Mode,
    ctx: &OutputGrammarContext<'_>,
) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
    let normalized = normalize_tool_choice(ctx)?;
    if normalized.is_none() && ctx.answer.is_none() {
        return Ok(None);
    }
    let calls = normalized.map(
        |NormalizedToolChoice {
             function_tools,
             choice,
             ..
         }| {
            let single_call = choice == BuilderToolChoice::Forced || !ctx.parallel_tool_calls;
            let tools = Format::tag(
                TOOLS_OPEN,
                Format::tags_with_separator(
                    function_tools.iter().map(call_tag).collect(),
                    "",
                    true,
                    single_call,
                ),
                TOOLS_CLOSE,
            );
            (choice, tools)
        },
    );
    let answer = ctx.answer;

    let mut elements = match mode {
        KimiK3Mode::Reasoning => vec![think_body(), after_reasoning(calls, answer)],
        KimiK3Mode::Response => rest_of_response(calls, answer),
        KimiK3Mode::Idle => vec![
            Format::optional(Format::tag(THINK_OPEN, think_text(), THINK_CLOSE)),
            after_reasoning(calls, answer),
        ],
        KimiK3Mode::Epilogue | KimiK3Mode::Tools | KimiK3Mode::Call { .. } | KimiK3Mode::Done => {
            unreachable!("initialization leaves the parser idle or inside `think` or `response`")
        }
    };
    elements.push(Format::optional(Format::const_string(MESSAGE_CLOSE)));

    Ok(Some(BuiltOutputGrammar::from_token_zero(Format::sequence(
        elements,
    ))))
}

/// Channels that may follow the closed `think` channel, given the tool
/// choice and `tools` channel when a tool is callable, and the user's answer
/// constraint for the `response` channel.
fn after_reasoning(calls: Option<(BuilderToolChoice, Format)>, answer: Option<&Format>) -> Format {
    let response = Format::tag(RESPONSE_OPEN, response_content(answer), RESPONSE_CLOSE);
    let Some((choice, tools)) = calls else {
        return response;
    };
    let empty = || Format::const_string(format!("{RESPONSE_OPEN}{RESPONSE_CLOSE}"));
    match (choice, answer) {
        (BuilderToolChoice::Auto, None) => Format::or(vec![
            Format::sequence(vec![response, Format::optional(tools.clone())]),
            tools,
        ]),
        (BuilderToolChoice::Auto, Some(_)) => Format::or(vec![
            Format::sequence(vec![response, Format::optional(tools.clone())]),
            Format::sequence(vec![Format::optional(empty()), tools]),
        ]),
        (BuilderToolChoice::Required | BuilderToolChoice::Forced, None) => {
            Format::sequence(vec![Format::optional(response), tools])
        }
        (BuilderToolChoice::Required | BuilderToolChoice::Forced, Some(_)) => {
            Format::sequence(vec![
                Format::optional(Format::or(vec![response, empty()])),
                tools,
            ])
        }
    }
}

/// The rest of the `response` channel the prompt opened, then the channels
/// that may follow it.
fn rest_of_response(
    calls: Option<(BuilderToolChoice, Format)>,
    answer: Option<&Format>,
) -> Vec<Format> {
    let response = Format::tag("", response_content(answer), RESPONSE_CLOSE);
    let Some((choice, tools)) = calls else {
        return vec![response];
    };
    let empty = || Format::const_string(RESPONSE_CLOSE);
    match (choice, answer) {
        (BuilderToolChoice::Auto, None) => vec![response, Format::optional(tools)],
        (BuilderToolChoice::Auto, Some(_)) => vec![Format::or(vec![
            Format::sequence(vec![response, Format::optional(tools.clone())]),
            Format::sequence(vec![empty(), tools]),
        ])],
        (BuilderToolChoice::Required | BuilderToolChoice::Forced, None) => vec![response, tools],
        (BuilderToolChoice::Required | BuilderToolChoice::Forced, Some(_)) => {
            vec![Format::or(vec![response, empty()]), tools]
        }
    }
}

/// The text of the `response` channel: the user's answer constraint, or free
/// text. A constrained answer rules out the empty `response` channel the chat
/// template writes before tool calls, so callers allow that channel where
/// calls follow.
fn response_content(answer: Option<&Format>) -> Format {
    answer.cloned().unwrap_or_else(response_text)
}

/// The rest of the `think` channel the prompt opened.
fn think_body() -> Format {
    Format::tag("", think_text(), THINK_CLOSE)
}

fn think_text() -> Format {
    Format::any_text_excluding(&[THINK_CLOSE, END_OF_MSG])
}

fn response_text() -> Format {
    // Keep marker-looking prefixes out of response text so the grammar must
    // resolve them through a valid channel boundary.
    Format::any_text_excluding(&[OPEN, CLOSE, END_OF_MSG])
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use serde_json::json;
    use xgrammar_structural_tag::ToolChoice;
    use xgrammar_structural_tag::format::{EndBoundary, TagFormat};

    use super::super::CALL_CLOSE;
    use super::*;
    use crate::output_grammar::test_utils::outline;
    use crate::output_grammar::{GrammarCoverage, ToolStrictLevel};
    use crate::tool::Tool;

    fn tools() -> Vec<Tool> {
        ["get_weather", "add"]
            .map(|name| Tool {
                name: name.to_string(),
                description: None,
                parameters: json!({ "type": "object", "properties": {} }),
                strict: None,
                defer_loading: None,
            })
            .into()
    }

    fn check(
        mode: KimiK3Mode,
        tool_choice: ToolChoice,
        parallel_tool_calls: bool,
        expected: Expect,
    ) {
        let tools = tools();
        let grammar = build_output_grammar(
            &mode,
            &OutputGrammarContext {
                tools: &tools,
                tool_choice: &tool_choice,
                tool_strict_level: ToolStrictLevel::Function,
                parallel_tool_calls,
                answer: None,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(grammar.coverage, GrammarCoverage::FromTokenZero);
        let mut format = grammar.format;
        elide_call_bodies(&mut format);
        expected.assert_eq(&outline(&format));
    }

    /// Like [`check`], for non-strict tools and a JSON answer constraint.
    fn check_answer(mode: KimiK3Mode, tools: &[Tool], tool_choice: ToolChoice, expected: Expect) {
        let answer = Format::json_schema(json!({ "type": "object" }));
        let grammar = build_output_grammar(
            &mode,
            &OutputGrammarContext {
                tools,
                tool_choice: &tool_choice,
                tool_strict_level: ToolStrictLevel::Auto,
                parallel_tool_calls: true,
                answer: Some(&answer),
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(grammar.coverage, GrammarCoverage::FromTokenZero);
        let mut format = grammar.format;
        elide_call_bodies(&mut format);
        expected.assert_eq(&outline(&format));
    }

    /// Replace each call body with `..`, so that outlines show the channel
    /// structure. The `arguments` tests cover call bodies.
    fn elide_call_bodies(format: &mut Format) {
        match format {
            Format::Sequence(format) => format.elements.iter_mut().for_each(elide_call_bodies),
            Format::Or(format) => format.elements.iter_mut().for_each(elide_call_bodies),
            Format::Optional(format) => elide_call_bodies(&mut format.content),
            Format::Tag(tag) => elide_tag(tag),
            Format::TagsWithSeparator(format) => format.tags.iter_mut().for_each(elide_tag),
            _ => {}
        }
    }

    fn elide_tag(tag: &mut TagFormat) {
        if matches!(&tag.end, EndBoundary::Text(end) if end == CALL_CLOSE) {
            *tag.content = Format::const_string("..");
        } else {
            elide_call_bodies(&mut tag.content);
        }
    }

    #[test]
    fn prompt_opened_think_channel() {
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::auto(),
            true,
            expect![[r#"
                sequence
                  tag `` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  or
                    sequence
                      tag `<|open|>response<|sep|>` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                      optional
                        tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                          tags_with_separator `` at_least_one
                            tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                            tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                    tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                      tags_with_separator `` at_least_one
                        tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                        tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::required(),
            true,
            expect![[r#"
                sequence
                  tag `` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  sequence
                    optional tag `<|open|>response<|sep|>` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                    tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                      tags_with_separator `` at_least_one
                        tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                        tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::function("add"),
            true,
            expect![[r#"
                sequence
                  tag `` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  sequence
                    optional tag `<|open|>response<|sep|>` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                    tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                      tags_with_separator `` at_least_one stop_after_first
                        tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
    }

    #[test]
    fn prompt_opened_response_channel() {
        check(
            KimiK3Mode::Response,
            ToolChoice::auto(),
            true,
            expect![[r#"
                sequence
                  tag `` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                  optional
                    tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                      tags_with_separator `` at_least_one
                        tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                        tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        check(
            KimiK3Mode::Response,
            ToolChoice::required(),
            true,
            expect![[r#"
                sequence
                  tag `` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                  tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                    tags_with_separator `` at_least_one
                      tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                      tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        check(
            KimiK3Mode::Response,
            ToolChoice::function("add"),
            true,
            expect![[r#"
                sequence
                  tag `` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                  tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                    tags_with_separator `` at_least_one stop_after_first
                      tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
    }

    #[test]
    fn no_prompt_opened_channel() {
        check(
            KimiK3Mode::Idle,
            ToolChoice::auto(),
            true,
            expect![[r#"
                sequence
                  optional tag `<|open|>think<|sep|>` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  or
                    sequence
                      tag `<|open|>response<|sep|>` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                      optional
                        tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                          tags_with_separator `` at_least_one
                            tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                            tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                    tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                      tags_with_separator `` at_least_one
                        tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                        tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
    }

    #[test]
    fn serial_tool_calls_stop_after_one_call() {
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::required(),
            false,
            expect![[r#"
                sequence
                  tag `` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  sequence
                    optional tag `<|open|>response<|sep|>` text excluding [`<|open|>`, `<|close|>`, `<|end_of_msg|>`] `<|close|>response<|sep|>`
                    tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                      tags_with_separator `` at_least_one stop_after_first
                        tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                        tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
    }

    #[test]
    fn non_strict_auto_builds_no_grammar() {
        let tools = tools();
        let grammar = build_output_grammar(
            &KimiK3Mode::Reasoning,
            &OutputGrammarContext {
                tools: &tools,
                tool_choice: &ToolChoice::auto(),
                tool_strict_level: ToolStrictLevel::Auto,
                parallel_tool_calls: true,
                answer: None,
            },
        )
        .unwrap();
        assert_eq!(grammar, None);
    }

    #[test]
    fn answer_constraint_fills_the_response_channel() {
        // Non-strict `auto` keeps the calls next to the constrained answer.
        // Calls may follow the empty `response` channel the chat template
        // writes before them.
        check_answer(
            KimiK3Mode::Reasoning,
            &tools(),
            ToolChoice::auto(),
            expect![[r#"
                sequence
                  tag `` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  or
                    sequence
                      tag `<|open|>response<|sep|>` json(object) `<|close|>response<|sep|>`
                      optional
                        tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                          tags_with_separator `` at_least_one
                            tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                            tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                    sequence
                      optional `<|open|>response<|sep|><|close|>response<|sep|>`
                      tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                        tags_with_separator `` at_least_one
                          tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                          tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        // A required call may follow a constrained answer.
        check_answer(
            KimiK3Mode::Reasoning,
            &tools(),
            ToolChoice::required(),
            expect![[r#"
                sequence
                  tag `` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  sequence
                    optional
                      or
                        tag `<|open|>response<|sep|>` json(object) `<|close|>response<|sep|>`
                        `<|open|>response<|sep|><|close|>response<|sep|>`
                    tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                      tags_with_separator `` at_least_one
                        tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                        tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        check_answer(
            KimiK3Mode::Response,
            &tools(),
            ToolChoice::auto(),
            expect![[r#"
                sequence
                  or
                    sequence
                      tag `` json(object) `<|close|>response<|sep|>`
                      optional
                        tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                          tags_with_separator `` at_least_one
                            tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                            tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                    sequence
                      `<|close|>response<|sep|>`
                      tag `<|open|>tools<|sep|>` .. `<|close|>tools<|sep|>`
                        tags_with_separator `` at_least_one
                          tag `<|open|>call tool="get_weather" index="` `..` `<|close|>call<|sep|>`
                          tag `<|open|>call tool="add" index="` `..` `<|close|>call<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
    }

    #[test]
    fn answer_constraint_without_a_callable_tool_requires_the_response() {
        check_answer(
            KimiK3Mode::Idle,
            &tools(),
            ToolChoice::none(),
            expect![[r#"
                sequence
                  optional tag `<|open|>think<|sep|>` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  tag `<|open|>response<|sep|>` json(object) `<|close|>response<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        check_answer(
            KimiK3Mode::Reasoning,
            &[],
            ToolChoice::auto(),
            expect![[r#"
                sequence
                  tag `` text excluding [`<|close|>think<|sep|>`, `<|end_of_msg|>`] `<|close|>think<|sep|>`
                  tag `<|open|>response<|sep|>` json(object) `<|close|>response<|sep|>`
                  optional `<|close|>message<|sep|>`
            "#]],
        );
        check_answer(
            KimiK3Mode::Response,
            &[],
            ToolChoice::auto(),
            expect![[r#"
            sequence
              tag `` json(object) `<|close|>response<|sep|>`
              optional `<|close|>message<|sep|>`
        "#]],
        );
    }
}
