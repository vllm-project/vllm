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

mod arguments;

use xgrammar_structural_tag::format::Format;
use xgrammar_structural_tag::tool::BuilderToolChoice;

use self::arguments::call_tag;
use super::{
    CLOSE, END_OF_MSG, KimiK3Mode, MESSAGE_CLOSE, OPEN, RESPONSE_CLOSE, RESPONSE_OPEN, THINK_CLOSE,
    THINK_OPEN, TOOLS_CLOSE, TOOLS_OPEN,
};
use crate::output_grammar::{
    self, BuiltOutputGrammar, NormalizedToolChoice, OutputGrammarContext, normalize_tool_choice,
};

/// Build the grammar of everything generated after a prompt that left the
/// parser in `mode`, or `None` when the request asks for no tool grammar.
pub(super) fn build_output_grammar(
    mode: &KimiK3Mode,
    ctx: &OutputGrammarContext<'_>,
) -> output_grammar::Result<Option<BuiltOutputGrammar>> {
    let Some(NormalizedToolChoice {
        function_tools,
        tool_choice,
    }) = normalize_tool_choice(ctx)?
    else {
        return Ok(None);
    };

    let single_call = tool_choice == BuilderToolChoice::Forced || !ctx.parallel_tool_calls;
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

    let mut elements = match mode {
        KimiK3Mode::Reasoning => vec![think_body(), after_reasoning(tool_choice, tools)],
        KimiK3Mode::Response => vec![response_body(), after_response(tool_choice, tools)],
        KimiK3Mode::Idle => vec![
            Format::optional(Format::tag(THINK_OPEN, think_text(), THINK_CLOSE)),
            after_reasoning(tool_choice, tools),
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

/// Channels that may follow the closed `think` channel.
fn after_reasoning(tool_choice: BuilderToolChoice, tools: Format) -> Format {
    let response = Format::tag(RESPONSE_OPEN, response_text(), RESPONSE_CLOSE);
    match tool_choice {
        BuilderToolChoice::Auto => Format::or(vec![
            Format::sequence(vec![response, Format::optional(tools.clone())]),
            tools,
        ]),
        BuilderToolChoice::Required | BuilderToolChoice::Forced => {
            Format::sequence(vec![Format::optional(response), tools])
        }
    }
}

/// Channels that may follow the closed `response` channel.
fn after_response(tool_choice: BuilderToolChoice, tools: Format) -> Format {
    match tool_choice {
        BuilderToolChoice::Auto => Format::optional(tools),
        BuilderToolChoice::Required | BuilderToolChoice::Forced => tools,
    }
}

/// The rest of the `think` channel the prompt opened.
fn think_body() -> Format {
    Format::tag("", think_text(), THINK_CLOSE)
}

/// The rest of the `response` channel the prompt opened.
fn response_body() -> Format {
    Format::tag("", response_text(), RESPONSE_CLOSE)
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
    use xgrammar_structural_tag::format::{EndBoundary, TagBoundary, TagFormat};

    use super::*;
    use crate::output_grammar::{GrammarCoverage, ToolStrictLevel};
    use crate::tool::Tool;

    fn tools() -> Vec<Tool> {
        ["get_weather", "add"]
            .map(|name| Tool {
                name: name.to_string(),
                description: None,
                parameters: json!({ "type": "object", "properties": {} }),
                strict: None,
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
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(grammar.coverage, GrammarCoverage::FromTokenZero);
        expected.assert_eq(&outline(&grammar.format));
    }

    /// Render the channel structure of a grammar, abbreviating XTML markers to
    /// `<name>` / `</name>`, free text to `text`, and call tags to `call(name)`.
    fn outline(format: &Format) -> String {
        match format {
            Format::Sequence(format) => {
                format.elements.iter().map(outline).collect::<Vec<_>>().join(" ")
            }
            Format::Or(format) => format!(
                "({})",
                format.elements.iter().map(outline).collect::<Vec<_>>().join(" | ")
            ),
            Format::Optional(format) => format!("[{}]", outline(&format.content)),
            Format::ConstString(format) => markers(&format.value),
            Format::AnyText(_) => "text".to_string(),
            Format::Tag(tag) => tag_outline(tag),
            Format::TagsWithSeparator(format) => {
                let calls = format.tags.iter().map(tag_outline).collect::<Vec<_>>();
                let repeat = match (format.at_least_one, format.stop_after_first) {
                    (true, true) => "",
                    (true, false) => "+",
                    (false, true) => "?",
                    (false, false) => "*",
                };
                format!("({}){repeat}", calls.join(" | "))
            }
            format => panic!("unexpected channel format: {format:?}"),
        }
    }

    fn tag_outline(tag: &TagFormat) -> String {
        let (TagBoundary::Text(begin), EndBoundary::Text(end)) = (&tag.begin, &tag.end) else {
            panic!("unexpected tag boundaries: {tag:?}");
        };
        if let Some(name) = begin.strip_prefix("<|open|>call tool=\"") {
            return format!("call({})", name.trim_end_matches("\" index=\""));
        }
        format!(
            "{}{}{}",
            markers(begin),
            outline(&tag.content),
            markers(end)
        )
    }

    fn markers(text: &str) -> String {
        text.replace(OPEN, "<").replace(CLOSE, "</").replace("<|sep|>", ">")
    }

    #[test]
    fn prompt_opened_think_channel() {
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::auto(),
            true,
            expect![
                "text</think> (<response>text</response> [<tools>(call(get_weather) | call(add))+</tools>] | <tools>(call(get_weather) | call(add))+</tools>) [</message>]"
            ],
        );
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::required(),
            true,
            expect![
                "text</think> [<response>text</response>] <tools>(call(get_weather) | call(add))+</tools> [</message>]"
            ],
        );
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::function("add"),
            true,
            expect![
                "text</think> [<response>text</response>] <tools>(call(add))</tools> [</message>]"
            ],
        );
    }

    #[test]
    fn prompt_opened_response_channel() {
        check(
            KimiK3Mode::Response,
            ToolChoice::auto(),
            true,
            expect![
                "text</response> [<tools>(call(get_weather) | call(add))+</tools>] [</message>]"
            ],
        );
        check(
            KimiK3Mode::Response,
            ToolChoice::required(),
            true,
            expect!["text</response> <tools>(call(get_weather) | call(add))+</tools> [</message>]"],
        );
        check(
            KimiK3Mode::Response,
            ToolChoice::function("add"),
            true,
            expect!["text</response> <tools>(call(add))</tools> [</message>]"],
        );
    }

    #[test]
    fn no_prompt_opened_channel() {
        check(
            KimiK3Mode::Idle,
            ToolChoice::auto(),
            true,
            expect![
                "[<think>text</think>] (<response>text</response> [<tools>(call(get_weather) | call(add))+</tools>] | <tools>(call(get_weather) | call(add))+</tools>) [</message>]"
            ],
        );
    }

    #[test]
    fn serial_tool_calls_stop_after_one_call() {
        check(
            KimiK3Mode::Reasoning,
            ToolChoice::required(),
            false,
            expect![
                "text</think> [<response>text</response>] <tools>(call(get_weather) | call(add))</tools> [</message>]"
            ],
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
            },
        )
        .unwrap();
        assert_eq!(grammar, None);
    }
}
