// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared parser core for pythonic tool calls.

mod llama4;
mod value;

pub use llama4::Llama4PythonicToolParser;
use value::{
    StringRun, decode_string_run, json_object_key, json_string_content, python_value, string_quote,
};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, opt, preceded, seq};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, one_of, take_while};

use super::utils::{incomplete, parse_buffered_event};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

type PythonicInput<'i> = Partial<&'i str>;

/// Model-specific configuration for the shared pythonic grammar.
///
/// Only the optional markers wrapping the tool-call list vary across models
/// that reuse this grammar; the list itself is byte-identical.
#[derive(Debug, Clone, Copy)]
struct PythonicConfig {
    /// Human-readable parser name used in error messages.
    parser_name: &'static str,
    /// Marker that may precede the tool-call list.
    start_marker: Option<&'static str>,
    /// Marker that may follow the tool-call list.
    end_marker: Option<&'static str>,
}

const PYTHONIC_CONFIG: PythonicConfig = PythonicConfig {
    parser_name: "pythonic",
    start_marker: None,
    end_marker: None,
};

const LLAMA4_PYTHONIC_CONFIG: PythonicConfig = PythonicConfig {
    parser_name: "Llama 4 pythonic",
    start_marker: Some("<|python_start|>"),
    end_marker: Some("<|python_end|>"),
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum PythonicMode {
    Start,
    Passthrough,
    ListStart,
    ListNext,
    Arguments { first: bool },
    StringValue { quote: char },
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PythonicEvent {
    CallStart {
        name: String,
    },
    Argument {
        key: String,
        value: serde_json::Value,
    },
    StringArgumentStart {
        key: String,
        quote: char,
    },
    StringChunk {
        text: String,
    },
    StringEnd {
        text: String,
    },
    CallEnd,
    ListEnd,
}

/// Tool parser for pythonic tool calls.
///
/// The whole assistant output is a Python list of keyword-argument calls, as
/// produced by Llama 3.2, Llama 4 and ToolACE:
///
/// ```text
/// [get_weather(city='San Francisco', metric='celsius'), get_weather(city='New York', metric='celsius')]
/// ```
///
/// Argument values are Python literals (strings, numbers, `True` / `False` /
/// `None`, lists and dicts) and are converted to JSON while streaming: the
/// function name is emitted as soon as `name(` is parsed, then `{`, then one
/// `"key":<json>` fragment per completed argument, then `}` at `)`. String
/// values are streamed as they arrive, so a long string argument (file
/// contents, code) does not stall the stream. JSON-style `true` / `false` /
/// `null` are accepted as well, mirroring `_JSON_NAME_LITERALS` in the Python
/// parser.
///
/// Natural text at the beginning of the stream permanently disables tool
/// parsing for that assistant output, like `Llama3JsonToolParser` does: the
/// model must not mix text and tool calls in one generation. Whitespace before
/// the opening `[` is tolerated, which the Python `TOOL_CALL_REGEX` match does
/// not do.
///
/// Text that opens with `[` but is not a list of calls (`[1, 2, 3]`, `[]`,
/// `[not a call`) is a parse error rather than plain text, matching the JSON
/// parsers of this frontend. The streaming layer recovers such errors by
/// re-emitting the buffered text as content, which is what the Python parser
/// does for a whole output that does not match its tool-call pattern.
///
/// Structured-output tags are intentionally unsupported: the trait default
/// `structural_tag_builder() -> None` is kept because xgrammar has no
/// structural tag for the pythonic format, and the Python parser leaves
/// `structural_tag_model = None` as well.
pub struct PythonicToolParser {
    buffer: String,
    mode: PythonicMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
    config: PythonicConfig,
}

impl PythonicToolParser {
    /// Create a pythonic tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self::with_config(PYTHONIC_CONFIG)
    }

    /// Create a parser for a model that wraps the same list in markers.
    fn with_config(config: PythonicConfig) -> Self {
        Self {
            buffer: String::new(),
            mode: PythonicMode::Start,
            active_tool_index: None,
            emitted_tool_count: 0,
            config,
        }
    }

    /// Commit the stream to pythonic parsing or permanent passthrough.
    fn commit_start(&mut self) -> bool {
        if !matches!(self.mode, PythonicMode::Start) {
            return true;
        }

        let Some(rest) = self.strip_start_marker(self.buffer.trim_start()) else {
            return false;
        };
        let Some(first) = rest.trim_start().chars().next() else {
            return false;
        };

        self.mode = if first == '[' {
            PythonicMode::ListStart
        } else {
            PythonicMode::Passthrough
        };
        true
    }

    /// Strip the configured start marker from buffered text.
    ///
    /// Returns `None` while the buffered text may still grow into the marker.
    fn strip_start_marker<'a>(&self, text: &'a str) -> Option<&'a str> {
        let Some(marker) = self.config.start_marker else {
            return Some(text);
        };
        match text.strip_prefix(marker) {
            Some(rest) => Some(rest),
            None if marker.starts_with(text) => None,
            None => Some(text),
        }
    }

    /// Strip the configured end marker from text after the tool-call list.
    fn strip_end_marker<'a>(&self, text: &'a str) -> &'a str {
        let Some(marker) = self.config.end_marker else {
            return text;
        };
        text.strip_suffix(marker).unwrap_or(text).trim_end()
    }

    /// Apply one parsed pythonic event to parser state and output.
    fn apply_event(&mut self, event: PythonicEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            PythonicEvent::CallStart { name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = PythonicMode::Arguments { first: true };
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments: String::new(),
                });
                self.push_arguments("{", output)?;
            }
            PythonicEvent::Argument { key, value } => {
                let key = json_object_key(&key)?;
                let value = serde_json::to_string(&value)
                    .map_err(|error| parsing_failed!("failed to serialize argument: {}", error))?;
                let fragment = format!("{}{key}:{value}", self.argument_separator());
                self.push_arguments(&fragment, output)?;
                self.mode = PythonicMode::Arguments { first: false };
            }
            PythonicEvent::StringArgumentStart { key, quote } => {
                let key = json_object_key(&key)?;
                let fragment = format!("{}{key}:\"", self.argument_separator());
                self.push_arguments(&fragment, output)?;
                self.mode = PythonicMode::StringValue { quote };
            }
            PythonicEvent::StringChunk { text } => {
                self.push_arguments(&json_string_content(&text)?, output)?;
            }
            PythonicEvent::StringEnd { text } => {
                let fragment = format!("{}\"", json_string_content(&text)?);
                self.push_arguments(&fragment, output)?;
                self.mode = PythonicMode::Arguments { first: false };
            }
            PythonicEvent::CallEnd => {
                self.push_arguments("}", output)?;
                self.active_tool_index = None;
                self.mode = PythonicMode::ListNext;
            }
            PythonicEvent::ListEnd => {
                self.mode = PythonicMode::Done;
            }
        }
        Ok(())
    }

    /// Return the JSON separator that precedes the next argument.
    fn argument_separator(&self) -> &'static str {
        match self.mode {
            PythonicMode::Arguments { first: false } => ",",
            _ => "",
        }
    }

    /// Append one argument-JSON fragment for the active tool call.
    fn push_arguments(&self, arguments: &str, output: &mut ToolParserOutput) -> Result<()> {
        let Some(tool_index) = self.active_tool_index else {
            return Err(parsing_failed!(
                "pythonic arguments without an active tool call"
            ));
        };
        if arguments.is_empty() {
            return Ok(());
        }
        output.push_call(ToolCallDelta {
            tool_index,
            name: None,
            arguments: arguments.to_string(),
        });
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = PythonicMode::Start;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for PythonicToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        if !self.commit_start() {
            return Ok(());
        }

        if matches!(self.mode, PythonicMode::Passthrough) {
            output.push_text(&self.buffer);
            self.buffer.clear();
            return Ok(());
        }

        let config = self.config;
        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_pythonic_event(input, &self.mode, config)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            PythonicMode::Start | PythonicMode::Passthrough => output.push_text(&self.buffer),
            PythonicMode::Done => {
                if !self.strip_end_marker(self.buffer.trim()).is_empty() {
                    return Err(parsing_failed!(
                        "trailing text after {} tool calls",
                        self.config.parser_name
                    ));
                }
            }
            _ => {
                return Err(parsing_failed!(
                    "incomplete {} tool call",
                    self.config.parser_name
                ));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        PythonicToolParser::reset(self)
    }
}

/// Parse a pythonic event for the current parser mode.
fn parse_next_pythonic_event(
    input: &mut PythonicInput<'_>,
    mode: &PythonicMode,
    config: PythonicConfig,
) -> ModalResult<PythonicEvent> {
    match mode {
        PythonicMode::Start | PythonicMode::Passthrough => {
            unreachable!("pythonic parser driver must commit before parsing events")
        }
        PythonicMode::ListStart => list_start_event(input, config),
        PythonicMode::ListNext => list_next_event(input),
        PythonicMode::Arguments { first } => arguments_event(input, *first),
        PythonicMode::StringValue { quote } => string_value_event(input, *quote),
        // Text after the closing `]` is only reported once, at `finish()`.
        PythonicMode::Done => incomplete(),
    }
}

/// Parse the opening `[` and the first call of a pythonic tool-call list.
fn list_start_event(
    input: &mut PythonicInput<'_>,
    config: PythonicConfig,
) -> ModalResult<PythonicEvent> {
    preceded(
        (ws0, optional_marker(config.start_marker), literal("["), ws0),
        call_start_event,
    )
    .context(StrContext::Label("pythonic tool call list"))
    .parse_next(input)
}

/// Parse the separator or terminator after one pythonic call.
fn list_next_event(input: &mut PythonicInput<'_>) -> ModalResult<PythonicEvent> {
    preceded(
        ws0,
        alt((
            list_end_event,
            preceded((literal(","), ws0), alt((list_end_event, call_start_event))),
        )),
    )
    .parse_next(input)
}

/// Parse the closing `]` of a pythonic tool-call list.
fn list_end_event(input: &mut PythonicInput<'_>) -> ModalResult<PythonicEvent> {
    literal("]").value(PythonicEvent::ListEnd).parse_next(input)
}

/// Parse the start of one pythonic call, up to its opening `(`.
fn call_start_event(input: &mut PythonicInput<'_>) -> ModalResult<PythonicEvent> {
    seq!(function_name, _: literal("("))
        .map(|(name,)| PythonicEvent::CallStart {
            name: name.to_string(),
        })
        .parse_next(input)
}

/// Parse the next event inside a pythonic argument list.
fn arguments_event(input: &mut PythonicInput<'_>, first: bool) -> ModalResult<PythonicEvent> {
    if first {
        return preceded(ws0, alt((call_end_event, argument_event)))
            .context(StrContext::Label("pythonic tool call arguments"))
            .parse_next(input);
    }
    preceded(
        ws0,
        alt((
            call_end_event,
            preceded((literal(","), ws0), alt((call_end_event, argument_event))),
        )),
    )
    .context(StrContext::Label("pythonic tool call arguments"))
    .parse_next(input)
}

/// Parse the closing `)` of one pythonic call.
fn call_end_event(input: &mut PythonicInput<'_>) -> ModalResult<PythonicEvent> {
    literal(")").value(PythonicEvent::CallEnd).parse_next(input)
}

/// Parse one keyword argument of a pythonic call, up to its value.
fn argument_event(input: &mut PythonicInput<'_>) -> ModalResult<PythonicEvent> {
    let (key,) = seq!(identifier, _: ws0, _: literal("="), _: ws0).parse_next(input)?;
    let key = key.to_string();

    // A string value is streamed as it arrives; any other literal is only
    // emitted once it is complete.
    alt((
        string_quote.map(|quote| PythonicEvent::StringArgumentStart {
            key: key.clone(),
            quote,
        }),
        python_value.map(|value| PythonicEvent::Argument {
            key: key.clone(),
            value,
        }),
    ))
    .parse_next(input)
}

/// Parse the next run of a streaming pythonic string argument.
fn string_value_event(input: &mut PythonicInput<'_>, quote: char) -> ModalResult<PythonicEvent> {
    let StringRun {
        text,
        consumed,
        closed,
    } = decode_string_run(input, quote)?;

    if consumed == 0 {
        return incomplete();
    }
    Ok(if closed {
        PythonicEvent::StringEnd { text }
    } else {
        PythonicEvent::StringChunk { text }
    })
}

/// Parse an optional marker wrapping the tool-call list.
fn optional_marker<'i>(
    marker: Option<&'static str>,
) -> impl Parser<PythonicInput<'i>, (), ErrMode<ContextError>> {
    move |input: &mut PythonicInput<'i>| match marker {
        Some(marker) => opt(literal(marker)).void().parse_next(input),
        None => Ok(()),
    }
}

/// Parse a pythonic function name.
///
/// Dotted names are accepted because the Python parser keeps them as the tool
/// name (`handle_single_tool` walks `ast.Attribute` chains).
fn function_name<'i>(input: &mut PythonicInput<'i>) -> ModalResult<&'i str> {
    (
        one_of(('a'..='z', 'A'..='Z', '_')),
        take_while(0.., ('a'..='z', 'A'..='Z', '0'..='9', '_', '.')),
    )
        .take()
        .parse_next(input)
}

/// Parse a Python identifier.
fn identifier<'i>(input: &mut PythonicInput<'i>) -> ModalResult<&'i str> {
    (
        one_of(('a'..='z', 'A'..='Z', '_')),
        take_while(0.., ('a'..='z', 'A'..='Z', '0'..='9', '_')),
    )
        .take()
        .parse_next(input)
}

/// Feed `chunks` through a parser, recovering parse errors as plain text the
/// way the streaming frontend does.
#[cfg(test)]
fn parse_recovering(config: PythonicConfig, chunks: &[&str]) -> ToolParserOutput {
    let mut parser = PythonicToolParser::with_config(config);
    let mut output = ToolParserOutput::default();
    let mut failed = false;

    for chunk in chunks {
        if failed {
            output.push_text(*chunk);
            continue;
        }
        if parser.parse_into(chunk, &mut output).is_err() {
            failed = true;
            output.push_text(parser.reset());
        }
    }

    if !failed {
        match parser.finish() {
            Ok(finished) => output.append(finished),
            Err(_) => output.push_text(parser.reset()),
        }
    }
    output.coalesce()
}

/// Parse `text` whole and in 1- and 3-character chunks, asserting that every
/// chunking commits the same output.
#[cfg(test)]
fn parse_chunkings(config: PythonicConfig, text: &str) -> ToolParserOutput {
    use crate::tool::test_utils::split_by_chars;

    let whole = parse_recovering(config, &[text]);
    for chunk_chars in [1, 3] {
        let chunked = parse_recovering(config, &split_by_chars(text, chunk_chars));
        assert_eq!(chunked, whole, "{chunk_chars}-char chunks must match");
    }
    whole
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::{PYTHONIC_CONFIG, PythonicToolParser, parse_chunkings};
    use crate::tool::test_utils::test_tools;
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    // Fixtures shared with `tests/tool_parsers/test_pythonic_tool_parser.py`.
    const SIMPLE_CALL: &str = "get_weather(city='San Francisco', metric='celsius')";
    const MORE_TYPES_CALL: &str = "register_user(name='John Doe', \
         age=37, \
         address={'city': 'San Francisco', 'state': 'CA'}, \
         role=None, \
         passed_test=True, \
         aliases=['John', 'Johnny'])";
    const ESCAPED_STRING_CALL: &str =
        r#"get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')"#;

    fn parse(text: &str) -> ToolParserOutput {
        parse_chunkings(PYTHONIC_CONFIG, text)
    }

    #[test]
    fn pythonic_extracts_simple_call() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&format!("[{SIMPLE_CALL}]")));
    }

    #[test]
    fn pythonic_extracts_parallel_calls() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "register_user",
                            ),
                            arguments: "{\"name\":\"John Doe\",\"age\":37,\"address\":{\"city\":\"San Francisco\",\"state\":\"CA\"},\"role\":null,\"passed_test\":true,\"aliases\":[\"John\",\"Johnny\"]}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(&format!("[{SIMPLE_CALL}, {MORE_TYPES_CALL}]")));
    }

    #[test]
    fn pythonic_converts_python_literals_to_json() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "register_user",
                            ),
                            arguments: "{\"name\":\"John Doe\",\"age\":37,\"address\":{\"city\":\"San Francisco\",\"state\":\"CA\"},\"role\":null,\"passed_test\":true,\"aliases\":[\"John\",\"Johnny\"]}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(&format!("[{MORE_TYPES_CALL}]")));
    }

    #[test]
    fn pythonic_extracts_empty_arguments() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "do_something_cool",
                            ),
                            arguments: "{\"additional_data\":{}}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 2,
                            name: Some(
                                "do_something_cool",
                            ),
                            arguments: "{\"steps\":[]}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(
            "[get_weather(), do_something_cool(additional_data={}), do_something_cool(steps=[])]",
        ));
    }

    #[test]
    fn pythonic_decodes_escaped_strings() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"Martha's Vineyard\",\"metric\":\"\\\"cool units\\\"\"}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(&format!("[{ESCAPED_STRING_CALL}]")));
    }

    #[test]
    fn pythonic_decodes_escapes_split_across_chunks() {
        // Every escape form must survive arriving one character at a time.
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "send",
                            ),
                            arguments: "{\"text\":\"☃ 😀 🚀 AA é\",\"raw\":\"a\\tb\\nc\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(
            r"[send(text='☃ \U0001f600 🚀 \x41\101 é', raw='a\tb\nc')]",
        ));
    }

    #[test]
    fn pythonic_keeps_brackets_inside_string_arguments() {
        // Brackets, commas and quotes inside a string must not end the call.
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "run",
                            ),
                            arguments: "{\"cmd\":\"grep -F \\\"]\\\" a.txt\",\"note\":\"one, two)\",\"flag\":true}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(
            r#"[run(cmd='grep -F "]" a.txt', note="one, two)", flag=True)]"#,
        ));
    }

    #[test]
    fn pythonic_accepts_signed_numbers() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "locate",
                            ),
                            arguments: "{\"lat\":-33.87,\"lon\":151,\"delta\":-2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse("[locate(lat=-33.87, lon=151, delta=-2)]"));
    }

    #[test]
    fn pythonic_accepts_whitespace_between_calls_and_arguments() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\":1,\"y\":2}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(
            "[\n  get_weather(\n    city = 'San Francisco',\n    metric = 'celsius',\n  ),\n  add(x=1, y=2),\n]",
        ));
    }

    #[test]
    fn pythonic_keeps_plain_text() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "How can I help you today?",
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse("How can I help you today?"));
    }

    #[test]
    fn pythonic_passthrough_never_reenters_tool_parsing() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "Let me check. [get_weather(city='San Francisco', metric='celsius')]",
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&format!("Let me check. [{SIMPLE_CALL}]")));
    }

    #[test]
    fn pythonic_recovers_lists_that_are_not_tool_calls() {
        for text in ["[1, 2, 3]", "[not a call", "[]", "[3.14]"] {
            let output = parse(text);
            assert_eq!(output.normal_text(), text, "{text:?} should stay text");
            assert!(output.calls().is_empty(), "{text:?} should have no calls");
        }
    }

    #[test]
    fn pythonic_streams_name_and_argument_fragments() {
        let mut parser = PythonicToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        parser.parse_into(&format!("[{SIMPLE_CALL}]"), &mut output).unwrap();
        output.append(parser.finish().unwrap());

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "{",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "\"city\":\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "San Francisco\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: ",\"metric\":\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "celsius\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn pythonic_streams_long_string_arguments_before_the_call_closes() {
        let mut parser = PythonicToolParser::new(&test_tools());
        let long_value = "x".repeat(4096);

        let output = parser.parse_chunk(&format!("[run(script='{long_value}")).unwrap();

        // The string content is committed before the call is closed.
        assert_eq!(output.calls().len(), 4);
        assert_eq!(output.calls()[0].name.as_deref(), Some("run"));
        assert_eq!(output.calls()[1].arguments, "{");
        assert_eq!(output.calls()[2].arguments, "\"script\":\"");
        assert_eq!(output.calls()[3].arguments, long_value);
    }

    #[test]
    fn pythonic_finish_fails_incomplete_call() {
        let mut parser = PythonicToolParser::new(&test_tools());
        parser.parse_chunk("[get_weather(city='San").unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete pythonic tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn pythonic_rejects_trailing_text_after_the_list() {
        let mut parser = PythonicToolParser::new(&test_tools());
        parser.parse_chunk(&format!("[{SIMPLE_CALL}] and that is all")).unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: trailing text after pythonic tool calls"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn pythonic_reports_malformed_arguments() {
        let mut parser = PythonicToolParser::new(&test_tools());

        let error = parser.parse_chunk("[get_weather(city=@)]").unwrap_err();

        expect![[
            r#"tool parser parsing failed: near "city=@)]": invalid pythonic tool call arguments"#
        ]]
        .assert_eq(&error.to_report_string());
    }
}
