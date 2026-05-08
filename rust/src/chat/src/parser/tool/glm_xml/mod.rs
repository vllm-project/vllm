use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, eof, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until, take_while};

use super::parameters::ToolSchemas;
use super::utils::{parse_buffered_event, safe_text_len, xml_unescape};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParserError, parsing_failed};
use crate::request::ChatTool;

mod glm45_moe;
mod glm47_moe;

pub use glm45_moe::Glm45MoeToolParser;
pub use glm47_moe::Glm47MoeToolParser;

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const ARG_KEY_START: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE_START: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

type GlmInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlmMode {
    Text,
    ToolCall,
    AfterToolCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Separator {
    /// GLM-4.5/4.6 format: function name must end at a newline before
    /// arguments.
    Newline,
    /// GLM-4.7 format: function name may end at whitespace or directly before
    /// `<arg_key>`.
    Flexible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GlmEvent {
    Text {
        len: usize,
    },
    ToolCallStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    IgnoredRest,
}

/// Tool parser core for GLM XML-style tool calls.
struct GlmXmlToolParser {
    buffer: String,
    mode: GlmMode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
    separator: Separator,
}

impl GlmXmlToolParser {
    /// Create a GLM XML tool parser with a function-name separator.
    fn new(tools: &[ChatTool], separator: Separator) -> Self {
        Self {
            buffer: String::new(),
            mode: GlmMode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            separator,
        }
    }

    /// Apply one parsed GLM event to parser state and output.
    fn apply_event(&mut self, event: GlmEvent, result: &mut ToolParseResult) -> Result<()> {
        match event {
            GlmEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            GlmEvent::ToolCallStart => self.mode = GlmMode::ToolCall,
            GlmEvent::ToolCall { name, raw_params } => {
                self.mode = GlmMode::AfterToolCall;
                let arguments = self.tool_parameters.convert_params_with_schema(&name, raw_params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                result.calls.push(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
            GlmEvent::IgnoredRest => {}
        }
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = GlmMode::Text;
        self.emitted_tool_count = 0;
    }

    /// Push one decoded text chunk through the GLM MoE parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_glm_event(input, self.mode, self.separator)
        })? {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        if !self.buffer.is_empty() {
            match self.mode {
                GlmMode::Text => result.normal_text.push_str(&self.buffer),
                GlmMode::ToolCall => return Err(parsing_failed!("incomplete GLM MoE tool call")),
                GlmMode::AfterToolCall => {}
            }
        }
        self.reset();
        Ok(result)
    }
}

/// Parse a GLM event for the current parser mode.
fn parse_next_glm_event(
    input: &mut GlmInput<'_>,
    mode: GlmMode,
    separator: Separator,
) -> ModalResult<GlmEvent> {
    match mode {
        GlmMode::Text => parse_text_event(input),
        GlmMode::ToolCall => tool_call_event(input, separator),
        GlmMode::AfterToolCall => after_tool_call_event(input),
    }
}

/// Parse a text-mode GLM event.
fn parse_text_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    alt((tool_call_start_event, safe_text_event)).parse_next(input)
}

/// Parse a GLM tool-call start marker.
fn tool_call_start_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    literal(TOOL_CALL_START).value(GlmEvent::ToolCallStart).parse_next(input)
}

/// Parse a safe text run before the next GLM marker.
fn safe_text_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    safe_text_len(input, TOOL_CALL_START).map(|len| GlmEvent::Text { len })
}

/// Parse text after a completed GLM tool call.
fn after_tool_call_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    ws0.void().parse_next(input)?;
    alt((tool_call_start_event, ignored_rest_event)).parse_next(input)
}

/// Parse a trailing rest after GLM tool calls.
fn ignored_rest_event(input: &mut GlmInput<'_>) -> ModalResult<GlmEvent> {
    rest.value(GlmEvent::IgnoredRest).parse_next(input)
}

/// Parse a complete GLM tool call.
fn tool_call_event(input: &mut GlmInput<'_>, separator: Separator) -> ModalResult<GlmEvent> {
    let (body,) = seq!(
        take_until(0.., TOOL_CALL_END),
        _: literal(TOOL_CALL_END),
    )
    .parse_next(input)?;

    parse_tool_call_body(body, separator)
}

/// Parse a GLM tool-call body.
fn parse_tool_call_body(body: &str, separator: Separator) -> ModalResult<GlmEvent> {
    let mut input = body;
    let (name, raw_params) = match separator {
        Separator::Newline => seq!(
            _: ws0,
            parse_newline_separated_function_name,
            parse_parameters,
            _: ws0,
            _: eof,
        )
        .parse_next(&mut input)?,
        Separator::Flexible => seq!(
            _: ws0,
            parse_flexible_function_name,
            parse_parameters,
            _: ws0,
            _: eof,
        )
        .parse_next(&mut input)?,
    };

    Ok(GlmEvent::ToolCall {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a GLM-4.5 newline-separated function name.
fn parse_newline_separated_function_name<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    terminated(take_until(1.., "\n"), "\n").map(str::trim).parse_next(input)
}

/// Parse a GLM-4.7 whitespace-or-tag-separated function name.
fn parse_flexible_function_name<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    terminated(
        take_while(1.., |ch: char| !ch.is_whitespace() && ch != '<'),
        ws0,
    )
    .parse_next(input)
}

/// Parse GLM argument key-value pairs.
fn parse_parameters(input: &mut &str) -> ModalResult<Vec<(String, String)>> {
    repeat(0.., terminated(parse_parameter, ws0)).parse_next(input)
}

/// Parse a GLM argument key-value pair.
fn parse_parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (key, value) = seq!(
        _: literal(ARG_KEY_START),
        take_until(1.., ARG_KEY_END),
        _: literal(ARG_KEY_END),
        _: ws0,
        _: literal(ARG_VALUE_START),
        take_until(0.., ARG_VALUE_END).map(str::trim).map(xml_unescape),
        _: literal(ARG_VALUE_END),
    )
    .parse_next(input)?;

    Ok((key.trim().to_string(), value.into_owned()))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::Glm45MoeToolParser;
    use crate::parser::tool::ToolParser;
    use crate::parser::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn glm45_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| {
                format!("<arg_key>{name}</arg_key>\n<arg_value>{value}</arg_value>")
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("<tool_call>{function_name}\n{params}\n</tool_call>")
    }

    #[test]
    fn glm45_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn glm45_parse_complete_extracts_single_tool_call() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let output = format!(
            "Let me search for that.\n{}",
            glm45_tool_call(
                "get_weather",
                &[("city", "Beijing"), ("date", "2024-12-25")]
            )
        );

        let result = parser.parse_complete(&output).unwrap();

        assert_eq!(result.normal_text, "Let me search for that.\n");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({"city": "Beijing", "date": "2024-12-25"})
        );
    }

    #[test]
    fn glm45_streaming_extracts_multiple_tool_calls() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let output = format!(
            "{}\n{}",
            glm45_tool_call("get_weather", &[("city", "Shanghai")]),
            glm45_tool_call("add", &[("x", "1"), ("y", "2")])
        );

        let chunks = split_by_chars(&output, 11);
        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "");
        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[1].name.as_deref(), Some("add"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[1].arguments).unwrap(),
            json!({"x": 1, "y": 2})
        );
    }

    #[test]
    fn glm45_parse_complete_unescapes_literal_closing_tags_in_arg_value() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&glm45_tool_call(
                "get_weather",
                &[
                    ("city", "Paris &lt;/arg_value&gt;&lt;/tool_call&gt;"),
                    ("date", "2026-05-08"),
                ],
            ))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "city": "Paris </arg_value></tool_call>",
                "date": "2026-05-08",
            })
        );
    }

    #[test]
    fn glm45_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &["hello ", "world"]);

        assert_eq!(result.normal_text, "hello world");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn glm45_streaming_preserves_prefix_text() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let result = collect_stream(
            &mut parser,
            &[
                "Prefix ",
                &glm45_tool_call("get_weather", &[("city", "Hangzhou")]),
            ],
        );

        assert_eq!(result.normal_text, "Prefix ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn glm45_streaming_handles_start_token_split_across_chunks() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "hello <tool",
                "_call>get_weather\n",
                "<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
            ],
        );

        assert_eq!(result.normal_text, "hello ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn glm45_streaming_does_not_emit_incomplete_tool_call() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let result = parser.push("<tool_call>get_weather\n<arg_key>city</arg_key>").unwrap();

        assert_eq!(result.normal_text, "");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn glm45_finish_fails_incomplete_tool_call() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        parser.push("<tool_call>get_weather\n<arg_key>city</arg_key>").unwrap();
        let error = parser.finish().unwrap_err();

        assert!(error.as_report().to_string().contains("incomplete GLM MoE tool call"));
    }

    #[test]
    fn glm45_malformed_tool_call_fails_fast() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let error = parser.push("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>").unwrap_err();

        assert!(error.as_report().to_string().contains("tool parser parsing failed"));
    }

    #[test]
    fn glm45_streaming_ignores_trailing_text_after_tool_calls() {
        let mut parser = Glm45MoeToolParser::new(&test_tools());

        let result = collect_stream(
            &mut parser,
            &[&format!(
                "{}<|endoftext|>",
                glm45_tool_call("get_weather", &[("city", "Paris")])
            )],
        );

        assert_eq!(result.normal_text, "");
        assert_eq!(result.calls.len(), 1);
    }
}
