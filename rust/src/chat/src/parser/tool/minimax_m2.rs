use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{parse_buffered_event, safe_text_len};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError, parsing_failed};
use crate::request::ChatTool;

const TOOL_CALL_START: &str = "<minimax:tool_call>";
const TOOL_CALL_END: &str = "</minimax:tool_call>";
const INVOKE_START: &str = "<invoke";
const INVOKE_END: &str = "</invoke>";
const PARAMETER_START: &str = "<parameter";
const PARAMETER_END: &str = "</parameter>";

type MinimaxM2Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MinimaxM2Mode {
    Text,
    ToolBlock,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM2Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    Invoke {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    ToolBlockEnd,
    IgnoredRest,
}

/// Tool parser for MiniMax M2 XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <minimax:tool_call><invoke name="get_weather">
/// <parameter name="city">Seattle</parameter>
/// </invoke></minimax:tool_call>
/// ```
///
/// Arguments are emitted only after a full `<invoke>` block is parsed.
pub struct MinimaxM2ToolParser {
    buffer: String,
    mode: MinimaxM2Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl MinimaxM2ToolParser {
    /// Create a MiniMax M2 tool parser.
    fn new(tools: &[ChatTool]) -> Self {
        Self {
            buffer: String::new(),
            mode: MinimaxM2Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed MiniMax M2 event to parser state and output.
    fn apply_event(&mut self, event: MinimaxM2Event, result: &mut ToolParseResult) -> Result<()> {
        match event {
            MinimaxM2Event::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            MinimaxM2Event::ToolBlockStart => self.mode = MinimaxM2Mode::ToolBlock,
            MinimaxM2Event::Invoke { name, raw_params } => {
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
            MinimaxM2Event::ToolBlockEnd => self.mode = MinimaxM2Mode::Done,
            MinimaxM2Event::IgnoredRest => {}
        }
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = MinimaxM2Mode::Text;
        self.emitted_tool_count = 0;
    }
}

impl ToolParser for MinimaxM2ToolParser {
    /// Create a boxed MiniMax M2 tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the MiniMax M2 parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_minimax_m2_event(input, self.mode)
        })? {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        match self.mode {
            MinimaxM2Mode::Text => {
                result.normal_text.push_str(&self.buffer);
            }
            MinimaxM2Mode::ToolBlock => {
                return Err(parsing_failed!("incomplete MiniMax M2 tool call"));
            }
            MinimaxM2Mode::Done => {}
        }
        self.reset();
        Ok(result)
    }
}

/// Parse a MiniMax M2 event for the current parser mode.
fn parse_next_minimax_m2_event(
    input: &mut MinimaxM2Input<'_>,
    mode: MinimaxM2Mode,
) -> ModalResult<MinimaxM2Event> {
    match mode {
        MinimaxM2Mode::Text => parse_text_event(input),
        MinimaxM2Mode::ToolBlock => parse_tool_block_event(input),
        MinimaxM2Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode MiniMax M2 event.
fn parse_text_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    alt((tool_block_start_event, safe_text_event)).parse_next(input)
}

/// Parse a MiniMax M2 tool-block start marker.
fn tool_block_start_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    literal(TOOL_CALL_START).value(MinimaxM2Event::ToolBlockStart).parse_next(input)
}

/// Parse a safe text run before the next MiniMax M2 marker.
fn safe_text_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    safe_text_len(input, TOOL_CALL_START).map(|len| MinimaxM2Event::Text { len })
}

/// Parse one event inside a MiniMax M2 tool block.
fn parse_tool_block_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    alt((tool_block_end_event, invoke_event)).parse_next(input)
}

/// Parse a MiniMax M2 tool-block end marker.
fn tool_block_end_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    (ws0, literal(TOOL_CALL_END))
        .value(MinimaxM2Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a complete MiniMax M2 invoke block.
fn invoke_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    let (name, raw_params) = seq!(
        _: ws0,
        _: literal(INVOKE_START),
        _: (ws1, literal("name=")),
        attr_value,
        _: literal(">"),
        repeat(0.., terminated(parameter, ws0)),
        _: literal(INVOKE_END),
    )
    .parse_next(input)?;

    Ok(MinimaxM2Event::Invoke {
        name: name.trim().to_string(),
        raw_params,
    })
}

/// Parse a MiniMax M2 parameter block.
fn parameter(input: &mut MinimaxM2Input<'_>) -> ModalResult<(String, String)> {
    let (name, value) = seq!(
        _: literal(PARAMETER_START),
        _: (ws1, literal("name=")),
        attr_value,
        _: literal(">"),
        take_until(0.., PARAMETER_END),
        _: literal(PARAMETER_END),
    )
    .parse_next(input)?;

    Ok((name.trim().to_string(), value.to_string()))
}

/// Parse a quoted or unquoted XML attribute value.
fn attr_value<'i>(input: &mut MinimaxM2Input<'i>) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

/// Parse ignored rest after the MiniMax M2 tool block ends.
fn ignored_rest_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    rest.value(MinimaxM2Event::IgnoredRest).parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{MinimaxM2ToolParser, TOOL_CALL_END, TOOL_CALL_START, ToolParser};
    use crate::parser::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn build_tool_block(invokes: &[(&str, Vec<(&str, &str)>)]) -> String {
        let invokes = invokes
            .iter()
            .map(|(function_name, params)| {
                let params = params
                    .iter()
                    .map(|(name, value)| format!(r#"<parameter name="{name}">{value}</parameter>"#))
                    .collect::<Vec<_>>()
                    .join("");
                format!(r#"<invoke name="{function_name}">{params}</invoke>"#)
            })
            .collect::<String>();
        format!("{TOOL_CALL_START}{invokes}{TOOL_CALL_END}")
    }

    #[test]
    fn minimax_m2_parse_complete_without_tool_call_keeps_text() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn minimax_m2_parse_complete_extracts_single_tool_call() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                vec![("city", "Seattle"), ("days", "5")],
            )]))
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "city": "Seattle", "days": 5 })
        );
    }

    #[test]
    fn minimax_m2_parse_complete_preserves_prefix_and_ignores_trailing_text() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = format!(
            "Let me check. {} This trailing text is ignored.",
            build_tool_block(&[("get_weather", vec![("city", "Seattle")])])
        );
        let result = parser.parse_complete(&output).unwrap();

        assert_eq!(result.normal_text, "Let me check. ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn minimax_m2_parse_complete_extracts_multiple_invokes() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_block(&[
                ("get_weather", vec![("city", "Seattle")]),
                ("get_weather", vec![("city", "NYC")]),
            ]))
            .unwrap();

        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[1].arguments).unwrap(),
            json!({ "city": "NYC" })
        );
    }

    #[test]
    fn minimax_m2_parse_complete_converts_schema_types() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_block(&[(
                "convert",
                vec![
                    ("whole", "5.0"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "42"),
                ],
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn minimax_m2_parse_complete_handles_multiline_parameters() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                "<minimax:tool_call>\
                 <invoke name=\"calculate_area\">\
                 <parameter name=\"shape\">\nrectangle\n</parameter>\
                 <parameter name=\"dimensions\">{\"width\":10,\n\"height\":20}</parameter>\
                 <parameter name=\"precision\">2</parameter>\
                 </invoke>\
                 </minimax:tool_call>",
            )
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "shape": "\nrectangle\n",
                "dimensions": { "width": 10, "height": 20 },
                "precision": 2,
            })
        );
    }

    #[test]
    fn minimax_m2_streaming_extracts_single_tool_call() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<minimax:tool_call>",
                r#"<invoke name="get_weather">"#,
                r#"<parameter name="city">Seattle</parameter>"#,
                "</invoke></minimax:tool_call>",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m2_streaming_preserves_prefix_text() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "Let me check. ",
                "<minimax:tool_call>",
                r#"<invoke name="get_weather"><parameter name="city">Seattle</parameter></invoke>"#,
                "</minimax:tool_call>",
            ],
        );

        assert_eq!(result.normal_text, "Let me check. ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn minimax_m2_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &["Hello, ", "world!"]);

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn minimax_m2_streaming_handles_marker_split_across_chunks() {
        let text = build_tool_block(&[("get_weather", vec![("city", "Seattle")])]);
        let chunks = split_by_chars(&text, 3);
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.calls.len(), 1);
        assert!(result.normal_text.is_empty());
    }

    #[test]
    fn minimax_m2_streaming_extracts_multiple_invokes_in_order() {
        let text = build_tool_block(&[
            ("get_weather", vec![("city", "Seattle")]),
            ("get_weather", vec![("city", "NYC")]),
        ]);
        let chunks = split_by_chars(&text, 7);
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
    }

    #[test]
    fn minimax_m2_streaming_ignores_text_after_tool_block() {
        let text = format!(
            "{} ignored",
            build_tool_block(&[("get_weather", vec![("city", "Seattle")])])
        );
        let chunks = split_by_chars(&text, 5);
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &chunks);

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn minimax_m2_streaming_does_not_emit_incomplete_tool_call() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = parser.push(r#"<minimax:tool_call><invoke name="get_weather">"#).unwrap();

        assert!(result.normal_text.is_empty());
        assert!(result.calls.is_empty());
    }

    #[test]
    fn minimax_m2_finish_fails_incomplete_tool_call() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        parser.push(r#"<minimax:tool_call><invoke name="get_weather">"#).unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m2_finish_fails_after_bare_tool_block_start() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        parser.push("<minimax:tool_call>").unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m2_malformed_tool_call_fails_fast() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let error = parser.push("<minimax:tool_call><bad></minimax:tool_call>").unwrap_err();

        expect!["tool parser parsing failed: "].assert_eq(&error.to_report_string());
    }
}
