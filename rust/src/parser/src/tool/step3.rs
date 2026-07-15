use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, opt, repeat, seq, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{
    MarkerScanState, incomplete, parse_buffered_event, safe_text_len, take_until_marker,
};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::Tool;

const TOOL_CALLS_START: &str = "<｜tool_calls_begin｜>";
const TOOL_CALLS_END: &str = "<｜tool_calls_end｜>";
const TOOL_CALL_START: &str = "<｜tool_call_begin｜>";
const TOOL_CALL_END: &str = "<｜tool_call_end｜>";
const TOOL_SEP: &str = "<｜tool_sep｜>";
const INVOKE_START: &str = "<steptml:invoke";
const INVOKE_END: &str = "</steptml:invoke>";
const PARAMETER_START: &str = "<steptml:parameter";
const PARAMETER_END: &str = "</steptml:parameter>";

const FUNCTION_PREFIX: &str = "function";

type Step3Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Step3Mode {
    Text,
    ToolBlock { tool_call_end_scan: MarkerScanState },
    AfterToolBlock,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Step3Event {
    Text {
        len: usize,
    },
    ToolCallsStart,
    ToolCall {
        name: String,
        raw_params: Vec<(String, String)>,
    },
    Separator,
    ToolCallsEnd,
}

/// Tool parser for Step3 StepTML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>
/// <steptml:invoke name="get_weather">
/// <steptml:parameter name="city">Tokyo</steptml:parameter>
/// </steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>
/// ```
///
/// Arguments are emitted only after a full `<｜tool_call_begin｜>...` block is parsed.
pub struct Step3ToolParser {
    buffer: String,
    mode: Step3Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl Step3ToolParser {
    /// Create a Step3 tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Step3Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed Step3 event to parser state and output.
    fn apply_event(&mut self, event: Step3Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            Step3Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            Step3Event::ToolCallsStart => {
                self.mode = Step3Mode::ToolBlock {
                    tool_call_end_scan: MarkerScanState::default(),
                };
            }
            Step3Event::ToolCall { name, raw_params } => {
                let arguments = self.tool_parameters.convert_params_with_schema(&name, raw_params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
            Step3Event::Separator => {}
            Step3Event::ToolCallsEnd => self.mode = Step3Mode::AfterToolBlock,
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = Step3Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for Step3ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_step3_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            Step3Mode::Text | Step3Mode::AfterToolBlock => {
                output.push_text(&self.buffer);
            }
            Step3Mode::ToolBlock { .. } => {
                return Err(parsing_failed!("incomplete Step3 tool call"));
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Step3ToolParser::reset(self)
    }
}

/// Parse a Step3 event for the current parser mode.
fn parse_next_step3_event(
    input: &mut Step3Input<'_>,
    mode: &mut Step3Mode,
) -> ModalResult<Step3Event> {
    match mode {
        Step3Mode::Text => parse_text_event(input),
        Step3Mode::ToolBlock { tool_call_end_scan } => {
            parse_tool_block_event(input, tool_call_end_scan)
        }
        Step3Mode::AfterToolBlock => parse_after_tool_block_event(input),
    }
}

/// Parse a text-mode Step3 event.
fn parse_text_event(input: &mut Step3Input<'_>) -> ModalResult<Step3Event> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse one event inside a Step3 tool-calls block.
fn parse_tool_block_event(
    input: &mut Step3Input<'_>,
    tool_call_end_scan: &mut MarkerScanState,
) -> ModalResult<Step3Event> {
    alt((
        tool_calls_end_event,
        tool_separator_event,
        |input: &mut Step3Input<'_>| tool_call_event(input, tool_call_end_scan),
    ))
    .parse_next(input)
}

/// Parse trailing text after a Step3 tool-calls block.
fn parse_after_tool_block_event(input: &mut Step3Input<'_>) -> ModalResult<Step3Event> {
    let text = rest.parse_next(input)?;
    if text.is_empty() {
        return incomplete();
    }

    Ok(Step3Event::Text { len: text.len() })
}

/// Parse a Step3 tool-calls start marker.
fn tool_calls_start_event(input: &mut Step3Input<'_>) -> ModalResult<Step3Event> {
    literal(TOOL_CALLS_START).value(Step3Event::ToolCallsStart).parse_next(input)
}

/// Parse a Step3 tool-calls end marker.
fn tool_calls_end_event(input: &mut Step3Input<'_>) -> ModalResult<Step3Event> {
    (ws0, literal(TOOL_CALLS_END)).value(Step3Event::ToolCallsEnd).parse_next(input)
}

/// Parse a separator between Step3 tool calls.
fn tool_separator_event(input: &mut Step3Input<'_>) -> ModalResult<Step3Event> {
    (ws0, literal(TOOL_SEP)).value(Step3Event::Separator).parse_next(input)
}

/// Parse a safe text run before the next Step3 marker.
fn safe_text_event(input: &mut Step3Input<'_>) -> ModalResult<Step3Event> {
    safe_text_len(input, TOOL_CALLS_START).map(|len| Step3Event::Text { len })
}

/// Parse a complete Step3 tool call.
fn tool_call_event(
    input: &mut Step3Input<'_>,
    tool_call_end_scan: &mut MarkerScanState,
) -> ModalResult<Step3Event> {
    let (body,) = seq!(
        _: ws0,
        _: literal(TOOL_CALL_START),
        take_until_marker(TOOL_CALL_END, tool_call_end_scan),
        _: literal(TOOL_CALL_END),
    )
    .parse_next(input)?;

    parse_tool_call_body(body)
}

/// Parse a Step3 tool-call body.
fn parse_tool_call_body(body: &str) -> ModalResult<Step3Event> {
    let mut input = body;
    let (_, _, event, _, _) =
        (ws0, opt(function_prefix), invoke_event, ws0, eof).parse_next(&mut input)?;
    Ok(event)
}

/// Parse an optional Step3 function prefix before StepTML.
fn function_prefix(input: &mut &str) -> ModalResult<()> {
    (ws0, literal(FUNCTION_PREFIX), ws0, literal(TOOL_SEP), ws0)
        .void()
        .parse_next(input)
}

/// Parse a complete StepTML invoke block.
fn invoke_event(input: &mut &str) -> ModalResult<Step3Event> {
    let (name, body) = seq!(
        _: literal(INVOKE_START),
        _: (ws1, literal("name=")),
        attr_value,
        _: literal(">"),
        take_until(0.., INVOKE_END),
        _: literal(INVOKE_END),
    )
    .parse_next(input)?;
    let raw_params = parse_invoke_params(body)?;

    Ok(Step3Event::ToolCall {
        name: name.trim().to_string(),
        raw_params,
    })
}

/// Parse all StepTML parameter blocks inside an invoke body.
fn parse_invoke_params(invoke_body: &str) -> ModalResult<Vec<(String, String)>> {
    let mut input = invoke_body;
    delimited(ws0, repeat(0.., terminated(parameter, ws0)), eof).parse_next(&mut input)
}

/// Parse a StepTML parameter block.
fn parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (name, value) = seq!(
        _: literal(PARAMETER_START),
        _: (ws1, literal("name=")),
        attr_value,
        _: literal(">"),
        take_until(0.., PARAMETER_END),
        _: literal(PARAMETER_END),
    )
    .parse_next(input)?;

    Ok((name.trim().to_string(), value.trim().to_string()))
}

/// Parse a quoted or unquoted XML attribute value.
fn attr_value<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use serde_json::{Value, json};

    use super::{Step3ToolParser, TOOL_CALLS_END, TOOL_CALLS_START, ToolParser};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_calls(calls: &[(&str, Vec<(&str, &str)>)]) -> String {
        build_tool_calls_with_prefix(calls, false)
    }

    fn build_tool_calls_with_prefix(calls: &[(&str, Vec<(&str, &str)>)], prefix: bool) -> String {
        let calls = calls
            .iter()
            .map(|(function_name, params)| build_tool_call(function_name, params, prefix))
            .collect::<Vec<_>>()
            .join(super::TOOL_SEP);
        format!("{TOOL_CALLS_START}{calls}{TOOL_CALLS_END}")
    }

    fn build_tool_call(function_name: &str, params: &[(&str, &str)], prefix: bool) -> String {
        let params = params
            .iter()
            .map(|(name, value)| {
                format!(r#"<steptml:parameter name="{name}">{value}</steptml:parameter>"#)
            })
            .collect::<String>();
        let prefix = if prefix {
            format!("{}{}", super::FUNCTION_PREFIX, super::TOOL_SEP)
        } else {
            String::new()
        };
        format!(
            r#"<｜tool_call_begin｜>{prefix}<steptml:invoke name="{function_name}">{params}</steptml:invoke><｜tool_call_end｜>"#
        )
    }

    fn assert_calls(output: &ToolParserOutput, expected: &[(&str, Value)]) {
        let calls = output.calls();
        assert_eq!(calls.len(), expected.len());
        for (index, (call, (name, arguments))) in calls.iter().zip(expected).enumerate() {
            assert_eq!(call.tool_index, index);
            assert_eq!(call.name.as_deref(), Some(*name));
            let parsed: Value = serde_json::from_str(&call.arguments).unwrap();
            assert_eq!(&parsed, arguments);
        }
    }

    #[test]
    fn step3_preserves_special_tokens() {
        let parser = Step3ToolParser::new(&test_tools());

        assert!(parser.preserve_special_tokens());
    }

    #[test]
    fn step3_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert_calls(&output, &[]);
    }

    #[test]
    fn step3_parse_complete_extracts_single_tool_call() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_calls(&[(
                "get_weather",
                vec![("city", "Tokyo"), ("days", "2")],
            )]))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_calls(
            &output,
            &[("get_weather", json!({ "city": "Tokyo", "days": 2 }))],
        );
    }

    #[test]
    fn step3_parse_complete_accepts_function_prefix() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_calls_with_prefix(
                &[("get_weather", vec![("city", "Tokyo")])],
                true,
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_calls(&output, &[("get_weather", json!({ "city": "Tokyo" }))]);
    }

    #[test]
    fn step3_parse_complete_extracts_parallel_tool_calls() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_calls(&[
                ("get_weather", vec![("city", "Tokyo")]),
                ("add", vec![("x", "2"), ("y", "3")]),
            ]))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_calls(
            &output,
            &[
                ("get_weather", json!({ "city": "Tokyo" })),
                ("add", json!({ "x": 2, "y": 3 })),
            ],
        );
    }

    #[test]
    fn step3_streaming_handles_marker_and_tag_splits() {
        let text = build_tool_calls(&[
            ("get_weather", vec![("city", "Tokyo")]),
            ("add", vec![("x", "2"), ("y", "3")]),
        ]);
        let chunks = split_by_chars(&text, 5);
        let mut parser = Step3ToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_calls(
            &output,
            &[
                ("get_weather", json!({ "city": "Tokyo" })),
                ("add", json!({ "x": 2, "y": 3 })),
            ],
        );
    }

    #[test]
    fn step3_parse_complete_preserves_surrounding_text() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let output = format!(
            "Let me check. {} Done.",
            build_tool_calls(&[("get_weather", vec![("city", "Tokyo")])])
        );
        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text(), "Let me check.  Done.");
        assert_calls(&output, &[("get_weather", json!({ "city": "Tokyo" }))]);
    }

    #[test]
    fn step3_parse_complete_handles_no_parameter_call() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let output = parser.parse_complete(&build_tool_calls(&[("refresh", vec![])])).unwrap();

        assert!(output.normal_text().is_empty());
        assert_calls(&output, &[("refresh", json!({}))]);
    }

    #[test]
    fn step3_finish_fails_incomplete_tool_call_and_reset_recovers_buffer() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        parser
            .parse_into(
                r#"<｜tool_calls_begin｜><｜tool_call_begin｜><steptml:invoke name="get_weather">"#,
                &mut output,
            )
            .unwrap();

        let error = parser.finish().unwrap_err();
        let buffered = parser.reset();

        expect!["tool parser parsing failed: incomplete Step3 tool call"]
            .assert_eq(&error.to_report_string());
        assert!(output.events.is_empty());
        expect![[r#"<｜tool_call_begin｜><steptml:invoke name="get_weather">"#]]
            .assert_eq(&buffered);
    }

    #[test]
    fn step3_malformed_tool_call_fails_fast() {
        let mut parser = Step3ToolParser::new(&test_tools());
        let error = parser
            .parse_chunk("<｜tool_calls_begin｜><｜tool_call_begin｜><bad><｜tool_call_end｜>")
            .unwrap_err();

        expect![
            "tool parser parsing failed: near \"<｜tool_call_begin｜><bad><｜tool_call_end｜>\": "
        ]
        .assert_eq(&error.to_report_string());
    }
}
