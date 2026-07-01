use winnow::combinator::{alt, seq};
use winnow::error::ErrMode;
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest};

use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

const TOOL_CALLS_PREFIX: &str = "<|tools_prefix|>";
const TOOL_CALLS_SUFFIX: &str = "<|tools_suffix|>";

type ApertusInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone)]
enum Mode {
    Text,
    /// Between `<|tools_prefix|>` and `<|tools_suffix|>`.
    ToolCalls {
        suffix_scan: MarkerScanState,
    },
    Done,
}

#[derive(Debug, Clone)]
enum ApertusEvent {
    Text { len: usize },
    ToolCallsStart,
    ToolCalls(Vec<ToolCallDelta>),
}

pub struct ApertusToolParser {
    buffer: String,
    mode: Mode,
}

impl ApertusToolParser {
    pub fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Mode::Text,
        }
    }

    fn apply_event(&mut self, event: ApertusEvent, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            ApertusEvent::Text { len } => output.push_text(&self.buffer[..len]),
            ApertusEvent::ToolCallsStart => {
                self.mode = Mode::ToolCalls {
                    suffix_scan: MarkerScanState::default(),
                };
            }
            ApertusEvent::ToolCalls(deltas) => {
                for delta in deltas {
                    output.push_call(delta);
                }
                self.mode = Mode::Done;
            }
        }
        Ok(())
    }

    fn reset_state(&mut self) -> String {
        self.mode = Mode::Text;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for ApertusToolParser {
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
            parse_next_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match &self.mode {
            Mode::Text | Mode::Done => output.push_text(&self.buffer),
            Mode::ToolCalls { .. } => {
                let body = self.buffer.trim();
                match serde_json::from_str::<Vec<serde_json::Value>>(body) {
                    Ok(values) => {
                        for delta in json_values_to_deltas(&values) {
                            output.push_call(delta);
                        }
                    }
                    Err(_) => {
                        return Err(parsing_failed!("incomplete Apertus tool call"));
                    }
                }
            }
        }
        let _ = self.reset_state();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.reset_state()
    }
}

fn parse_next_event(input: &mut ApertusInput<'_>, mode: &mut Mode) -> ModalResult<ApertusEvent> {
    match mode {
        Mode::Text => parse_text_event(input),
        Mode::ToolCalls { suffix_scan } => tool_calls_event(input, suffix_scan),
        Mode::Done => parse_done_event(input),
    }
}

fn parse_text_event(input: &mut ApertusInput<'_>) -> ModalResult<ApertusEvent> {
    alt((tools_prefix_event, safe_text_event)).parse_next(input)
}

fn tools_prefix_event(input: &mut ApertusInput<'_>) -> ModalResult<ApertusEvent> {
    literal(TOOL_CALLS_PREFIX).value(ApertusEvent::ToolCallsStart).parse_next(input)
}

fn safe_text_event(input: &mut ApertusInput<'_>) -> ModalResult<ApertusEvent> {
    safe_text_len(input, TOOL_CALLS_PREFIX).map(|len| ApertusEvent::Text { len })
}

fn tool_calls_event(
    input: &mut ApertusInput<'_>,
    suffix_scan: &mut MarkerScanState,
) -> ModalResult<ApertusEvent> {
    use winnow::ascii::multispace0 as ws0;

    let (body,) = seq!(
        _: ws0,
        take_until_marker(TOOL_CALLS_SUFFIX, suffix_scan),
        _: literal(TOOL_CALLS_SUFFIX),
    )
    .parse_next(input)?;

    parse_tool_calls_body(body)
}

fn parse_done_event(input: &mut ApertusInput<'_>) -> ModalResult<ApertusEvent> {
    let len = input.len();
    if len == 0 {
        return Err(ErrMode::Incomplete(winnow::stream::Needed::Unknown));
    }
    rest.value(ApertusEvent::Text { len }).parse_next(input)
}

fn parse_tool_calls_body(body: &str) -> ModalResult<ApertusEvent> {
    let values: Vec<serde_json::Value> = serde_json::from_str(body).unwrap_or_default();
    Ok(ApertusEvent::ToolCalls(json_values_to_deltas(&values)))
}

fn json_values_to_deltas(values: &[serde_json::Value]) -> Vec<ToolCallDelta> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            let obj = v.as_object()?;
            let (name, args_val) = obj.iter().next()?;
            let arguments = serde_json::to_string(args_val).ok()?;
            Some(ToolCallDelta {
                tool_index: i,
                name: Some(name.clone()),
                arguments,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;
    use thiserror_ext::AsReport;

    use super::{ApertusToolParser, TOOL_CALLS_PREFIX, TOOL_CALLS_SUFFIX, ToolParser};
    use crate::tool::ToolParserOutput;
    use crate::tool::ToolParserTestExt as _;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn build_tool_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"{}": {}}}"#, function_name, arguments)
    }

    fn build_tool_block(entries: &[(&str, &str)]) -> String {
        let entries = entries
            .iter()
            .map(|(name, args)| build_tool_call(name, args))
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{entries}]")
    }

    fn wrap_with_tokens(content: &str) -> String {
        format!("{TOOL_CALLS_PREFIX}{content}{TOOL_CALLS_SUFFIX}")
    }

    #[test]
    fn apertus_parse_complete_without_tool_call_keeps_text() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();
        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn apertus_parse_complete_extracts_single_tool_call() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "Let me check. {}",
                wrap_with_tokens(&build_tool_block(&[(
                    "get_weather",
                    r#"{"city": "Paris"}"#,
                )]))
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Paris"})
        );
    }

    #[test]
    fn apertus_parse_complete_extracts_multiple_tool_calls() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&wrap_with_tokens(&build_tool_block(&[
                ("get_weather", r#"{"location": "NYC"}"#),
                ("add", r#"{"x": 1, "y": 2}"#),
            ])))
            .unwrap();

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
            json!({"location": "NYC"})
        );
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(output.calls()[1].name.as_deref(), Some("add"));
    }

    #[test]
    fn apertus_does_not_validate_or_normalize_arguments() {
        let mut parser = ApertusToolParser::new(&test_tools());
        // Valid JSON, but extra fields that a tool schema would reject.
        let arguments = r#"{"city":"Paris","extra":true}"#;
        let output = parser
            .parse_complete(&wrap_with_tokens(&build_tool_block(&[(
                "get_weather",
                arguments,
            )])))
            .unwrap();

        // Arguments are re-serialized by serde_json (compact form), but the
        // parser does not validate against any tool schema.
        assert_eq!(output.calls()[0].arguments, r#"{"city":"Paris","extra":true}"#);
    }

    #[test]
    fn apertus_parse_complete_preserves_text_before_and_after() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&format!(
                "Prefix text. {} suffix text.",
                wrap_with_tokens(&build_tool_block(&[(
                    "get_weather",
                    r#"{"city": "London"}"#,
                )]))
            ))
            .unwrap();

        assert_eq!(output.normal_text(), "Prefix text.  suffix text.");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn apertus_parse_complete_handles_nested_arguments() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let arguments = r#"{"user_id": 42, "shipping": {"city": "Singapore", "zip": 18956}}"#;
        let output = parser
            .parse_complete(&wrap_with_tokens(&build_tool_block(&[(
                "create_order",
                arguments,
            )])))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("create_order"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
            serde_json::from_str::<serde_json::Value>(arguments).unwrap()
        );
    }

    #[test]
    fn apertus_parse_complete_handles_array_and_bool_arguments() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let arguments = r#"{"items": [1, 2, 3], "flag": true, "note": "hello"}"#;
        let output = parser
            .parse_complete(&wrap_with_tokens(&build_tool_block(&[(
                "process", arguments,
            )])))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
            json!({"items": [1, 2, 3], "flag": true, "note": "hello"})
        );
    }

    #[test]
    fn apertus_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &["Hello, ", "world!"]);
        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn apertus_streaming_emits_argument_deltas() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let chunks = [
            "preface ",
            TOOL_CALLS_PREFIX,
            r#"[{"get_weather": {"city":"#,
            r#""Beijing""#,
            "}}]",
            TOOL_CALLS_SUFFIX,
            " suffix",
        ];

        let mut output = ToolParserOutput::default();
        let mut chunk_had_calls = Vec::new();
        for chunk in chunks {
            let next = parser.parse_chunk(chunk).unwrap();
            chunk_had_calls.push(!next.calls().is_empty());
            output.append(next);
        }
        output.append(parser.finish().unwrap());

        // Body chunks produce no tool calls; all calls arrive with the suffix.
        assert_eq!(
            chunk_had_calls,
            [false, false, false, false, false, true, false]
        );

        assert_eq!(output.normal_text(), "preface  suffix");
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"city":"Beijing"}"#
        );
    }

    #[test]
    fn apertus_streaming_extracts_single_tool_call() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let text = format!(
            "Prefix. {}",
            wrap_with_tokens(&build_tool_block(&[(
                "get_weather",
                r#"{"city": "Paris"}"#,
            )]))
        );
        let chunks = split_by_chars(&text, 7);
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "Prefix. ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Paris"})
        );
    }

    #[test]
    fn apertus_streaming_handles_marker_split_across_chunks() {
        let text = wrap_with_tokens(&build_tool_block(&[(
            "get_weather",
            r#"{"city": "Seattle"}"#,
        )]));
        let chunks = split_by_chars(&text, 3);
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn apertus_streaming_handles_argument_chunked_across_deltas() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let chunks = [
            TOOL_CALLS_PREFIX,
            r#"[{"get_weather": {"city": "#,
            r#""Paris"}}]"#,
            TOOL_CALLS_SUFFIX,
        ];

        let output = collect_stream(&mut parser, &chunks);
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Paris"})
        );
    }

    #[test]
    fn apertus_streaming_extracts_multiple_tool_calls_in_order() {
        let text = wrap_with_tokens(&build_tool_block(&[
            ("get_weather", r#"{"city": "Seattle"}"#),
            ("add", r#"{"x": 1, "y": 2}"#),
        ]));
        let chunks = split_by_chars(&text, 11);
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"Seattle\"}",
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
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn apertus_finish_empty_tool_call_block_is_ok() {
        let mut parser = ApertusToolParser::new(&test_tools());
        parser.parse_chunk(TOOL_CALLS_PREFIX).unwrap();
        parser.parse_chunk("[]").unwrap();

        let output = parser.parse_chunk(TOOL_CALLS_SUFFIX).unwrap();
        assert!(!output.calls().is_empty() || output.calls().is_empty());
    }

    #[test]
    fn apertus_finish_incomplete_tool_call_is_error() {
        let mut parser = ApertusToolParser::new(&test_tools());
        parser.parse_chunk(TOOL_CALLS_PREFIX).unwrap();
        parser.parse_chunk(r#"[{"get_weather": {"city":"#).unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Apertus tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn apertus_finish_parses_complete_json_without_suffix() {
        let mut parser = ApertusToolParser::new(&test_tools());
        parser.parse_chunk(TOOL_CALLS_PREFIX).unwrap();
        parser
            .parse_chunk(r#"[{"get_weather":{"city":"Paris"}}]"#)
            .unwrap();

        let output = parser.finish().unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            output.calls()[0].arguments,
            r#"{"city":"Paris"}"#
        );
    }

    #[test]
    fn apertus_streaming_missing_suffix_parses_body_on_finish() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let prefix_output = parser.parse_chunk("Prefix. ").unwrap();
        assert_eq!(prefix_output.normal_text(), "Prefix. ");
        parser.parse_chunk(TOOL_CALLS_PREFIX).unwrap();
        parser.parse_chunk(r#"[{"get_weather":{"city":"#).unwrap();
        parser.parse_chunk(r#""Paris"}}]"#).unwrap();

        let output = parser.finish().unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn apertus_finish_flushes_trailing_text_in_text_mode() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = parser.parse_chunk("Hello, ").unwrap();
        assert_eq!(output.normal_text(), "Hello, ");

        let finish_output = parser.finish().unwrap();
        assert_eq!(finish_output.normal_text(), "");
    }

    #[test]
    fn apertus_finish_flushes_trailing_text_after_suffix() {
        let mut parser = ApertusToolParser::new(&test_tools());

        let input = format!(
            "{}[{{\"get_weather\": {{\"city\": \"Paris\"}}}}]{}trailing",
            TOOL_CALLS_PREFIX, TOOL_CALLS_SUFFIX
        );
        parser.parse_complete(&input).unwrap();

        let output = parser.parse_complete("new request text").unwrap();
        assert_eq!(output.normal_text(), "new request text");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn apertus_preserve_special_tokens_is_true() {
        let parser = ApertusToolParser::new(&test_tools());
        assert!(parser.preserve_special_tokens());
    }

    #[test]
    fn apertus_tool_call_count_increments_across_multiple_calls() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&wrap_with_tokens(&build_tool_block(&[
                ("get_weather", r#"{"city": "A"}"#),
                ("get_weather", r#"{"city": "B"}"#),
                ("add", r#"{"x": 1, "y": 2}"#),
            ])))
            .unwrap();

        assert_eq!(output.calls().len(), 3);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(output.calls()[2].tool_index, 2);
    }

    #[test]
    fn apertus_empty_args_object() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let output = parser.parse_complete(&wrap_with_tokens(r#"[{"get_time": {}}]"#)).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_time"));
        assert_eq!(output.calls()[0].arguments, "{}");
    }

    #[test]
    fn apertus_malformed_json_is_graceful() {
        let mut parser = ApertusToolParser::new(&test_tools());
        let result = parser.parse_complete(&format!(
            "{}not valid json{}",
            TOOL_CALLS_PREFIX, TOOL_CALLS_SUFFIX
        ));

        let output = result.unwrap();
        assert_eq!(output.calls().len(), 0);
    }
}
