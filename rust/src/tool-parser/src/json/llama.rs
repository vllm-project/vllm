use winnow::ascii::multispace0 as ws0;
use winnow::combinator::seq;
use winnow::error::{ModalResult, StrContext};
use winnow::prelude::*;
use winnow::token::literal;

use super::{
    JsonToolCallConfig, JsonToolCallEvent, JsonToolCallWhitespace, JsonToolInput,
    argument_delta_event, tool_call_header_event,
};
use crate::utils::{JsonObjectScanState, parse_buffered_event};
use crate::{Result, Tool, ToolCallDelta, ToolParseResult, ToolParser};

#[derive(Debug, Clone, PartialEq, Eq)]
enum LlamaJsonMode {
    Start,
    Header,
    Arguments { json_scan: JsonObjectScanState },
    AfterCall,
    Passthrough,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LlamaJsonEvent {
    ToolCallHeader { function_name: String },
    Arguments { len: usize },
    ToolCallClose,
    Separator,
}

/// Tool parser for strict Llama JSON-template tool calls.
///
/// Example tool call content:
///
/// ```text
/// {"name":"get_weather","parameters":{"location":"Tokyo"}}; {"name":"add","parameters":{"x":1,"y":2}}
/// ```
///
/// Arguments are already OpenAI-style JSON text, so they are streamed as raw
/// argument deltas without schema conversion or JSON normalization.
///
/// Natural text at the beginning of the stream permanently disables tool
/// parsing for that assistant output.
pub struct Llama3JsonToolParser {
    buffer: String,
    mode: LlamaJsonMode,
    active_tool_index: Option<usize>,
    emitted_tool_count: usize,
}

impl Llama3JsonToolParser {
    /// Create a Llama JSON tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: LlamaJsonMode::Start,
            active_tool_index: None,
            emitted_tool_count: 0,
        }
    }

    /// Commit the stream to JSON parsing or permanent passthrough.
    fn commit_start(&mut self) -> bool {
        if !matches!(self.mode, LlamaJsonMode::Start) {
            return true;
        }

        if self.buffer.is_empty() {
            return false;
        }

        if self.buffer.starts_with('{') {
            self.mode = LlamaJsonMode::Header;
        } else {
            self.mode = LlamaJsonMode::Passthrough;
        }
        true
    }

    /// Apply one parsed Llama JSON event to parser state and output.
    fn apply_event(&mut self, event: LlamaJsonEvent, result: &mut ToolParseResult) -> Result<()> {
        match event {
            LlamaJsonEvent::ToolCallHeader { function_name } => {
                let tool_index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.active_tool_index = Some(tool_index);
                self.mode = LlamaJsonMode::Arguments {
                    json_scan: JsonObjectScanState::default(),
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: Some(function_name),
                    arguments: String::new(),
                });
            }
            LlamaJsonEvent::Arguments { len: consumed_len } => {
                let Some(tool_index) = self.active_tool_index else {
                    return Err(parsing_failed!(
                        "Llama JSON arguments without an active tool call"
                    ));
                };
                result.calls.push(ToolCallDelta {
                    tool_index,
                    name: None,
                    arguments: self.buffer[..consumed_len].to_string(),
                });
            }
            LlamaJsonEvent::ToolCallClose => {
                self.active_tool_index = None;
                self.mode = LlamaJsonMode::AfterCall;
            }
            LlamaJsonEvent::Separator => {
                self.active_tool_index = None;
                self.mode = LlamaJsonMode::Header;
            }
        }
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = LlamaJsonMode::Start;
        self.active_tool_index = None;
        self.emitted_tool_count = 0;
    }
}

impl ToolParser for Llama3JsonToolParser {
    /// Create a boxed Llama JSON tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the Llama JSON parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        if !self.commit_start() {
            return Ok(result);
        }

        if matches!(self.mode, LlamaJsonMode::Passthrough) {
            result.normal_text.push_str(&self.buffer);
            self.buffer.clear();
            return Ok(result);
        }

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_llama_json_event(input, &mut self.mode)
        })? {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        match &self.mode {
            LlamaJsonMode::Start | LlamaJsonMode::Passthrough => {
                result.normal_text.push_str(&self.buffer);
            }
            LlamaJsonMode::AfterCall if self.buffer.trim().is_empty() => {}
            LlamaJsonMode::Header | LlamaJsonMode::Arguments { .. } => {
                return Err(parsing_failed!("incomplete Llama JSON tool call"));
            }
            LlamaJsonMode::AfterCall => {
                return Err(parsing_failed!("invalid Llama JSON"));
            }
        }
        self.reset();
        Ok(result)
    }
}

/// Parse a Llama JSON event for the current parser mode.
fn parse_next_llama_json_event(
    input: &mut JsonToolInput<'_>,
    mode: &mut LlamaJsonMode,
) -> ModalResult<LlamaJsonEvent> {
    match mode {
        LlamaJsonMode::Start | LlamaJsonMode::Passthrough => {
            unreachable!("Llama JSON parser driver must commit before parsing events")
        }
        LlamaJsonMode::Header => llama_tool_call_header_event(input),
        LlamaJsonMode::Arguments { json_scan } => parse_llama_arguments_event(input, json_scan),
        LlamaJsonMode::AfterCall => after_call_event(input),
    }
}

/// Parse a Llama JSON tool-call header.
fn llama_tool_call_header_event(input: &mut JsonToolInput<'_>) -> ModalResult<LlamaJsonEvent> {
    const CONFIG: JsonToolCallConfig = JsonToolCallConfig {
        parser_name: "Llama JSON",
        start_marker: "",
        end_marker: "",
        marker_whitespace: JsonToolCallWhitespace::Optional,
        delimiter: Some(";"),
        name_key: "name",
        arguments_key: "parameters",
    };

    match tool_call_header_event(input, CONFIG)? {
        JsonToolCallEvent::ToolCallHeader { function_name } => {
            Ok(LlamaJsonEvent::ToolCallHeader { function_name })
        }
        _ => unreachable!("tool_call_header_event only emits ToolCallHeader"),
    }
}

/// Parse one event inside a Llama JSON arguments payload.
fn parse_llama_arguments_event(
    input: &mut JsonToolInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<LlamaJsonEvent> {
    if json_scan.complete() {
        tool_call_close_event(input)
    } else {
        match argument_delta_event(input, json_scan)? {
            JsonToolCallEvent::Arguments { len } => Ok(LlamaJsonEvent::Arguments { len }),
            _ => unreachable!("argument_delta_event only emits Arguments"),
        }
    }
}

/// Parse the outer closing brace for one Llama JSON tool call.
fn tool_call_close_event(input: &mut JsonToolInput<'_>) -> ModalResult<LlamaJsonEvent> {
    literal("}").value(LlamaJsonEvent::ToolCallClose).parse_next(input)
}

/// Parse a semicolon separator after one Llama JSON tool call.
fn after_call_event(input: &mut JsonToolInput<'_>) -> ModalResult<LlamaJsonEvent> {
    seq!(
        _: ws0,
        _: literal(";"),
        _: ws0,
    )
    .value(LlamaJsonEvent::Separator)
    .context(StrContext::Label("Llama JSON"))
    .parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::Llama3JsonToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::{ToolParseResult, ToolParser};

    fn build_tool_call(function_name: &str, parameters: &str) -> String {
        format!(r#"{{"name":"{function_name}","parameters":{parameters}}}"#)
    }

    #[test]
    fn llama_json_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn llama_json_passthrough_never_reenters_tool_parsing() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let mut result = parser.push("plain text first ").unwrap();
        result.append(
            parser.push(&build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)).unwrap(),
        );
        result.append(parser.finish().unwrap());

        assert_eq!(
            result.normal_text,
            r#"plain text first {"name":"get_weather","parameters":{"location":"Tokyo"}}"#
        );
        assert!(result.calls.is_empty());
    }

    #[test]
    fn llama_json_does_not_support_python_tag_prefix() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let input = format!(
            "<|python_tag|>{}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let result = parser.parse_complete(&input).unwrap();

        assert_eq!(result.normal_text, input);
        assert!(result.calls.is_empty());
    }

    #[test]
    fn llama_json_rejects_leading_whitespace_before_tool_call() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let input = format!(
            "\n  {}",
            build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
        );
        let result = parser.parse_complete(&input).unwrap();

        assert_eq!(result.normal_text, input);
        assert!(result.calls.is_empty());
    }

    #[test]
    fn llama_json_extracts_raw_parameters_object() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let arguments = r#"{ "location": "Tokyo", "days": 3 }"#;
        let result = parser.parse_complete(&build_tool_call("get_weather", arguments)).unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn llama_json_rejects_arguments_key() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let error = parser
            .parse_complete(r#"{"name":"get_weather","arguments":{"location":"Tokyo"}}"#)
            .unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid Llama JSON
            expected `parameters`"#]]
        .assert_eq(&error.to_report_string());
    }

    #[test]
    fn llama_json_extracts_multiple_semicolon_separated_calls() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let input = format!(
            "{} \n; {}",
            build_tool_call("get_weather", r#"{"location":"Shanghai"}"#),
            build_tool_call("add", r#"{"x":1,"y":2}"#),
        );
        let result = parser.parse_complete(&input).unwrap();

        expect![[r#"
            ToolParseResult {
                normal_text: "",
                calls: [
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some(
                            "get_weather",
                        ),
                        arguments: "{\"location\":\"Shanghai\"}",
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some(
                            "add",
                        ),
                        arguments: "{\"x\":1,\"y\":2}",
                    },
                ],
            }
        "#]]
        .assert_debug_eq(&result);
    }

    #[test]
    fn llama_json_streaming_emits_argument_deltas() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let chunks = [
            "{\"name\":\"get_weather\",\"parameters\":",
            "{\"location\":",
            "\"Beijing\"",
            "}}",
        ];

        let mut result = ToolParseResult::default();
        let mut observed_arguments = Vec::new();
        for chunk in chunks {
            let next = parser.push(chunk).unwrap();
            observed_arguments.extend(
                next.calls
                    .iter()
                    .filter(|call| call.name.is_none())
                    .map(|call| call.arguments.clone()),
            );
            result.append(next);
        }
        result.append(parser.finish().unwrap());

        assert_eq!(observed_arguments, ["{\"location\":", "\"Beijing\"", "}"]);
        assert_eq!(
            result.coalesce_calls().calls[0].arguments,
            r#"{"location":"Beijing"}"#
        );
    }

    #[test]
    fn llama_json_streaming_handles_split_objects_and_separator() {
        let input = format!(
            "{};{}",
            build_tool_call("get_weather", r#"{"location":"Dallas","state":"TX"}"#),
            build_tool_call("add", r#"{"x":4,"y":5}"#),
        );
        let chunks = split_by_chars(&input, 6);
        let mut parser = Llama3JsonToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text, "");
        assert_eq!(result.calls.len(), 2);
        assert_eq!(
            result.calls[0].arguments,
            r#"{"location":"Dallas","state":"TX"}"#
        );
        assert_eq!(result.calls[1].name.as_deref(), Some("add"));
        assert_eq!(result.calls[1].arguments, r#"{"x":4,"y":5}"#);
    }

    #[test]
    fn llama_json_handles_nested_multiline_and_escaped_string_parameters() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let arguments = r#"{
  "payload": {"items": [1, {"value": "literal { brace } and \"quote\""}]},
  "flag": true
}"#;
        let result = parser.parse_complete(&build_tool_call("convert", arguments)).unwrap();

        assert_eq!(result.calls[0].arguments, arguments);
    }

    #[test]
    fn llama_json_keeps_trailing_whitespace_after_tool_call() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&format!(
                "{}\n\t ",
                build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
            ))
            .unwrap();

        assert_eq!(result.normal_text, "");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn llama_json_finish_fails_incomplete_tool_call() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        parser.push(r#"{"name":"get_weather","parameters":{"location""#).unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete Llama JSON tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn llama_json_malformed_field_order_fails_fast() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let error = parser.push(r#"{"parameters":{},"name":"get_weather"}"#).unwrap_err();

        expect![[r#"
            tool parser parsing failed: invalid Llama JSON
            expected `name`"#]]
        .assert_eq(&error.to_report_string());
    }

    #[test]
    fn llama_json_trailing_non_separator_content_errors() {
        let mut parser = Llama3JsonToolParser::new(&test_tools());
        let error = parser
            .push(&format!(
                "{} trailing",
                build_tool_call("get_weather", r#"{"location":"Tokyo"}"#)
            ))
            .unwrap_err();

        expect!["tool parser parsing failed: invalid Llama JSON"]
            .assert_eq(&error.to_report_string());
    }
}
