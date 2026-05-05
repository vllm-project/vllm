use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, repeat, terminated};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};

use super::parameters::ToolSchemas;
use super::utils::{parse_buffered_event, safe_text_len};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError, parsing_failed};
use crate::request::ChatTool;

const INVOKE_START: &str = "<｜DSML｜invoke";
const INVOKE_END: &str = "</｜DSML｜invoke>";
const PARAMETER_START: &str = "<｜DSML｜parameter";
const PARAMETER_END: &str = "</｜DSML｜parameter>";

type DsmlInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, Copy)]
pub(super) struct DsmlTokens {
    pub tool_calls_start: &'static str,
    pub tool_calls_end: &'static str,
}

impl DsmlTokens {
    const V32: Self = Self {
        tool_calls_start: "<｜DSML｜function_calls>",
        tool_calls_end: "</｜DSML｜function_calls>",
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DsmlMode {
    Text,
    ToolBlock,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DsmlEvent {
    Text {
        len: usize,
    },
    ToolCallsStart,
    Invoke {
        name: String,
        raw_params: Vec<DsmlParameter>,
    },
    ToolCallsEnd,
    IgnoredRest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DsmlParameter {
    name: String,
    value: String,
    is_string: bool,
}

/// Tool parser for DeepSeek V3.2 models.
///
/// Example tool call content:
///
/// ```text
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// ```
///
/// Arguments are emitted only after a full `invoke` block is parsed.
///
/// DeepSeek V3.2 relies on DSML markers such as `｜DSML｜`, which are
/// represented as special tokens in the tokenizer and therefore must be
/// preserved during decode for parsing to work.
pub struct DeepSeekV32ToolParser {
    buffer: String,
    mode: DsmlMode,
    emitted_invoke_count: usize,
    tool_parameters: ToolSchemas,
    tokens: DsmlTokens,
}

impl DeepSeekV32ToolParser {
    /// Create a parser with DeepSeek V3.2 DSML tokens.
    fn new(tools: &[ChatTool]) -> Self {
        Self::with_tokens(tools, DsmlTokens::V32)
    }

    /// Create a parser with custom DSML tokens, for reuse by DeepSeek V4 which
    /// has different markers but mostly shared logic.
    pub(super) fn with_tokens(tools: &[ChatTool], tokens: DsmlTokens) -> Self {
        Self {
            buffer: String::new(),
            mode: DsmlMode::Text,
            emitted_invoke_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
            tokens,
        }
    }

    /// Apply one parsed DSML event to parser state and output.
    fn apply_event(&mut self, event: DsmlEvent, result: &mut ToolParseResult) -> Result<()> {
        match event {
            DsmlEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            DsmlEvent::ToolCallsStart => self.mode = DsmlMode::ToolBlock,
            DsmlEvent::Invoke { name, raw_params } => {
                let mut arguments = serde_json::Map::with_capacity(raw_params.len());
                for param in raw_params {
                    let value = if param.is_string {
                        serde_json::Value::String(param.value)
                    } else {
                        self.tool_parameters.convert_param_with_schema(
                            &name,
                            &param.name,
                            &param.value,
                        )
                    };
                    arguments.insert(param.name, value);
                }
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                result.calls.push(ToolCallDelta {
                    tool_index: self.emitted_invoke_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_invoke_count += 1;
            }
            DsmlEvent::ToolCallsEnd => self.mode = DsmlMode::Done,
            DsmlEvent::IgnoredRest => {}
        };
        Ok(())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = DsmlMode::Text;
        self.emitted_invoke_count = 0;
    }
}

impl ToolParser for DeepSeekV32ToolParser {
    /// Create a boxed DeepSeek V3.2 tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve DSML special tokens when tool parsing is enabled.
    fn adjust_request(&self, request: &mut crate::request::ChatRequest) -> Result<()> {
        if request.tool_parsing_enabled() {
            // Preserve DSML sentinels like `｜DSML｜function_calls` during decode.
            request.decode_options.skip_special_tokens = false;
        }
        Ok(())
    }

    /// Push one decoded text chunk through the DSML parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        // Extract tool calls from streaming model output.
        //
        // Uses a buffer-until-complete-invoke strategy: text is buffered until
        // a complete invoke block is available, then parsed and emitted in one
        // shot.
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_dsml_event(input, self.mode, self.tokens)
        })? {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();
        if self.mode == DsmlMode::Text && !self.buffer.is_empty() {
            result.normal_text.push_str(&self.buffer);
        }
        self.reset();
        Ok(result)
    }
}

/// Parse a DSML event for the current parser mode.
fn parse_next_dsml_event(
    input: &mut DsmlInput<'_>,
    mode: DsmlMode,
    tokens: DsmlTokens,
) -> ModalResult<DsmlEvent> {
    match mode {
        DsmlMode::Text => parse_text_event(input, tokens),
        DsmlMode::ToolBlock => parse_tool_block_event(input, tokens),
        DsmlMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode DSML event.
fn parse_text_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    alt((
        |input: &mut DsmlInput<'_>| tool_calls_start_event(input, tokens),
        |input: &mut DsmlInput<'_>| safe_text_event(input, tokens),
    ))
    .parse_next(input)
}

/// Parse a tool-block DSML event.
fn parse_tool_block_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    ws0.void().parse_next(input)?;
    alt((invoke_event, |input: &mut DsmlInput<'_>| {
        tool_calls_end_event(input, tokens)
    }))
    .parse_next(input)
}

/// Parse a DSML function-calls start marker.
fn tool_calls_start_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    literal(tokens.tool_calls_start)
        .value(DsmlEvent::ToolCallsStart)
        .parse_next(input)
}

/// Parse a DSML function-calls end marker.
fn tool_calls_end_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    literal(tokens.tool_calls_end).value(DsmlEvent::ToolCallsEnd).parse_next(input)
}

/// Parse a trailing rest after DSML function calls.
fn ignored_rest_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    rest.value(DsmlEvent::IgnoredRest).parse_next(input)
}

/// Parse a safe text run before the next DSML marker.
fn safe_text_event(input: &mut DsmlInput<'_>, tokens: DsmlTokens) -> ModalResult<DsmlEvent> {
    safe_text_len(input, tokens.tool_calls_start).map(|len| DsmlEvent::Text { len })
}

/// Parse a DSML invoke block.
fn invoke_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let (_, _, name, _, _, body, _) = (
        literal(INVOKE_START),
        ws1,
        dsml_name_attr,
        ws0,
        ">",
        take_until(0.., INVOKE_END),
        literal(INVOKE_END),
    )
        .parse_next(input)?;
    let raw_params = parse_invoke_params(body)?;
    Ok(DsmlEvent::Invoke {
        name: name.to_string(),
        raw_params,
    })
}

/// Parse a DSML invoke body.
fn parse_invoke_params(invoke_body: &str) -> ModalResult<Vec<DsmlParameter>> {
    let mut input = invoke_body;
    delimited(ws0, repeat(0.., terminated(parse_parameter, ws0)), eof).parse_next(&mut input)
}

/// Parse a DSML parameter block.
fn parse_parameter(input: &mut &str) -> ModalResult<DsmlParameter> {
    let (_, _, name, _, is_string, _, _, value, _) = (
        literal(PARAMETER_START),
        ws1,
        name_attr,
        ws1,
        string_attr,
        ws0,
        ">",
        take_until(0.., PARAMETER_END),
        literal(PARAMETER_END),
    )
        .parse_next(input)?;
    Ok(DsmlParameter {
        name: name.to_string(),
        value: value.to_string(),
        is_string: is_string == "true",
    })
}

/// Parse a name attribute.
fn name_attr<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}

/// Parse a string attribute.
fn string_attr<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited("string=\"", alt(("true", "false")), "\"").parse_next(input)
}

/// Parse a DSML name attribute.
fn dsml_name_attr<'i>(input: &mut DsmlInput<'i>) -> ModalResult<&'i str> {
    delimited("name=\"", take_until(1.., "\""), "\"").parse_next(input)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{DeepSeekV32ToolParser, ToolParser};
    use crate::parser::tool::test_utils::{collect_stream, split_by_chars, test_tools};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| {
                format!(
                    r#"<｜DSML｜parameter name="{name}" string="true">{value}</｜DSML｜parameter>"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"{function_name}\">\n{params}\n</｜DSML｜invoke>\n</｜DSML｜function_calls>"
        )
    }

    #[test]
    fn deepseek_v32_adjust_request_keeps_special_tokens() {
        let parser = DeepSeekV32ToolParser::new(&test_tools());
        let mut request = crate::request::ChatRequest::for_test();
        request.tools = test_tools();
        request.tool_choice = crate::request::ChatToolChoice::Auto;
        request.decode_options.skip_special_tokens = true;

        parser.adjust_request(&mut request).unwrap();
        assert!(!request.decode_options.skip_special_tokens);
    }

    #[test]
    fn deepseek_v32_parse_complete_without_tool_call_keeps_text() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v32_parse_complete_extracts_single_tool_call() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2024-01-16")],
            ))
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "location": "SF",
                "date": "2024-01-16"
            })
        );
    }

    #[test]
    fn deepseek_v32_parse_complete_preserves_prefix_text() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let output = format!(
            "Thinking... {}",
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let result = parser.parse_complete(&output).unwrap();

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_parse_complete_converts_schema_types() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                "<｜DSML｜function_calls>\n\
                 <｜DSML｜invoke name=\"convert\">\n\
                 <｜DSML｜parameter name=\"whole\" string=\"false\">5.0</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"flag\" string=\"false\">true</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"payload\" string=\"false\">{\"nested\":true}</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"items\" string=\"false\">[1,2]</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"empty\" string=\"false\">null</｜DSML｜parameter>\n\
                 </｜DSML｜invoke>\n\
                 </｜DSML｜function_calls>",
            )
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": null,
            })
        );
    }

    #[test]
    fn deepseek_v32_parse_complete_string_attr_overrides_schema_types() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                "<｜DSML｜function_calls>\n\
                 <｜DSML｜invoke name=\"convert\">\n\
                 <｜DSML｜parameter name=\"whole\" string=\"true\">5.0</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"flag\" string=\"true\">true</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"payload\" string=\"true\">{\"nested\":true}</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"items\" string=\"true\">[1,2]</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"empty\" string=\"true\">null</｜DSML｜parameter>\n\
                 </｜DSML｜invoke>\n\
                 </｜DSML｜function_calls>",
            )
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "whole": "5.0",
                "flag": "true",
                "payload": "{\"nested\":true}",
                "items": "[1,2]",
                "empty": "null",
            })
        );
    }

    #[test]
    fn deepseek_v32_streaming_extracts_single_tool_call() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_preserves_prefix_text() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "Thinking... ",
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
            ],
        );

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &["Hello, ", "world!"]);

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v32_streaming_extracts_multiple_tool_calls_in_order() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[&format!(
                "{}\n{}",
                build_tool_call("get_weather", &[("location", "SF")])
                    .trim_end_matches("</｜DSML｜function_calls>"),
                "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>"
            )],
        );

        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[1].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[1].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_handles_start_token_split_across_chunks() {
        let text = build_tool_call("get_weather", &[("location", "SF")]);
        let chunks = split_by_chars(&text, 5);
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_handles_bpe_chunked_dsml_opener() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜",
                "function",
                "_c",
                "all",
                "s",
                ">\n",
                "<｜DSML｜",
                "invoke",
                " name=\"",
                "get_weather",
                "\">\n",
                "<｜DSML｜",
                "parameter",
                " name=\"location\" string=\"true\">",
                "Beijing",
                "</｜DSML｜",
                "parameter>\n",
                "</｜DSML｜",
                "invoke>\n",
                "</｜DSML｜",
                "function_calls>",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "Beijing" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_truncated_parameter_does_not_leak_eos() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo",
                "<｜end▁of▁sentence｜>",
            ],
        );

        assert!(result.calls.is_empty());
        assert!(!result.normal_text.contains("<｜end▁of▁sentence｜>"));
    }

    #[test]
    fn deepseek_v32_streaming_drops_eos_after_complete_tool_calls() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls><｜end▁of▁sentence｜>",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn deepseek_v32_streaming_ignores_text_after_complete_tool_calls() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
                "trailing text",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_streaming_does_not_emit_incomplete_invoke() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v32_parser_state_resets_after_finish() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let first = parser
            .parse_complete(&build_tool_call("get_weather", &[("location", "SF")]))
            .unwrap();
        let second = parser
            .parse_complete(&build_tool_call("get_weather", &[("location", "NYC")]))
            .unwrap();

        assert_eq!(first.calls.len(), 1);
        assert_eq!(second.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&second.calls[0].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_matches_parse_complete() {
        let full_text = build_tool_call("add", &[("x", "3"), ("y", "4")]);
        let chunks = split_by_chars(&full_text, 7);
        let mut streaming_parser = DeepSeekV32ToolParser::new(&test_tools());
        let streamed = collect_stream(&mut streaming_parser, &chunks);

        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let complete = parser.parse_complete(&full_text).unwrap();

        assert_eq!(streamed.normal_text, complete.normal_text);
        assert_eq!(streamed.calls, complete.calls);
    }
}
