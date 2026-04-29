use serde_json::{Number, Value};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, eof, repeat, terminated};
use winnow::error::{ErrMode, Needed};
use winnow::prelude::*;
use winnow::stream::{Offset, Partial, Stream};
use winnow::token::{literal, rest, take_until};

use super::utils::partial_prefix_len;
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError, parsing_failed};
use crate::request::ChatTool;

const TOOL_CALLS_START: &str = "<｜DSML｜function_calls>";
const TOOL_CALLS_END: &str = "</｜DSML｜function_calls>";
const INVOKE_START: &str = "<｜DSML｜invoke";
const INVOKE_END: &str = "</｜DSML｜invoke>";
const PARAMETER_START: &str = "<｜DSML｜parameter";
const PARAMETER_END: &str = "</｜DSML｜parameter>";

type DsmlInput<'i> = Partial<&'i str>;

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
        raw_params: Vec<(String, String)>,
    },
    ToolCallsEnd,
    IgnoredRest,
}

/// Tool parser for DeepSeek V3.2 models.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/bf45e6d0a558da2b8d7b60efb07b4aa394f3b60b/vllm/tool_parsers/deepseekv32_tool_parser.py>
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
/// Streaming strategy: **buffer until one complete invoke closes**
///
/// Unlike parsers that stream argument fragments incrementally, DeepSeek V3.2
/// waits until one full `<｜DSML｜invoke>...</｜DSML｜invoke>` block is available
/// and then emits one complete tool call with full JSON arguments.
///
/// DeepSeek V3.2 relies on DSML markers such as `｜DSML｜`, which are represented
/// as special tokens in the tokenizer and therefore must be preserved during
/// decode for parsing to work.
pub struct DeepSeekV32ToolParser {
    buffer: String,
    mode: DsmlMode,
    emitted_invoke_count: usize,
    tools: Vec<ChatTool>,
}

impl DeepSeekV32ToolParser {
    fn new(tools: &[ChatTool]) -> Self {
        Self {
            buffer: String::new(),
            mode: DsmlMode::Text,
            emitted_invoke_count: 0,
            tools: tools.to_vec(),
        }
    }

    fn apply_event(&mut self, event: DsmlEvent, result: &mut ToolParseResult) -> Result<()> {
        match event {
            DsmlEvent::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            DsmlEvent::ToolCallsStart => self.mode = DsmlMode::ToolBlock,
            DsmlEvent::Invoke { name, raw_params } => {
                let arguments = self.convert_params_with_schema(&name, raw_params)?;
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

    /// Convert raw string parameter values using the tool schema types.
    fn convert_params_with_schema(
        &self,
        function_name: &str,
        params: Vec<(String, String)>,
    ) -> Result<serde_json::Map<String, Value>> {
        let mut converted = serde_json::Map::new();
        for (name, value) in params {
            let types = self.lookup_param_types(function_name, &name);
            let converted_value = self.convert_param_value(&value, &types);
            converted.insert(name, converted_value);
        }
        Ok(converted)
    }

    /// Look up one parameter's declared schema types, defaulting to `string`.
    fn lookup_param_types(&self, function_name: &str, param_name: &str) -> Vec<String> {
        let Some(tool) = self.tools.iter().find(|tool| tool.name == function_name) else {
            return vec!["string".to_string()];
        };
        let Some(properties) = tool.parameters.get("properties").and_then(Value::as_object) else {
            return vec!["string".to_string()];
        };
        let Some(param_schema) = properties.get(param_name) else {
            return vec!["string".to_string()];
        };
        match param_schema.get("type") {
            Some(Value::String(kind)) => vec![kind.clone()],
            Some(Value::Array(kinds)) => {
                let kinds = kinds
                    .iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>();
                if kinds.is_empty() {
                    vec!["string".to_string()]
                } else {
                    kinds
                }
            }
            _ => vec!["string".to_string()],
        }
    }

    /// Convert one parameter value to the first compatible schema type.
    fn convert_param_value(&self, value: &str, param_types: &[String]) -> Value {
        if value.eq_ignore_ascii_case("null") {
            return Value::Null;
        }

        for param_type in param_types {
            if let Ok(converted) = convert_param_value_checked(value, param_type) {
                return converted;
            }
        }

        Value::String(value.to_string())
    }

    /// Reset all streaming state.
    fn reset(&mut self) {
        self.buffer.clear();
        self.mode = DsmlMode::Text;
        self.emitted_invoke_count = 0;
    }
}

impl ToolParser for DeepSeekV32ToolParser {
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn adjust_request(&self, request: &mut crate::request::ChatRequest) -> Result<()> {
        if request.tool_parsing_enabled() {
            // Preserve DSML sentinels like `｜DSML｜function_calls` during decode.
            request.decode_options.skip_special_tokens = false;
        }
        Ok(())
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        // Extract tool calls from streaming model output.
        //
        // Uses a buffer-until-complete-invoke strategy: text is buffered until
        // a complete invoke block is available, then parsed and emitted in one
        // shot.
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        loop {
            let mut input = Partial::new(self.buffer.as_str());
            let checkpoint = input.checkpoint();
            let event = match parse_next_dsml_event(&mut input, self.mode) {
                Ok(event) => event,
                Err(ErrMode::Incomplete(_)) => break,
                Err(error) => {
                    return Err(parsing_failed!("failed to parse DSML event: {}", error));
                }
            };
            let consumed_len = input.offset_from(&checkpoint);
            if consumed_len == 0 {
                break;
            }
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

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
fn parse_next_dsml_event(input: &mut DsmlInput<'_>, mode: DsmlMode) -> ModalResult<DsmlEvent> {
    match mode {
        DsmlMode::Text => parse_text_event(input),
        DsmlMode::ToolBlock => parse_tool_block_event(input),
        DsmlMode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode DSML event.
fn parse_text_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    alt((tool_calls_start_event, safe_text_event)).parse_next(input)
}

/// Parse a tool-block DSML event.
fn parse_tool_block_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    ws0.void().parse_next(input)?;
    alt((invoke_event, tool_calls_end_event)).parse_next(input)
}

/// Parse a DSML function-calls start marker.
fn tool_calls_start_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    literal(TOOL_CALLS_START)
        .value(DsmlEvent::ToolCallsStart)
        .parse_next(input)
}

/// Parse a DSML function-calls end marker.
fn tool_calls_end_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    literal(TOOL_CALLS_END)
        .value(DsmlEvent::ToolCallsEnd)
        .parse_next(input)
}

/// Parse a trailing rest after DSML function calls.
fn ignored_rest_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    rest.value(DsmlEvent::IgnoredRest).parse_next(input)
}

/// Parse a safe text run before the next DSML marker.
fn safe_text_event(input: &mut DsmlInput<'_>) -> ModalResult<DsmlEvent> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    if let Some(start_idx) = text.find(TOOL_CALLS_START) {
        input.next_slice(start_idx);
        return Ok(DsmlEvent::Text { len: start_idx });
    }

    let keep_len = partial_prefix_len(text, TOOL_CALLS_START);
    let emit_len = text.len().saturating_sub(keep_len);
    if emit_len == 0 {
        return incomplete();
    }

    input.next_slice(emit_len);
    Ok(DsmlEvent::Text { len: emit_len })
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
fn parse_invoke_params(invoke_body: &str) -> ModalResult<Vec<(String, String)>> {
    let mut input = invoke_body;
    delimited(ws0, repeat(0.., terminated(parse_parameter, ws0)), eof).parse_next(&mut input)
}

/// Parse a DSML parameter block.
fn parse_parameter(input: &mut &str) -> ModalResult<(String, String)> {
    let (_, _, name, _, _, _, _, value, _) = (
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
    Ok((name.to_string(), value.to_string()))
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

fn incomplete<T>() -> ModalResult<T> {
    Err(ErrMode::Incomplete(Needed::Unknown))
}

/// Convert a parameter value to the requested type.
fn convert_param_value_checked(value: &str, param_type: &str) -> std::result::Result<Value, ()> {
    match param_type.to_ascii_lowercase().as_str() {
        "string" | "str" | "text" => Ok(Value::String(value.to_string())),
        "integer" | "int" => value
            .parse::<i64>()
            .map(Number::from)
            .map(Value::Number)
            .map_err(|_| ()),
        "number" | "float" => {
            let parsed = value.parse::<f64>().map_err(|_| ())?;
            if parsed.is_finite()
                && parsed.fract() == 0.0
                && parsed >= i64::MIN as f64
                && parsed <= i64::MAX as f64
            {
                Ok(Value::Number(Number::from(parsed as i64)))
            } else {
                Number::from_f64(parsed).map(Value::Number).ok_or(())
            }
        }
        "boolean" | "bool" => {
            let trimmed = value.trim();
            match trimmed.to_ascii_lowercase().as_str() {
                "true" | "1" => Ok(Value::Bool(true)),
                "false" | "0" => Ok(Value::Bool(false)),
                _ => Err(()),
            }
        }
        "object" | "array" => serde_json::from_str(value).map_err(|_| ()),
        _ => serde_json::from_str(value).map_err(|_| ()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{DeepSeekV32ToolParser, ToolParser, convert_param_value_checked};
    use crate::request::ChatTool;

    fn test_tools() -> Vec<ChatTool> {
        vec![
            ChatTool {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" },
                        "date": { "type": "string" }
                    }
                }),
                strict: None,
            },
            ChatTool {
                name: "add".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": { "type": "integer" },
                        "y": { "type": "integer" }
                    }
                }),
                strict: None,
            },
            ChatTool {
                name: "convert".to_string(),
                description: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "whole": { "type": "number" },
                        "flag": { "type": "boolean" },
                        "payload": { "type": "object" },
                        "items": { "type": "array" },
                        "empty": { "type": "string" }
                    }
                }),
                strict: None,
            },
        ]
    }

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

    fn collect_stream(chunks: &[&str], tools: &[ChatTool]) -> crate::parser::tool::ToolParseResult {
        let mut parser = DeepSeekV32ToolParser::new(tools);
        let mut result = crate::parser::tool::ToolParseResult::default();
        for chunk in chunks {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        result.coalesce_calls()
    }

    fn split_by_chars(text: &str, chunk_chars: usize) -> Vec<&str> {
        let mut chunks = Vec::new();
        let mut start = 0;
        let mut count = 0;

        for (index, _) in text.char_indices() {
            if count == chunk_chars {
                chunks.push(&text[start..index]);
                start = index;
                count = 0;
            }
            count += 1;
        }

        if start < text.len() {
            chunks.push(&text[start..]);
        }

        chunks
    }

    #[test]
    fn deepseek_v32_convert_param_value_handles_supported_types() {
        assert_eq!(
            convert_param_value_checked("42", "integer").unwrap(),
            json!(42)
        );
        assert_eq!(
            convert_param_value_checked("5.0", "number").unwrap(),
            json!(5)
        );
        assert_eq!(
            convert_param_value_checked("true", "boolean").unwrap(),
            json!(true)
        );
        assert_eq!(
            convert_param_value_checked(r#"{"k":1}"#, "object").unwrap(),
            json!({ "k": 1 })
        );
        assert_eq!(
            convert_param_value_checked("[1,2]", "array").unwrap(),
            json!([1, 2])
        );
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
            .parse_complete(&build_tool_call(
                "convert",
                &[
                    ("whole", "5.0"),
                    ("flag", "1"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "NULL"),
                ],
            ))
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "whole": 5,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": null,
            })
        );
    }

    #[test]
    fn deepseek_v32_streaming_extracts_single_tool_call() {
        let result = collect_stream(
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
            ],
            &test_tools(),
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
        let result = collect_stream(
            &[
                "Thinking... ",
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
            ],
            &test_tools(),
        );

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_streaming_without_tool_call_emits_text_incrementally() {
        let result = collect_stream(&["Hello, ", "world!"], &test_tools());

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v32_streaming_extracts_multiple_tool_calls_in_order() {
        let result = collect_stream(
            &[&format!(
                "{}\n{}",
                build_tool_call("get_weather", &[("location", "SF")])
                    .trim_end_matches("</｜DSML｜function_calls>"),
                "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>"
            )],
            &test_tools(),
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
        let result = collect_stream(&chunks, &test_tools());

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_handles_bpe_chunked_dsml_opener() {
        let result = collect_stream(
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
            &test_tools(),
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
        let result = collect_stream(
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo",
                "<｜end▁of▁sentence｜>",
            ],
            &test_tools(),
        );

        assert!(result.calls.is_empty());
        assert!(!result.normal_text.contains("<｜end▁of▁sentence｜>"));
    }

    #[test]
    fn deepseek_v32_streaming_drops_eos_after_complete_tool_calls() {
        let result = collect_stream(
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls><｜end▁of▁sentence｜>",
            ],
            &test_tools(),
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn deepseek_v32_streaming_ignores_text_after_complete_tool_calls() {
        let result = collect_stream(
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
                "trailing text",
            ],
            &test_tools(),
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_streaming_does_not_emit_incomplete_invoke() {
        let result = collect_stream(
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
            ],
            &test_tools(),
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
        let streamed = collect_stream(&chunks, &test_tools());

        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let complete = parser.parse_complete(&full_text).unwrap();

        assert_eq!(streamed.normal_text, complete.normal_text);
        assert_eq!(streamed.calls, complete.calls);
    }
}
