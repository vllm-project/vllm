use serde_json::{Map, Number, Value};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, opt, separated, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};
use winnow::token::{literal, take_till, take_until};

use super::utils::{incomplete, parse_buffered_event, partial_prefix_len, safe_text_len};
use super::{Result, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::tool::Tool;

const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

type Gemma4Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq)]
enum Gemma4Event {
    Text { len: usize },
    ToolCallStart,
    ToolCallHeader { name: String },
    ToolCall { args: Map<String, Value> },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Gemma4ArgsScanState {
    scanned_len: usize,
    in_string: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
enum Gemma4Mode {
    #[default]
    Text,
    Header,
    ToolCall {
        name: String,
        args_scan: Gemma4ArgsScanState,
    },
}

/// Tool parser for Google Gemma4 models.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/bf45e6d0a558da2b8d7b60efb07b4aa394f3b60b/vllm/tool_parsers/gemma4_tool_parser.py>
///
/// Handles the Gemma4 function call format:
///
/// `<|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>`
///
/// Arguments are emitted only after a full Gemma4 tool call is parsed.
pub struct Gemma4ToolParser {
    buffer: String,
    mode: Gemma4Mode,
    emitted_tool_count: usize,
}

impl Gemma4ToolParser {
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Gemma4Mode::default(),
            emitted_tool_count: 0,
        }
    }

    fn apply_event(&mut self, event: Gemma4Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            Gemma4Event::Text { len: consumed_len } => {
                output.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            Gemma4Event::ToolCallStart => self.mode = Gemma4Mode::Header,
            Gemma4Event::ToolCallHeader { name } => {
                self.mode = Gemma4Mode::ToolCall {
                    name,
                    args_scan: Gemma4ArgsScanState::default(),
                };
            }
            Gemma4Event::ToolCall { args } => {
                let mode = std::mem::replace(&mut self.mode, Gemma4Mode::Text);
                let Gemma4Mode::ToolCall { name, .. } = mode else {
                    return Err(parsing_failed!(
                        "Gemma4 arguments without an active tool call"
                    ));
                };
                let arguments = serde_json::to_string(&args)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                output.calls.push(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        let raw = match std::mem::replace(&mut self.mode, Gemma4Mode::Text) {
            Gemma4Mode::Text => std::mem::take(&mut self.buffer),
            Gemma4Mode::Header => {
                format!("{}{}", TOOL_CALL_START, std::mem::take(&mut self.buffer))
            }
            Gemma4Mode::ToolCall { name, .. } => {
                format!(
                    "{}{}{}{{{}",
                    TOOL_CALL_START,
                    CALL_PREFIX,
                    name,
                    std::mem::take(&mut self.buffer)
                )
            }
        };
        self.mode = Gemma4Mode::Text;
        self.emitted_tool_count = 0;
        raw
    }
}

impl ToolParser for Gemma4ToolParser {
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

        while let Some((event, consumed_len)) = {
            parse_buffered_event(&self.buffer, |input| {
                parse_next_gemma4_event(input, &mut self.mode)
            })?
        } {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();

        match &self.mode {
            Gemma4Mode::Text => output.normal_text.push_str(&self.buffer),
            Gemma4Mode::Header | Gemma4Mode::ToolCall { .. } => {
                return Err(parsing_failed!("incomplete Gemma4 tool call"));
            }
        }

        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Gemma4ToolParser::reset(self)
    }
}

/// Parse one Gemma4 event from buffered streaming input.
fn parse_next_gemma4_event(
    input: &mut Gemma4Input<'_>,
    mode: &mut Gemma4Mode,
) -> ModalResult<Gemma4Event> {
    match mode {
        Gemma4Mode::Text => parse_text_event(input),
        Gemma4Mode::Header => tool_call_header_event(input),
        Gemma4Mode::ToolCall { args_scan, .. } => tool_call_args_event(input, args_scan),
    }
}

/// Parse a Gemma4 text-mode event.
fn parse_text_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    alt((tool_call_start_event, safe_text_event)).parse_next(input)
}

/// Parse a Gemma4 tool-call start marker.
fn tool_call_start_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    literal(TOOL_CALL_START).value(Gemma4Event::ToolCallStart).parse_next(input)
}

/// Parse a Gemma4 tool-call header.
fn tool_call_header_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    let (name,) = seq!(
        _: literal(CALL_PREFIX),
        gemma4_tool_name,
        _: literal("{"),
    )
    .parse_next(input)?;
    Ok(Gemma4Event::ToolCallHeader { name })
}

/// Parse complete Gemma4 tool-call arguments.
fn tool_call_args_event(
    input: &mut Gemma4Input<'_>,
    args_scan: &mut Gemma4ArgsScanState,
) -> ModalResult<Gemma4Event> {
    let raw_args = gemma4_raw_args_until_tool_call_end(input, args_scan)?;
    let Some(args_input) = raw_args.strip_suffix('}') else {
        return Err(ErrMode::Cut(ContextError::new()));
    };
    let args = parse_gemma4_args(args_input)?;

    Ok(Gemma4Event::ToolCall { args })
}

/// Parse a Gemma4 tool name.
fn gemma4_tool_name(input: &mut Gemma4Input<'_>) -> ModalResult<String> {
    let name = take_until(1.., "{").parse_next(input)?.trim();
    if name.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(name.to_string())
}

/// Parse a safe text run before the next Gemma4 marker.
fn safe_text_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    safe_text_len(input, TOOL_CALL_START).map(|len| Gemma4Event::Text { len })
}

/// Parse raw Gemma4 arguments through the first end marker outside a Gemma string.
fn gemma4_raw_args_until_tool_call_end<'i>(
    input: &mut Gemma4Input<'i>,
    state: &mut Gemma4ArgsScanState,
) -> ModalResult<&'i str> {
    let text = **input;
    if state.scanned_len > text.len() {
        return incomplete();
    }

    loop {
        let rest = &text[state.scanned_len..];
        if state.in_string {
            let Some(string_delim) = rest.find(STRING_DELIM) else {
                state.scanned_len = safe_scan_len(text, state.scanned_len, &[STRING_DELIM]);
                return incomplete();
            };

            state.scanned_len += string_delim + STRING_DELIM.len();
            state.in_string = false;
            continue;
        }

        let next_string_delim = rest.find(STRING_DELIM);
        let next_tool_call_end = rest.find(TOOL_CALL_END);
        match (next_string_delim, next_tool_call_end) {
            (Some(string_delim), Some(tool_call_end)) if tool_call_end < string_delim => {
                let end = state.scanned_len + tool_call_end;
                state.scanned_len = end + TOOL_CALL_END.len();
                input.next_slice(state.scanned_len);
                return Ok(&text[..end]);
            }
            (Some(string_delim), _) => {
                state.scanned_len += string_delim + STRING_DELIM.len();
                state.in_string = true;
            }
            (None, Some(tool_call_end)) => {
                let end = state.scanned_len + tool_call_end;
                state.scanned_len = end + TOOL_CALL_END.len();
                input.next_slice(state.scanned_len);
                return Ok(&text[..end]);
            }
            (None, None) => {
                state.scanned_len =
                    safe_scan_len(text, state.scanned_len, &[STRING_DELIM, TOOL_CALL_END]);
                return incomplete();
            }
        }
    }
}

/// Return the scan length while holding back a split marker prefix.
fn safe_scan_len(text: &str, start: usize, markers: &[&str]) -> usize {
    let max_partial = markers
        .iter()
        .map(|marker| partial_prefix_len(&text[start..], marker))
        .max()
        .unwrap_or(0);
    text.len() - max_partial
}

/// Parse complete Gemma4 custom key-value arguments.
fn parse_gemma4_args(args: &str) -> ModalResult<Map<String, Value>> {
    let mut input = args;
    terminated(gemma4_args, eof).parse_next(&mut input)
}

/// Parse Gemma4's custom key-value argument object content.
fn gemma4_args(input: &mut &str) -> ModalResult<Map<String, Value>> {
    let pairs: Vec<(String, Value)> = delimited(
        ws0,
        terminated(
            separated(0.., gemma4_pair, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)?;
    Ok(pairs.into_iter().collect())
}

/// Parse a Gemma4 key-value pair.
fn gemma4_pair(input: &mut &str) -> ModalResult<(String, Value)> {
    let (key, value) = seq!(
        _: ws0,
        gemma4_key,
        _: ws0,
        _: literal(":"),
        _: ws0,
        gemma4_value,
    )
    .parse_next(input)?;
    Ok((key, value))
}

/// Parse a Gemma4 bare key.
fn gemma4_key(input: &mut &str) -> ModalResult<String> {
    let key = take_till(1.., |char: char| char == ':').parse_next(input)?.trim();
    if key.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(key.to_string())
}

/// Parse a Gemma4 value.
fn gemma4_value(input: &mut &str) -> ModalResult<Value> {
    alt((
        gemma4_string.map(|value: &str| Value::String(value.to_string())),
        gemma4_object.map(Value::Object),
        gemma4_array_value.map(Value::Array),
        gemma4_bare_value,
    ))
    .parse_next(input)
}

/// Parse a Gemma4 string delimited by `<|"|>`.
fn gemma4_string<'i>(input: &mut &'i str) -> ModalResult<&'i str> {
    delimited(
        literal(STRING_DELIM),
        take_until(0.., STRING_DELIM),
        literal(STRING_DELIM),
    )
    .parse_next(input)
}

/// Parse a nested Gemma4 object.
fn gemma4_object(input: &mut &str) -> ModalResult<Map<String, Value>> {
    delimited(literal("{"), gemma4_args, literal("}")).parse_next(input)
}

/// Parse a Gemma4 array value.
fn gemma4_array_value(input: &mut &str) -> ModalResult<Vec<Value>> {
    delimited(literal("["), gemma4_array_content, literal("]")).parse_next(input)
}

/// Parse Gemma4 array content.
fn gemma4_array_content(input: &mut &str) -> ModalResult<Vec<Value>> {
    delimited(
        ws0,
        terminated(
            separated(0.., gemma4_value, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)
}

/// Parse a Gemma4 bare scalar.
fn gemma4_bare_value(input: &mut &str) -> ModalResult<Value> {
    take_till(1.., |char: char| matches!(char, ',' | '}' | ']'))
        .map(parse_gemma4_scalar)
        .parse_next(input)
}

/// Parse a Gemma4 comma separator.
fn comma_separator(input: &mut &str) -> ModalResult<()> {
    delimited(ws0, literal(","), ws0).void().parse_next(input)
}

fn parse_gemma4_scalar(value: &str) -> Value {
    let value = value.trim();
    if value.is_empty() {
        return Value::String(String::new());
    }
    if value == "true" {
        return Value::Bool(true);
    }
    if value == "false" {
        return Value::Bool(false);
    }
    if matches!(value, "null" | "none" | "nil" | "NULL" | "None" | "NIL") {
        return Value::Null;
    }
    if value.contains('.') {
        if let Ok(parsed) = value.parse::<f64>()
            && let Some(number) = Number::from_f64(parsed)
        {
            return Value::Number(number);
        }
    } else if let Ok(parsed) = value.parse::<i64>() {
        return Value::Number(Number::from(parsed));
    }

    Value::String(value.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;
    use winnow::combinator::{eof, terminated};
    use winnow::error::ErrMode;
    use winnow::prelude::*;

    use super::{
        Gemma4ToolParser, ToolCallDelta, ToolParser, ToolParserOutput, gemma4_array_content,
        parse_gemma4_args,
    };
    use crate::tool::{Tool, ToolParserTestExt as _};

    fn parse_gemma4_array(array: &str) -> super::Result<Vec<Value>> {
        let mut input = array;
        match terminated(gemma4_array_content, eof).parse_next(&mut input) {
            Ok(value) => Ok(value),
            Err(ErrMode::Incomplete(_)) => Err(parsing_failed!("incomplete Gemma4 array")),
            Err(ErrMode::Backtrack(error) | ErrMode::Cut(error)) => {
                Err(parsing_failed!("{}", error))
            }
        }
    }

    fn test_tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "get_time".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "write_file".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "Edit".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "search".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "set".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "get_status".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            Tool {
                name: "todowrite".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
        ]
    }

    fn collect_stream(chunks: &[&str]) -> ToolParserOutput {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        for chunk in chunks {
            output.append(parser.parse_chunk(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        output.coalesce_calls()
    }

    fn first_call(output: &ToolParserOutput) -> &ToolCallDelta {
        output.calls.first().expect("expected one tool call")
    }

    #[test]
    fn gemma4_parse_args_handles_scalars_and_nested_values() {
        let parsed = parse_gemma4_args(
            "name:<|\"|>test<|\"|>,count:42,active:true,score:114.514,nested:{inner:<|\"|>value<|\"|>},items:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]",
        )
        .unwrap();

        assert_eq!(
            Value::Object(parsed),
            json!({
                "name": "test",
                "count": 42,
                "active": true,
                "score": 114.514,
                "nested": { "inner": "value" },
                "items": ["a", "b"],
            })
        );
    }

    #[test]
    fn gemma4_parse_args_handles_empty_arguments() {
        let parsed = parse_gemma4_args("").unwrap();
        assert_eq!(Value::Object(parsed), json!({}));
    }

    #[test]
    fn gemma4_parse_array_handles_bare_values() {
        let parsed = parse_gemma4_array("42,true,114.514").unwrap();
        assert_eq!(Value::Array(parsed), json!([42, true, 114.514]));
    }

    #[test]
    fn gemma4_parse_complete_extracts_single_tool_call() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let output = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>")
            .unwrap();

        assert!(output.normal_text.is_empty());
        assert_eq!(output.calls.len(), 1);
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn gemma4_parse_complete_rejects_incomplete_tool_call() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let error = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>London")
            .unwrap_err();

        assert!(error.to_report_string().contains("incomplete Gemma4 tool call"));
    }

    #[test]
    fn gemma4_streaming_basic_single_tool_call() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>Paris",
            ", France",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert!(output.normal_text.is_empty());
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "Paris, France" })
        );
    }

    #[test]
    fn gemma4_streaming_text_before_and_after_tool_call() {
        let output = collect_stream(&[
            "Let me check ",
            "the weather. ",
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>London<|\"|>}",
            "<tool_call|><",
            "div>",
        ]);

        assert_eq!(output.normal_text, "Let me check the weather. <div>");
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn gemma4_streaming_waits_for_complete_tool_call() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();

        for chunk in [
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>Paris<|\"|>}",
        ] {
            output.append(parser.parse_chunk(chunk).unwrap());
            assert!(output.calls.is_empty());
        }

        output.append(parser.parse_chunk("<tool_call|>").unwrap());
        let output = output.coalesce_calls();

        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "Paris" })
        );
    }

    #[test]
    fn gemma4_streaming_handles_boolean_split_across_chunks() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:search{input:{all:tru",
            "e}}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("search"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "input": { "all": true } })
        );
    }

    #[test]
    fn gemma4_streaming_handles_false_split_across_chunks() {
        let output = collect_stream(&["<|tool_call>", "call:set{flag:fals", "e}", "<tool_call|>"]);

        assert_eq!(first_call(&output).name.as_deref(), Some("set"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "flag": false })
        );
    }

    #[test]
    fn gemma4_streaming_handles_number_split_across_chunks() {
        let output = collect_stream(&["<|tool_call>", "call:set{count:4", "2}", "<tool_call|>"]);

        assert_eq!(first_call(&output).name.as_deref(), Some("set"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "count": 42 })
        );
    }

    #[test]
    fn gemma4_streaming_handles_split_string_delimiter() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:todowrite{",
            "content:<|\"|>Buy milk<|",
            "\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("todowrite"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "content": "Buy milk" })
        );
        assert!(!first_call(&output).arguments.contains("<|"));
    }

    #[test]
    fn gemma4_streaming_handles_split_tool_call_end_marker() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:get_weather{location:<|\"|>Paris<|\"|>}<tool",
            "_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "Paris" })
        );
    }

    #[test]
    fn gemma4_streaming_handles_end_marker_literal_inside_string() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:todowrite{",
            "content:<|\"|>literal }<tool_call|> inside",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("todowrite"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "content": "literal }<tool_call|> inside" })
        );
    }

    #[test]
    fn gemma4_streaming_handles_html_argument_without_duplication() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:write_file{",
            "path:<|\"|>index.html<|\"|>,",
            "content:<|\"|><!DOCTYPE html>\n<",
            "html lang=\"zh-CN\">\n<",
            "head>\n    <",
            "meta charset=\"UTF-8\">\n    <",
            "meta name=\"viewport\" content=\"width=device-width\">\n",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("write_file"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({
                "path": "index.html",
                "content": "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width\">\n",
            })
        );
    }

    #[test]
    fn gemma4_streaming_trailing_bare_bool_is_not_duplicated() {
        let output = collect_stream(&[
            "<|tool_call>",
            "call:Edit{",
            "file_path:<|\"|>src/env.py<|\"|>,",
            "old_string:<|\"|>old_val<|\"|>,",
            "new_string:<|\"|>new_val<|\"|>,",
            "replace_all:",
            "false}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&output).name.as_deref(), Some("Edit"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({
                "file_path": "src/env.py",
                "old_string": "old_val",
                "new_string": "new_val",
                "replace_all": false,
            })
        );
        assert_eq!(
            first_call(&output).arguments.matches("replace_all").count(),
            1
        );
    }

    #[test]
    fn gemma4_finish_flushes_partial_start_marker_as_text() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut output = parser.parse_chunk("<").unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text, "<");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn gemma4_finish_rejects_complete_args_without_end_marker() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        for chunk in ["<|tool_call>", "call:get_status{}"] {
            parser.parse_chunk(chunk).unwrap();
        }

        let error = parser.finish().unwrap_err();

        assert!(error.to_report_string().contains("incomplete Gemma4 tool call"));
    }

    #[test]
    fn gemma4_reset_preserves_internally_buffered_arguments() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        for chunk in [
            "<|tool_call>",
            "call:write_file{",
            "content:<|\"|>hello ",
            "world<|\"|>",
        ] {
            parser.parse_chunk(chunk).unwrap();
        }

        let raw = parser.reset();

        assert_eq!(
            raw,
            "<|tool_call>call:write_file{content:<|\"|>hello world<|\"|>"
        );
    }

    #[test]
    fn gemma4_reset_preserves_completed_arguments_after_parse_error() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let input = "<|tool_call>call:set{broken}<tool_call|>";

        let _error = parser.parse_chunk(input).unwrap_err();
        let raw = parser.reset();

        assert_eq!(raw, input);
    }
}
