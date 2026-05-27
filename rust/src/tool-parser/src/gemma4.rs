use serde_json::{Map, Number, Value};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, opt, separated, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, take_till, take_until};

use super::utils::{parse_buffered_event, safe_text_len};
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser};
use crate::Tool;

const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

type Gemma4Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq)]
enum Gemma4Event {
    Text {
        len: usize,
    },
    ToolCall {
        name: String,
        args: Map<String, Value>,
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
    emitted_tool_count: usize,
}

impl Gemma4ToolParser {
    fn new(_tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            emitted_tool_count: 0,
        }
    }

    fn apply_event(&mut self, event: Gemma4Event, result: &mut ToolParseResult) -> Result<()> {
        match event {
            Gemma4Event::Text { len: consumed_len } => {
                result.normal_text.push_str(&self.buffer[..consumed_len]);
            }
            Gemma4Event::ToolCall { name, args } => {
                let arguments = serde_json::to_string(&args)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;

                result.calls.push(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
            }
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.emitted_tool_count = 0;
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

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        while let Some((event, consumed_len)) =
            parse_buffered_event(&self.buffer, parse_next_gemma4_event)?
        {
            self.apply_event(event, &mut result)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(result)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();

        if !self.buffer.is_empty() {
            if self.buffer.starts_with(TOOL_CALL_START) {
                self.reset();
                return Err(parsing_failed!("incomplete Gemma4 tool call"));
            }
            result.normal_text.push_str(&self.buffer);
        }

        self.reset();
        Ok(result)
    }
}

/// Parse one Gemma4 event from buffered streaming input.
fn parse_next_gemma4_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    alt((tool_call_event, safe_text_event)).parse_next(input)
}

/// Parse a complete Gemma4 tool call.
// TODO: incremental parsing arguments to reduce scanning from O(n^2) to O(n).
fn tool_call_event(input: &mut Gemma4Input<'_>) -> ModalResult<Gemma4Event> {
    let (name, args) = seq!(
        _: literal(TOOL_CALL_START),
        _: literal(CALL_PREFIX),
        gemma4_tool_name,
        _: literal("{"),
        gemma4_args,
        _: literal("}"),
        _: literal(TOOL_CALL_END),
    )
    .parse_next(input)?;

    Ok(Gemma4Event::ToolCall { name, args })
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

/// Parse Gemma4's custom key-value argument object content.
fn gemma4_args(input: &mut Gemma4Input<'_>) -> ModalResult<Map<String, Value>> {
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
fn gemma4_pair(input: &mut Gemma4Input<'_>) -> ModalResult<(String, Value)> {
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
fn gemma4_key(input: &mut Gemma4Input<'_>) -> ModalResult<String> {
    let key = take_till(1.., |char: char| char == ':').parse_next(input)?.trim();
    if key.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(key.to_string())
}

/// Parse a Gemma4 value.
fn gemma4_value(input: &mut Gemma4Input<'_>) -> ModalResult<Value> {
    alt((
        gemma4_string.map(|value: &str| Value::String(value.to_string())),
        gemma4_object.map(Value::Object),
        gemma4_array_value.map(Value::Array),
        gemma4_bare_value,
    ))
    .parse_next(input)
}

/// Parse a Gemma4 string delimited by `<|"|>`.
fn gemma4_string<'i>(input: &mut Gemma4Input<'i>) -> ModalResult<&'i str> {
    delimited(
        literal(STRING_DELIM),
        take_until(0.., STRING_DELIM),
        literal(STRING_DELIM),
    )
    .parse_next(input)
}

/// Parse a nested Gemma4 object.
fn gemma4_object(input: &mut Gemma4Input<'_>) -> ModalResult<Map<String, Value>> {
    delimited(literal("{"), gemma4_args, literal("}")).parse_next(input)
}

/// Parse a Gemma4 array value.
fn gemma4_array_value(input: &mut Gemma4Input<'_>) -> ModalResult<Vec<Value>> {
    delimited(literal("["), gemma4_array_content, literal("]")).parse_next(input)
}

/// Parse Gemma4 array content.
fn gemma4_array_content(input: &mut Gemma4Input<'_>) -> ModalResult<Vec<Value>> {
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
fn gemma4_bare_value(input: &mut Gemma4Input<'_>) -> ModalResult<Value> {
    take_till(1.., |char: char| matches!(char, ',' | '}' | ']'))
        .map(parse_gemma4_scalar)
        .parse_next(input)
}

/// Parse a Gemma4 comma separator.
fn comma_separator(input: &mut Gemma4Input<'_>) -> ModalResult<()> {
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
    use winnow::stream::Partial;

    use super::{
        Gemma4ToolParser, ToolCallDelta, ToolParseResult, ToolParser, gemma4_args,
        gemma4_array_content,
    };
    use crate::Tool;

    fn parse_gemma4_args(args: &str) -> super::Result<serde_json::Map<String, Value>> {
        let mut input = Partial::new(args);
        let _ = input.complete();
        match terminated(gemma4_args, eof).parse_next(&mut input) {
            Ok(value) => Ok(value),
            Err(ErrMode::Incomplete(_)) => Err(parsing_failed!("incomplete Gemma4 arguments")),
            Err(ErrMode::Backtrack(error) | ErrMode::Cut(error)) => {
                Err(parsing_failed!("{}", error))
            }
        }
    }

    fn parse_gemma4_array(array: &str) -> super::Result<Vec<Value>> {
        let mut input = Partial::new(array);
        let _ = input.complete();
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

    fn collect_stream(chunks: &[&str]) -> ToolParseResult {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut result = ToolParseResult::default();
        for chunk in chunks {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        result.coalesce_calls()
    }

    fn first_call(result: &ToolParseResult) -> &ToolCallDelta {
        result.calls.first().expect("expected one tool call")
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
        let result = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>")
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(first_call(&result).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
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
        let result = collect_stream(&[
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>Paris",
            ", France",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert!(result.normal_text.is_empty());
        assert_eq!(first_call(&result).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "location": "Paris, France" })
        );
    }

    #[test]
    fn gemma4_streaming_text_before_and_after_tool_call() {
        let result = collect_stream(&[
            "Let me check ",
            "the weather. ",
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>London<|\"|>}",
            "<tool_call|><",
            "div>",
        ]);

        assert_eq!(result.normal_text, "Let me check the weather. <div>");
        assert_eq!(first_call(&result).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn gemma4_streaming_waits_for_complete_tool_call() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut result = ToolParseResult::default();

        for chunk in [
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>Paris<|\"|>}",
        ] {
            result.append(parser.push(chunk).unwrap());
            assert!(result.calls.is_empty());
        }

        result.append(parser.push("<tool_call|>").unwrap());
        let result = result.coalesce_calls();

        assert_eq!(first_call(&result).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "location": "Paris" })
        );
    }

    #[test]
    fn gemma4_streaming_handles_boolean_split_across_chunks() {
        let result = collect_stream(&[
            "<|tool_call>",
            "call:search{input:{all:tru",
            "e}}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&result).name.as_deref(), Some("search"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "input": { "all": true } })
        );
    }

    #[test]
    fn gemma4_streaming_handles_false_split_across_chunks() {
        let result = collect_stream(&["<|tool_call>", "call:set{flag:fals", "e}", "<tool_call|>"]);

        assert_eq!(first_call(&result).name.as_deref(), Some("set"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "flag": false })
        );
    }

    #[test]
    fn gemma4_streaming_handles_number_split_across_chunks() {
        let result = collect_stream(&["<|tool_call>", "call:set{count:4", "2}", "<tool_call|>"]);

        assert_eq!(first_call(&result).name.as_deref(), Some("set"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "count": 42 })
        );
    }

    #[test]
    fn gemma4_streaming_handles_split_string_delimiter() {
        let result = collect_stream(&[
            "<|tool_call>",
            "call:todowrite{",
            "content:<|\"|>Buy milk<|",
            "\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&result).name.as_deref(), Some("todowrite"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "content": "Buy milk" })
        );
        assert!(!first_call(&result).arguments.contains("<|"));
    }

    #[test]
    fn gemma4_streaming_handles_end_marker_literal_inside_string() {
        let result = collect_stream(&[
            "<|tool_call>",
            "call:todowrite{",
            "content:<|\"|>literal }<tool_call|> inside",
            "<|\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&result).name.as_deref(), Some("todowrite"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({ "content": "literal }<tool_call|> inside" })
        );
    }

    #[test]
    fn gemma4_streaming_handles_html_argument_without_duplication() {
        let result = collect_stream(&[
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

        assert_eq!(first_call(&result).name.as_deref(), Some("write_file"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({
                "path": "index.html",
                "content": "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width\">\n",
            })
        );
    }

    #[test]
    fn gemma4_streaming_trailing_bare_bool_is_not_duplicated() {
        let result = collect_stream(&[
            "<|tool_call>",
            "call:Edit{",
            "file_path:<|\"|>src/env.py<|\"|>,",
            "old_string:<|\"|>old_val<|\"|>,",
            "new_string:<|\"|>new_val<|\"|>,",
            "replace_all:",
            "false}",
            "<tool_call|>",
        ]);

        assert_eq!(first_call(&result).name.as_deref(), Some("Edit"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({
                "file_path": "src/env.py",
                "old_string": "old_val",
                "new_string": "new_val",
                "replace_all": false,
            })
        );
        assert_eq!(
            first_call(&result).arguments.matches("replace_all").count(),
            1
        );
    }

    #[test]
    fn gemma4_finish_flushes_partial_start_marker_as_text() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut result = parser.push("<").unwrap();
        result.append(parser.finish().unwrap());

        assert_eq!(result.normal_text, "<");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn gemma4_finish_rejects_complete_args_without_end_marker() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        for chunk in ["<|tool_call>", "call:get_status{}"] {
            parser.push(chunk).unwrap();
        }

        let error = parser.finish().unwrap_err();

        assert!(error.to_report_string().contains("incomplete Gemma4 tool call"));
    }
}
