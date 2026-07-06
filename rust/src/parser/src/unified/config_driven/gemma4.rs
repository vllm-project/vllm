//! Gemma4 format for the config-driven unified parser.
//!
//! Original Python implementation:
//! <https://github.com/vllm-project/vllm/blob/main/vllm/parser/gemma4.py>
//!
//! Handles Gemma4 reasoning and function-call formats:
//!
//! `<|channel>thought\nreasoning<channel|>`
//!
//! `<|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>`

use serde_json::{Map, Number, Value};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, eof, opt, separated, seq, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::prelude::*;
use winnow::stream::Stream;
use winnow::token::{literal, take_till, take_until};

use super::{ArgsEndScan, ConfigDrivenParser, Input, ParserFormat};
use crate::tool::ToolSchema;
use crate::utils::{incomplete, partial_prefix_len};

const CHANNEL_START: &str = "<|channel>";
const CHANNEL_END: &str = "<channel|>";
const REASONING_START: &str = "<|channel>thought\n";
const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

/// Zero-sized marker type providing the Google Gemma4 format.
pub struct Gemma4Format;

impl ParserFormat for Gemma4Format {
    const NAME: &'static str = "Gemma4";
    const REASONING_START: Option<&'static str> = Some(REASONING_START);
    const REASONING_END: Option<&'static str> = Some(CHANNEL_END);
    const CALL_START: &'static str = TOOL_CALL_START;
    const CALL_END: &'static str = TOOL_CALL_END;
    const REASONING_BOUNDARY_TOKENS: Option<(&'static str, &'static str)> =
        Some((CHANNEL_START, CHANNEL_END));

    type ArgsScan = Gemma4ArgsScan;

    fn tool_header(input: &mut Input<'_>) -> ModalResult<String> {
        let (name,) = seq!(
            _: literal(CALL_PREFIX),
            gemma4_tool_name,
            _: literal("{"),
        )
        .parse_next(input)?;
        Ok(name)
    }

    fn tool_args(_schema: &ToolSchema, body: &str) -> ModalResult<Map<String, Value>> {
        let Some(args_input) = body.strip_suffix('}') else {
            return Err(ErrMode::Cut(ContextError::new()));
        };
        parse_gemma4_args(args_input)
    }
}

/// Config-driven unified parser for Google Gemma4 models.
///
/// Arguments are emitted only after a full Gemma4 tool call is parsed.
pub type Gemma4ConfigDrivenParser = ConfigDrivenParser<Gemma4Format>;

/// Resumable Gemma4 argument scanner: locates the call end marker outside
/// `<|"|>`-delimited strings, so marker-shaped literals inside string values
/// do not terminate the call early.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Gemma4ArgsScan {
    scanned_len: usize,
    in_string: bool,
}

impl ArgsEndScan for Gemma4ArgsScan {
    fn scan<'i>(&mut self, input: &mut Input<'i>, end: &str) -> ModalResult<&'i str> {
        let text = **input;
        if self.scanned_len > text.len() {
            return incomplete();
        }

        loop {
            let rest = &text[self.scanned_len..];
            if self.in_string {
                let Some(string_delim) = rest.find(STRING_DELIM) else {
                    self.scanned_len = safe_scan_len(text, self.scanned_len, &[STRING_DELIM]);
                    return incomplete();
                };

                self.scanned_len += string_delim + STRING_DELIM.len();
                self.in_string = false;
                continue;
            }

            let next_string_delim = rest.find(STRING_DELIM);
            let next_call_end = rest.find(end);
            match (next_string_delim, next_call_end) {
                (Some(string_delim), Some(call_end)) if call_end < string_delim => {
                    let body_end = self.scanned_len + call_end;
                    self.scanned_len = body_end + end.len();
                    input.next_slice(self.scanned_len);
                    return Ok(&text[..body_end]);
                }
                (Some(string_delim), _) => {
                    self.scanned_len += string_delim + STRING_DELIM.len();
                    self.in_string = true;
                }
                (None, Some(call_end)) => {
                    let body_end = self.scanned_len + call_end;
                    self.scanned_len = body_end + end.len();
                    input.next_slice(self.scanned_len);
                    return Ok(&text[..body_end]);
                }
                (None, None) => {
                    self.scanned_len = safe_scan_len(text, self.scanned_len, &[STRING_DELIM, end]);
                    return incomplete();
                }
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

/// Parse a Gemma4 tool name.
fn gemma4_tool_name(input: &mut Input<'_>) -> ModalResult<String> {
    let name = take_until(1.., "{").parse_next(input)?.trim();
    if name.is_empty() {
        return Err(ErrMode::Cut(ContextError::new()));
    }
    Ok(name.to_string())
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
    use std::sync::Arc;

    use serde_json::{Value, json};
    use thiserror_ext::AsReport;
    use vllm_tokenizer::test_utils::TestTokenizer;
    use winnow::combinator::{eof, terminated};
    use winnow::error::ErrMode;
    use winnow::prelude::*;

    use super::{
        CHANNEL_END, CHANNEL_START, Gemma4ConfigDrivenParser, gemma4_array_content,
        parse_gemma4_args,
    };
    use crate::tool::{Tool, ToolCallDelta};
    use crate::unified::{
        Result, UnifiedParser, UnifiedParserError, UnifiedParserEvent, UnifiedParserOutput,
        parsing_failed,
    };

    const CHANNEL_START_ID: u32 = 256;
    const CHANNEL_END_ID: u32 = 257;
    const TURN_BOUNDARY_ID: u32 = 258;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_special_token(CHANNEL_START, CHANNEL_START_ID)
            .with_special_token(CHANNEL_END, CHANNEL_END_ID)
            .with_special_token("<turn-boundary>", TURN_BOUNDARY_ID)
    }

    trait UnifiedParserTestExt {
        fn parse_chunk(&mut self, chunk: &str) -> Result<UnifiedParserOutput>;
        fn parse_complete(&mut self, text: &str) -> Result<UnifiedParserOutput>;
    }

    impl UnifiedParserTestExt for Gemma4ConfigDrivenParser {
        fn parse_chunk(&mut self, chunk: &str) -> Result<UnifiedParserOutput> {
            let mut output = UnifiedParserOutput::default();
            self.parse_into(chunk, &mut output)?;
            Ok(output)
        }

        fn parse_complete(&mut self, text: &str) -> Result<UnifiedParserOutput> {
            let mut output = self.parse_chunk(text)?;
            output.append(self.finish()?);
            Ok(output)
        }
    }

    trait UnifiedOutputTestExt {
        fn normal_text(&self) -> String;
        fn reasoning_text(&self) -> String;
        fn calls(&self) -> Vec<&ToolCallDelta>;
        fn coalesce(self) -> Self;
    }

    impl UnifiedOutputTestExt for UnifiedParserOutput {
        fn normal_text(&self) -> String {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::Text(text) => Some(text.as_str()),
                    UnifiedParserEvent::Reasoning(_) | UnifiedParserEvent::ToolCall(_) => None,
                })
                .collect()
        }

        fn reasoning_text(&self) -> String {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::Reasoning(text) => Some(text.as_str()),
                    UnifiedParserEvent::Text(_) | UnifiedParserEvent::ToolCall(_) => None,
                })
                .collect()
        }

        fn calls(&self) -> Vec<&ToolCallDelta> {
            self.events
                .iter()
                .filter_map(|event| match event {
                    UnifiedParserEvent::Text(_) | UnifiedParserEvent::Reasoning(_) => None,
                    UnifiedParserEvent::ToolCall(call) => Some(call),
                })
                .collect()
        }

        fn coalesce(self) -> Self {
            self
        }
    }

    fn parse_gemma4_array(array: &str) -> Result<Vec<Value>> {
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

    fn test_parser() -> Gemma4ConfigDrivenParser {
        Gemma4ConfigDrivenParser::new(&test_tools(), Arc::new(tokenizer())).unwrap()
    }

    #[test]
    fn gemma4_create_requires_channel_start_token() {
        let error =
            match Gemma4ConfigDrivenParser::new(&test_tools(), Arc::new(TestTokenizer::new())) {
                Ok(_) => panic!("expected missing token error"),
                Err(error) => error,
            };

        assert!(matches!(
            error,
            UnifiedParserError::MissingToken { token } if token == CHANNEL_START
        ));
    }

    fn collect_stream(chunks: &[&str]) -> UnifiedParserOutput {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            output.append(parser.parse_chunk(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        output.coalesce()
    }

    fn first_call(output: &UnifiedParserOutput) -> ToolCallDelta {
        (*output.calls().first().expect("expected one tool call")).clone()
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
        let mut parser = test_parser();
        let output = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>London<|\"|>}<tool_call|>")
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn gemma4_parse_complete_rejects_incomplete_tool_call() {
        let mut parser = test_parser();
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

        assert!(output.normal_text().is_empty());
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

        assert_eq!(output.normal_text(), "Let me check the weather. <div>");
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "London" })
        );
    }

    #[test]
    fn gemma4_streaming_waits_for_complete_tool_call() {
        let mut parser = test_parser();
        let mut output = UnifiedParserOutput::default();

        for chunk in [
            "<|tool_call>",
            "call:get_weather{",
            "location:<|\"|>Paris<|\"|>}",
        ] {
            output.append(parser.parse_chunk(chunk).unwrap());
            assert!(output.calls().is_empty());
        }

        output.append(parser.parse_chunk("<tool_call|>").unwrap());
        let output = output.coalesce();

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
        let mut parser = test_parser();
        let mut output = parser.parse_chunk("<").unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), "<");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn gemma4_streaming_emits_reasoning_then_text() {
        let output = collect_stream(&["<|channel>thought\nreason<channel|>answer"]);

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "answer");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn gemma4_streaming_holds_split_reasoning_start() {
        let mut parser = test_parser();

        let first = parser.parse_chunk("<|channel>").unwrap();
        assert!(first.events.is_empty());

        let mut output = parser.parse_chunk("thought\nrea").unwrap();
        output.append(parser.parse_chunk("son<channel|>answer").unwrap());
        output.append(parser.finish().unwrap());

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn gemma4_initialize_open_channel_prompt_starts_in_reasoning() {
        let mut parser = test_parser();
        parser.initialize(&[CHANNEL_START_ID, 3000, 3001]).unwrap();

        let output = parser.parse_complete("reason<channel|>answer").unwrap();

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn gemma4_initialize_turn_prompt_starts_in_text() {
        let mut parser = test_parser();
        parser.initialize(&[TURN_BOUNDARY_ID, 3000, 3001]).unwrap();

        let output = parser.parse_complete("<|channel>thought\nreason<channel|>answer").unwrap();

        assert_eq!(output.reasoning_text(), "reason");
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn gemma4_initialize_special_token_caps_boundary_scan() {
        let mut parser = test_parser();
        parser.initialize(&[CHANNEL_START_ID, 3000, TURN_BOUNDARY_ID, 3001]).unwrap();

        let output = parser.parse_complete("answer").unwrap();

        assert!(output.reasoning_text().is_empty());
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn gemma4_initialize_closed_channel_prompt_starts_in_text() {
        let mut parser = test_parser();
        parser.initialize(&[CHANNEL_START_ID, 3000, 3001, CHANNEL_END_ID]).unwrap();

        let output = parser.parse_complete("answer").unwrap();

        assert!(output.reasoning_text().is_empty());
        assert_eq!(output.normal_text(), "answer");
    }

    #[test]
    fn gemma4_reasoning_tool_call_implicitly_ends_reasoning() {
        let output = collect_stream(&[
            "<|channel>thought\nNeed weather.",
            "<|tool_call>",
            "call:get_weather{location:<|\"|>Paris<|\"|>}",
            "<tool_call|>",
        ]);

        assert_eq!(output.reasoning_text(), "Need weather.");
        assert!(output.normal_text().is_empty());
        assert_eq!(first_call(&output).name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&output).arguments).unwrap(),
            json!({ "location": "Paris" })
        );
    }

    #[test]
    fn gemma4_bare_channel_start_is_plain_text() {
        let output = collect_stream(&["<|channel>plain"]);

        assert_eq!(output.normal_text(), "<|channel>plain");
        assert!(output.reasoning_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn gemma4_finish_rejects_complete_args_without_end_marker() {
        let mut parser = test_parser();
        for chunk in ["<|tool_call>", "call:get_status{}"] {
            parser.parse_chunk(chunk).unwrap();
        }

        let error = parser.finish().unwrap_err();

        assert!(error.to_report_string().contains("incomplete Gemma4 tool call"));
    }

    #[test]
    fn gemma4_reset_preserves_internally_buffered_arguments() {
        let mut parser = test_parser();
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
        let mut parser = test_parser();
        let input = "<|tool_call>call:set{broken}<tool_call|>";

        let _error = parser.parse_chunk(input).unwrap_err();
        let raw = parser.reset();

        assert_eq!(raw, input);
    }
}
