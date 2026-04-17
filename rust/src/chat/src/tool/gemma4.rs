use serde_json::{Map, Number, Value};

use super::streaming::StreamingToolState;
use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError};
use crate::request::{ChatRequest, ChatTool};

const TOOL_CALL_START: &str = "<|tool_call>";
const TOOL_CALL_END: &str = "<tool_call|>";
const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

/// Tool parser for Google Gemma4 models.
///
/// Original Python implementation:
/// <https://github.com/vllm-project/vllm/blob/bf45e6d0a558da2b8d7b60efb07b4aa394f3b60b/vllm/tool_parsers/gemma4_tool_parser.py>
///
/// Handles the Gemma4 function call format:
///
/// `<|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>`
///
/// Streaming strategy: **accumulate-then-parse-then-diff**
///
/// Instead of trying to convert Gemma4's custom format to JSON
/// token-by-token (which fails because Gemma4 uses bare keys, custom
/// delimiters, and structural braces that differ from JSON), this parser:
///
/// 1. Accumulates the raw Gemma4 argument string during streaming
/// 2. Parses it with `parse_gemma4_args()` into a JSON object
/// 3. Converts to JSON with `Value::to_string()`
/// 4. Diffs against the previously-streamed JSON string
/// 5. Emits only the new JSON fragment as the delta
///
/// This follows the same pattern used by FunctionGemma, Hermes, and Llama
/// tool parsers.
pub struct Gemma4ToolParser {
    buffer: String,
    state: StreamingToolState,
}

impl Gemma4ToolParser {
    fn new(_tools: &[ChatTool]) -> Self {
        Self {
            buffer: String::new(),
            state: StreamingToolState::default(),
        }
    }

    fn process_text_mode(&mut self, result: &mut ToolParseResult) -> bool {
        if let Some(start_idx) = self.buffer.find(TOOL_CALL_START) {
            if start_idx > 0 {
                result.normal_text.push_str(&self.buffer[..start_idx]);
                self.buffer.drain(..start_idx);
                return true;
            }

            self.buffer.drain(..TOOL_CALL_START.len());
            self.state.begin_tool_call();
            return true;
        }

        let keep_len = partial_prefix_len(&self.buffer, TOOL_CALL_START);
        let emit_len = self.buffer.len().saturating_sub(keep_len);
        if emit_len > 0 {
            result.normal_text.push_str(&self.buffer[..emit_len]);
            self.buffer.drain(..emit_len);
            return true;
        }

        false
    }

    fn process_tool_mode(&mut self, result: &mut ToolParseResult) -> Result<bool> {
        let Some(tool_index) = self.state.active_tool_index() else {
            return Ok(false);
        };
        let mut progressed = false;

        if !self.state.active_tool_name_sent() {
            match parse_tool_header(&self.buffer) {
                ToolHeader::NeedMore => return Ok(progressed),
                ToolHeader::Invalid(message) => return Err(parse_failed(message)),
                ToolHeader::Ready { name, consumed_len } => {
                    self.state.mark_active_tool_name_sent();
                    self.buffer.drain(..consumed_len);
                    result.calls.push(ToolCallDelta {
                        tool_index,
                        name: Some(name),
                        arguments: String::new(),
                    });
                    progressed = true;
                }
            }
        }

        let tail_state = scan_tool_tail(&self.buffer);
        let raw_args_end = match tail_state {
            ToolTailState::Complete { args_end, .. }
            | ToolTailState::PendingAfterBrace { args_end } => args_end,
            ToolTailState::Incomplete => self.buffer.len(),
        };
        let raw_args = self.buffer[..raw_args_end].to_string();
        let args_complete = matches!(tail_state, ToolTailState::Complete { .. });
        if self.emit_argument_diff(tool_index, &raw_args, !args_complete, result)? {
            progressed = true;
        }

        if let ToolTailState::Complete { consumed_len, .. } = tail_state {
            self.buffer.drain(..consumed_len);
            self.state.clear_active_tool();
            progressed = true;
        }

        Ok(progressed)
    }

    /// Parse raw Gemma4 arguments, convert to JSON, diff, and emit.
    ///
    /// This is the core of the accumulate-then-parse-then-diff strategy:
    /// 1. Parse `raw_args` with `parse_gemma4_args()`
    /// 2. Convert to JSON string
    /// 3. Withhold trailing closing characters (`"}`) that may move as more tokens arrive
    /// 4. Diff against previously streamed JSON and emit only new chars
    ///
    /// Why withholding is necessary:
    ///
    /// Gemma4's custom format produces *structurally incomplete* JSON
    /// during streaming. For example, when `<|"|>Paris` arrives
    /// without a closing delimiter, `parse_gemma4_args()` treats it
    /// as a complete value and produces `{"location":"Paris"}`. But
    /// when `, France<|"|>` arrives next, the JSON becomes
    /// `{"location":"Paris, France"}`. If we had sent the closing
    /// `"}` from the first parse, the concatenated client output
    /// would be `{"location":"Paris"}France"}`, which is garbage.
    ///
    /// The solution: never send trailing closing chars during
    /// streaming. They get flushed by `finish()` when the stream ends.
    fn emit_argument_diff(
        &mut self,
        tool_index: usize,
        raw_args: &str,
        partial: bool,
        result: &mut ToolParseResult,
    ) -> Result<bool> {
        let current_args = parse_gemma4_args(raw_args, partial)?;
        let current_args_json = Value::Object(current_args).to_string();

        let safe_json = if partial {
            strip_partial_json_suffix(current_args_json)
        } else {
            current_args_json
        };

        let streamed_args = self.state.active_streamed_arguments().unwrap_or_default();
        if safe_json.is_empty() || safe_json == streamed_args {
            return Ok(false);
        }

        let diff = if streamed_args.is_empty() {
            safe_json.clone()
        } else {
            let prefix = find_common_prefix(streamed_args, &safe_json);
            if prefix.len() < streamed_args.len() {
                self.state.set_active_streamed_arguments(prefix);
                return Ok(false);
            }
            safe_json[streamed_args.len()..].to_string()
        };

        if diff.is_empty() {
            return Ok(false);
        }

        self.state.set_active_streamed_arguments(safe_json);
        result.calls.push(ToolCallDelta {
            tool_index,
            name: None,
            arguments: diff,
        });
        Ok(true)
    }
}

impl ToolParser for Gemma4ToolParser {
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn adjust_request(&self, request: &mut ChatRequest) -> Result<()> {
        if request.tool_parsing_enabled() {
            // Don't skip special tokens — <|tool_call> etc. are needed.
            request.decode_options.skip_special_tokens = false;
        }
        Ok(())
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.buffer.push_str(chunk);
        let mut result = ToolParseResult::default();

        loop {
            let progressed = if self.state.active_tool_index().is_some() {
                self.process_tool_mode(&mut result)?
            } else {
                self.process_text_mode(&mut result)
            };
            if !progressed {
                break;
            }
        }

        Ok(result)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        let mut result = ToolParseResult::default();

        if let Some(tool_index) = self.state.active_tool_index() {
            if self.state.active_tool_name_sent() {
                match scan_tool_tail(&self.buffer) {
                    ToolTailState::Complete { args_end, .. }
                    | ToolTailState::PendingAfterBrace { args_end } => {
                        let raw_args = self.buffer[..args_end].to_string();
                        self.emit_argument_diff(tool_index, &raw_args, false, &mut result)?;
                    }
                    ToolTailState::Incomplete => {}
                }
            } else {
                result.normal_text.push_str(TOOL_CALL_START);
                result.normal_text.push_str(&self.buffer);
            }
        } else if !self.buffer.is_empty() {
            result.normal_text.push_str(&self.buffer);
        }

        self.buffer.clear();
        self.state.reset();
        Ok(result)
    }
}

enum ToolHeader {
    NeedMore,
    Invalid(String),
    Ready { name: String, consumed_len: usize },
}

enum ToolTailState {
    Incomplete,
    PendingAfterBrace {
        args_end: usize,
    },
    Complete {
        args_end: usize,
        consumed_len: usize,
    },
}

fn parse_tool_header(buffer: &str) -> ToolHeader {
    // Expect "call:name{args...}".
    if CALL_PREFIX.starts_with(buffer) && buffer.len() < CALL_PREFIX.len() {
        return ToolHeader::NeedMore;
    }
    if !buffer.starts_with(CALL_PREFIX) {
        return ToolHeader::Invalid("Gemma4 tool call must start with `call:`".to_string());
    }

    let Some(brace_rel) = buffer[CALL_PREFIX.len()..].find('{') else {
        return ToolHeader::NeedMore;
    };
    let brace_idx = CALL_PREFIX.len() + brace_rel;
    let name = buffer[CALL_PREFIX.len()..brace_idx].trim();
    if name.is_empty() {
        return ToolHeader::Invalid("Gemma4 tool call is missing a function name".to_string());
    }

    ToolHeader::Ready {
        name: name.to_string(),
        consumed_len: brace_idx + 1,
    }
}

fn scan_tool_tail(input: &str) -> ToolTailState {
    let mut i = 0usize;
    let mut object_depth = 0usize;
    let mut array_depth = 0usize;

    while i < input.len() {
        if input[i..].starts_with(STRING_DELIM) {
            let start = i + STRING_DELIM.len();
            let Some(rel_end) = input[start..].find(STRING_DELIM) else {
                return ToolTailState::Incomplete;
            };
            i = start + rel_end + STRING_DELIM.len();
            continue;
        }

        let next = input[i..]
            .chars()
            .next()
            .expect("scan index must stay in bounds");
        match next {
            '{' => object_depth += 1,
            '[' => array_depth += 1,
            '}' => {
                if object_depth == 0 && array_depth == 0 {
                    let after_brace = i + next.len_utf8();
                    if input[after_brace..].starts_with(TOOL_CALL_END) {
                        return ToolTailState::Complete {
                            args_end: i,
                            consumed_len: after_brace + TOOL_CALL_END.len(),
                        };
                    }
                    return ToolTailState::PendingAfterBrace { args_end: i };
                }
                object_depth -= 1;
            }
            ']' => {
                array_depth = array_depth.saturating_sub(1);
            }
            _ => {}
        }
        i += next.len_utf8();
    }

    ToolTailState::Incomplete
}

/// Parse a single Gemma4 value (after key:) into a JSON value.
fn parse_gemma4_value(value: &str) -> Result<Value> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(Value::String(String::new()));
    }
    // Boolean
    if value == "true" {
        return Ok(Value::Bool(true));
    }
    if value == "false" {
        return Ok(Value::Bool(false));
    }
    // Null
    if matches!(value, "null" | "none" | "nil" | "NULL" | "None" | "NIL") {
        return Ok(Value::Null);
    }
    // Number (int or float)
    if value.contains('.') {
        if let Ok(parsed) = value.parse::<f64>() {
            let Some(number) = Number::from_f64(parsed) else {
                return Err(parse_failed("Gemma4 float argument is not finite"));
            };
            return Ok(Value::Number(number));
        }
    } else if let Ok(parsed) = value.parse::<i64>() {
        return Ok(Value::Number(Number::from(parsed)));
    }

    // Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    Ok(Value::String(value.to_string()))
}

/// Parse Gemma4's custom key:value format into a JSON object.
///
/// Format examples:
///
/// ```text
/// location:<|"|>Tokyo<|"|>
/// location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
/// count:42,flag:true
/// nested:{inner_key:<|"|>val<|"|>}
/// items:[<|"|>a<|"|>,<|"|>b<|"|>]
/// ```
///
/// When `partial` is true (streaming), bare values at end of string are
/// omitted because they may be incomplete and type-unstable
/// (e.g. partial boolean parsed as bare string).
fn parse_gemma4_args(args: &str, partial: bool) -> Result<Map<String, Value>> {
    let mut result = Map::new();
    if args.trim().is_empty() {
        return Ok(result);
    }

    let mut i = 0usize;
    while i < args.len() {
        // Skip whitespace and commas
        skip_separators(args, &mut i);
        if i >= args.len() {
            break;
        }

        // Parse key (unquoted, ends at ':')
        let Some(colon_rel) = args[i..].find(':') else {
            break;
        };
        let colon_idx = i + colon_rel;
        let key = args[i..colon_idx].trim();
        i = colon_idx + 1;

        if i >= args.len() {
            if !partial {
                result.insert(key.to_string(), Value::String(String::new()));
            }
            break;
        }

        skip_value_whitespace(args, &mut i);
        if i >= args.len() {
            if !partial {
                result.insert(key.to_string(), Value::String(String::new()));
            }
            break;
        }

        let value = if args[i..].starts_with(STRING_DELIM) {
            // String value: <|"|>...<|"|>
            i += STRING_DELIM.len();
            let value_start = i;
            let Some(end_rel) = args[i..].find(STRING_DELIM) else {
                // Unterminated string — take rest
                result.insert(
                    key.to_string(),
                    Value::String(args[value_start..].to_string()),
                );
                break;
            };
            let end_idx = i + end_rel;
            let parsed = Value::String(args[value_start..end_idx].to_string());
            i = end_idx + STRING_DELIM.len();
            parsed
        } else if args[i..].starts_with('{') {
            // Nested object: {...}
            let value_start = i + 1;
            i += 1;
            let mut depth = 1usize;
            while i < args.len() && depth > 0 {
                if args[i..].starts_with(STRING_DELIM) {
                    // Skip over string contents to avoid counting { inside strings
                    i = skip_over_string_delim(args, i).unwrap_or(args.len());
                    continue;
                }
                let next = args[i..].chars().next().expect("index must stay in bounds");
                match next {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                i += next.len_utf8();
            }
            let inner = if depth > 0 {
                // Incomplete nested object — recurse as partial.
                &args[value_start..i]
            } else {
                &args[value_start..i - 1]
            };
            Value::Object(parse_gemma4_args(inner, depth > 0)?)
        } else if args[i..].starts_with('[') {
            // Array: [...]
            let value_start = i + 1;
            i += 1;
            let mut depth = 1usize;
            while i < args.len() && depth > 0 {
                if args[i..].starts_with(STRING_DELIM) {
                    i = skip_over_string_delim(args, i).unwrap_or(args.len());
                    continue;
                }
                let next = args[i..].chars().next().expect("index must stay in bounds");
                match next {
                    '[' => depth += 1,
                    ']' => depth -= 1,
                    _ => {}
                }
                i += next.len_utf8();
            }
            let inner = if depth > 0 {
                &args[value_start..i]
            } else {
                &args[value_start..i - 1]
            };
            Value::Array(parse_gemma4_array(inner, depth > 0)?)
        } else {
            // Bare value (number, boolean, etc.)
            let value_start = i;
            while i < args.len() {
                let next = args[i..].chars().next().expect("index must stay in bounds");
                if matches!(next, ',' | '}' | ']') {
                    break;
                }
                i += next.len_utf8();
            }
            if partial && i >= args.len() {
                // Value may be incomplete (e.g. partial boolean) —
                // withhold to avoid type instability during streaming.
                break;
            }
            parse_gemma4_value(&args[value_start..i])?
        };

        result.insert(key.to_string(), value);
    }

    Ok(result)
}

/// Parse a Gemma4 array content string into a JSON array.
fn parse_gemma4_array(array: &str, partial: bool) -> Result<Vec<Value>> {
    let mut result = Vec::new();
    let mut i = 0usize;

    while i < array.len() {
        skip_separators(array, &mut i);
        if i >= array.len() {
            break;
        }

        let value = if array[i..].starts_with(STRING_DELIM) {
            // String element
            i += STRING_DELIM.len();
            let Some(end_rel) = array[i..].find(STRING_DELIM) else {
                result.push(Value::String(array[i..].to_string()));
                break;
            };
            let end_idx = i + end_rel;
            let parsed = Value::String(array[i..end_idx].to_string());
            i = end_idx + STRING_DELIM.len();
            parsed
        } else if array[i..].starts_with('{') {
            // Nested object
            let value_start = i + 1;
            i += 1;
            let mut depth = 1usize;
            while i < array.len() && depth > 0 {
                if array[i..].starts_with(STRING_DELIM) {
                    i = skip_over_string_delim(array, i).unwrap_or(array.len());
                    continue;
                }
                let next = array[i..]
                    .chars()
                    .next()
                    .expect("index must stay in bounds");
                match next {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                i += next.len_utf8();
            }
            let inner = if depth > 0 {
                &array[value_start..i]
            } else {
                &array[value_start..i - 1]
            };
            Value::Object(parse_gemma4_args(inner, depth > 0)?)
        } else if array[i..].starts_with('[') {
            // Nested array
            let value_start = i + 1;
            i += 1;
            let mut depth = 1usize;
            while i < array.len() && depth > 0 {
                if array[i..].starts_with(STRING_DELIM) {
                    i = skip_over_string_delim(array, i).unwrap_or(array.len());
                    continue;
                }
                let next = array[i..]
                    .chars()
                    .next()
                    .expect("index must stay in bounds");
                match next {
                    '[' => depth += 1,
                    ']' => depth -= 1,
                    _ => {}
                }
                i += next.len_utf8();
            }
            let inner = if depth > 0 {
                &array[value_start..i]
            } else {
                &array[value_start..i - 1]
            };
            Value::Array(parse_gemma4_array(inner, depth > 0)?)
        } else {
            // Bare value
            let value_start = i;
            while i < array.len() {
                let next = array[i..]
                    .chars()
                    .next()
                    .expect("index must stay in bounds");
                if matches!(next, ',' | ']') {
                    break;
                }
                i += next.len_utf8();
            }
            if partial && i >= array.len() {
                break;
            }
            parse_gemma4_value(&array[value_start..i])?
        };

        result.push(value);
    }

    Ok(result)
}

fn skip_over_string_delim(input: &str, start: usize) -> Option<usize> {
    let value_start = start + STRING_DELIM.len();
    input[value_start..]
        .find(STRING_DELIM)
        .map(|end_rel| value_start + end_rel + STRING_DELIM.len())
}

fn skip_separators(input: &str, index: &mut usize) {
    while *index < input.len() {
        let next = input[*index..]
            .chars()
            .next()
            .expect("index must stay in bounds");
        if !matches!(next, ' ' | ',' | '\n' | '\t') {
            break;
        }
        *index += next.len_utf8();
    }
}

fn skip_value_whitespace(input: &str, index: &mut usize) {
    while *index < input.len() {
        let next = input[*index..]
            .chars()
            .next()
            .expect("index must stay in bounds");
        if !matches!(next, ' ' | '\n' | '\t') {
            break;
        }
        *index += next.len_utf8();
    }
}

fn strip_partial_json_suffix(mut json: String) -> String {
    while matches!(
        json.chars().last(),
        Some('}' | '"' | ']' | '<' | '|' | '\\' | '>')
    ) {
        json.pop();
    }
    json
}

fn partial_prefix_len(buffer: &str, token: &str) -> usize {
    (1..token.len())
        .rev()
        .find(|&len| buffer.ends_with(&token[..len]))
        .unwrap_or(0)
}

fn find_common_prefix(lhs: &str, rhs: &str) -> String {
    lhs.chars()
        .zip(rhs.chars())
        .take_while(|(lhs_char, rhs_char)| lhs_char == rhs_char)
        .map(|(matched, _)| matched)
        .collect()
}

fn parse_failed(message: impl Into<String>) -> ToolParserError {
    tool_parser::errors::ParserError::ParsingFailed(message.into()).into()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{
        Gemma4ToolParser, ToolCallDelta, ToolParseResult, ToolParser, parse_gemma4_args,
        parse_gemma4_array,
    };
    use crate::request::{ChatRequest, ChatTool};

    fn test_tools() -> Vec<ChatTool> {
        vec![
            ChatTool {
                name: "get_weather".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            ChatTool {
                name: "get_time".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            ChatTool {
                name: "write_file".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            ChatTool {
                name: "Edit".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            ChatTool {
                name: "search".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            ChatTool {
                name: "set".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            ChatTool {
                name: "get_status".to_string(),
                description: None,
                parameters: json!({ "type": "object" }),
                strict: None,
            },
            ChatTool {
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
    fn gemma4_adjust_request_keeps_special_tokens() {
        let parser = Gemma4ToolParser::new(&test_tools());
        let mut request = ChatRequest::for_test();
        request.tools = test_tools();
        request.tool_choice = crate::request::ChatToolChoice::Auto;
        request.decode_options.skip_special_tokens = true;

        parser.adjust_request(&mut request).unwrap();
        assert!(!request.decode_options.skip_special_tokens);
    }

    #[test]
    fn gemma4_parse_args_handles_scalars_and_nested_values() {
        let parsed = parse_gemma4_args(
            "name:<|\"|>test<|\"|>,count:42,active:true,score:114.514,nested:{inner:<|\"|>value<|\"|>},items:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]",
            false,
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
    fn gemma4_parse_args_partial_withholds_empty_trailing_value() {
        let parsed = parse_gemma4_args("name:<|\"|>test<|\"|>,flag:", true).unwrap();
        assert_eq!(Value::Object(parsed), json!({ "name": "test" }));
    }

    #[test]
    fn gemma4_parse_array_handles_bare_values() {
        let parsed = parse_gemma4_array("42,true,114.514", false).unwrap();
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
    fn gemma4_parse_complete_keeps_incomplete_tool_call_as_tool_intent() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let result = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>London")
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert_eq!(first_call(&result).name.as_deref(), Some("get_weather"));
        assert_eq!(first_call(&result).arguments, "{\"location\":\"London");
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
    fn gemma4_finish_flushes_complete_args_without_end_marker() {
        let mut parser = Gemma4ToolParser::new(&test_tools());
        let mut result = ToolParseResult::default();
        for chunk in ["<|tool_call>", "call:get_status{}"] {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        let result = result.coalesce_calls();

        assert_eq!(first_call(&result).name.as_deref(), Some("get_status"));
        assert_eq!(
            serde_json::from_str::<Value>(&first_call(&result).arguments).unwrap(),
            json!({})
        );
    }
}
