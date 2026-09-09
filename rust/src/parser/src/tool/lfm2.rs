// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use rustpython_parser::{
    Mode,
    ast::{Constant, Expr, Mod},
    parse,
};
use serde_json::{Map, Number, Value};
use thiserror_ext::AsReport as _;

use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::utils::partial_prefix_len;

const START: &str = "<|tool_call_start|>";
const END: &str = "<|tool_call_end|>";

/// Parser for LFM2 list-wrapped Python-style function calls.
pub struct Lfm2ToolParser {
    buffer: String,
    mode: Lfm2Mode,
    next_tool_index: usize,
    emitted_tool_call: bool,
}

enum Lfm2Mode {
    Text,
    ToolBlock,
    Done,
}

impl Default for Lfm2ToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Lfm2ToolParser {
    /// Create an empty request-scoped parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            mode: Lfm2Mode::Text,
            next_tool_index: 0,
            emitted_tool_call: false,
        }
    }

    fn reset(&mut self) -> String {
        let buffered = match self.mode {
            Lfm2Mode::ToolBlock if !self.emitted_tool_call => {
                format!("{START}{}", self.buffer)
            }
            Lfm2Mode::ToolBlock => std::mem::take(&mut self.buffer),
            Lfm2Mode::Text | Lfm2Mode::Done => std::mem::take(&mut self.buffer),
        };
        self.buffer.clear();
        self.mode = Lfm2Mode::Text;
        self.next_tool_index = 0;
        self.emitted_tool_call = false;
        buffered
    }

    fn emit_complete_calls(&mut self, output: &mut ToolParserOutput) -> Result<bool> {
        let Some(list_end) = complete_list_end(&self.buffer) else {
            return Ok(false);
        };
        let source = self.buffer[..list_end].trim();
        let normalized = normalize_source(source);
        let module = parse(&normalized, Mode::Expression, "<lfm2_tool_call>").map_err(|error| {
            parsing_failed!("invalid LFM2 tool-call expression: {}", error.as_report())
        })?;
        let Mod::Expression(expression) = module else {
            return Err(parsing_failed!("expected an LFM2 expression"));
        };
        let Expr::List(list) = *expression.body else {
            return Err(parsing_failed!("expected an LFM2 tool-call list"));
        };

        let calls = list.elts.into_iter().map(parse_call).collect::<Result<Vec<_>>>()?;

        for (name, arguments) in calls {
            output.push_call(ToolCallDelta {
                tool_index: self.next_tool_index,
                name: Some(name),
                arguments,
            });
            self.next_tool_index += 1;
        }
        self.emitted_tool_call = true;
        self.buffer.drain(..list_end);
        Ok(true)
    }
}

impl ToolParser for Lfm2ToolParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new()))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        loop {
            match self.mode {
                Lfm2Mode::Text => {
                    if let Some(marker_start) = self.buffer.find(START) {
                        output.push_text(self.buffer[..marker_start].to_string());
                        self.buffer.drain(..marker_start + START.len());
                        self.mode = Lfm2Mode::ToolBlock;
                        continue;
                    }

                    let keep_len = partial_prefix_len(&self.buffer, START);
                    let emit_len = self.buffer.len() - keep_len;
                    if emit_len > 0 {
                        output.push_text(self.buffer.drain(..emit_len).collect::<String>());
                    }
                    return Ok(());
                }
                Lfm2Mode::ToolBlock => {
                    match self.emit_complete_calls(output) {
                        Ok(true) => continue,
                        Ok(false) => {}
                        Err(_) if self.buffer.contains(END) => {}
                        Err(_) => return Ok(()),
                    }

                    let Some(marker_start) = self.buffer.find(END) else {
                        return Ok(());
                    };
                    let body = self.buffer[..marker_start].to_string();
                    self.buffer.drain(..marker_start + END.len());
                    if !body.trim().is_empty() || !self.emitted_tool_call {
                        output.push_text(format!("{START}{body}{END}"));
                    }
                    self.mode = Lfm2Mode::Done;
                }
                Lfm2Mode::Done => {
                    if let Some(last_end) = self.buffer.rfind(END) {
                        self.buffer.drain(..last_end + END.len());
                        output.push_text(std::mem::take(&mut self.buffer));
                        return Ok(());
                    }
                    let trailing = self.buffer.trim_start();
                    if trailing.starts_with('[') || trailing.starts_with('<') {
                        return Ok(());
                    }
                    output.push_text(std::mem::take(&mut self.buffer));
                    return Ok(());
                }
            }
        }
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        output.push_text(self.reset());
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Lfm2ToolParser::reset(self)
    }
}

fn parse_call(expr: Expr) -> Result<(String, String)> {
    let Expr::Call(call) = expr else {
        return Err(parsing_failed!("expected an Lfm2 function call"));
    };
    if !call.args.is_empty() {
        return Err(parsing_failed!(
            "Lfm2 tool calls do not support positional arguments"
        ));
    }
    let function = callable_name(&call.func)?;

    let mut arguments = Map::with_capacity(call.keywords.len());
    for keyword in call.keywords {
        let Some(name) = keyword.arg else {
            return Err(parsing_failed!("Lfm2 tool calls do not support **kwargs"));
        };
        let name = name.as_str();
        let name = name
            .strip_prefix(RESERVED_KW_PREFIX)
            .filter(|name| is_python_keyword(name))
            .unwrap_or(name);
        arguments.insert(name.to_string(), expression_to_json(&keyword.value)?);
    }
    let arguments = serde_json::to_string(&Value::Object(arguments)).map_err(|error| {
        parsing_failed!("failed to serialize Lfm2 arguments: {}", error.as_report())
    })?;
    Ok((function, arguments))
}

fn callable_name(expr: &Expr) -> Result<String> {
    let mut parts = Vec::new();
    let mut current = expr;
    loop {
        match current {
            Expr::Attribute(attribute) => {
                parts.push(attribute.attr.as_str());
                current = &attribute.value;
            }
            Expr::Name(name) => {
                parts.push(name.id.as_str());
                parts.reverse();
                return Ok(parts.join("."));
            }
            _ => return Err(parsing_failed!("unsupported Lfm2 function reference")),
        }
    }
}

const RESERVED_KW_PREFIX: &str = "_lfm2_kw_";

fn normalize_source(source: &str) -> String {
    let chars = source.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(source.len());
    let mut quote = None;
    let mut escaped = false;
    let mut index = 0;

    while index < chars.len() {
        let character = chars[index];
        if let Some(active_quote) = quote {
            if escaped {
                output.push(character);
                escaped = false;
            } else if character == '\\' {
                output.push(character);
                escaped = true;
            } else if character == active_quote {
                output.push(character);
                quote = None;
            } else if character.is_control() {
                push_escaped_control(&mut output, character);
            } else {
                output.push(character);
            }
            index += 1;
            continue;
        }

        if matches!(character, '\'' | '"') {
            quote = Some(character);
            output.push(character);
            index += 1;
            continue;
        }

        if is_identifier_start(character) {
            let start = index;
            index += 1;
            while index < chars.len() && is_identifier_continue(chars[index]) {
                index += 1;
            }
            let identifier = chars[start..index].iter().collect::<String>();
            let mut equals_index = index;
            while equals_index < chars.len() && chars[equals_index].is_whitespace() {
                equals_index += 1;
            }
            if equals_index < chars.len()
                && chars[equals_index] == '='
                && is_python_keyword(&identifier)
            {
                output.push_str(RESERVED_KW_PREFIX);
            }
            output.push_str(&identifier);
            continue;
        }

        if character.is_ascii_digit() && (index == 0 || !is_identifier_continue(chars[index - 1])) {
            let start = index;
            index += 1;
            while index < chars.len() && chars[index].is_ascii_digit() {
                index += 1;
            }
            let next = chars.get(index).copied();
            let digits = &chars[start..index];
            if digits.len() > 1
                && digits[0] == '0'
                && !matches!(
                    next,
                    Some('.' | 'e' | 'E' | 'x' | 'X' | 'o' | 'O' | 'b' | 'B')
                )
            {
                let first_nonzero = digits.iter().position(|digit| *digit != '0');
                match first_nonzero {
                    Some(offset) => output.extend(digits[offset..].iter()),
                    None => output.push('0'),
                }
            } else {
                output.extend(digits.iter());
            }
            continue;
        }

        output.push(character);
        index += 1;
    }

    output
}

fn push_escaped_control(output: &mut String, character: char) {
    match character {
        '\n' => output.push_str("\\n"),
        '\r' => output.push_str("\\r"),
        '\t' => output.push_str("\\t"),
        _ => output.push_str(&format!("\\u{:04x}", character as u32)),
    }
}

fn is_identifier_start(character: char) -> bool {
    character == '_' || character.is_ascii_alphabetic()
}

fn is_identifier_continue(character: char) -> bool {
    character == '_' || character.is_ascii_alphanumeric()
}

fn is_python_keyword(identifier: &str) -> bool {
    matches!(
        identifier,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "case"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "match"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "while"
            | "with"
            | "yield"
    )
}

fn complete_list_end(source: &str) -> Option<usize> {
    let mut brackets = Vec::new();
    let mut quote = None;
    let mut escaped = false;

    for (index, character) in source.char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            continue;
        }

        match character {
            '\'' | '"' => quote = Some(character),
            '(' | '[' | '{' => brackets.push(character),
            ')' if brackets.pop() != Some('(') => return None,
            ')' => {}
            ']' => {
                if brackets.pop() != Some('[') {
                    return None;
                }
                if brackets.is_empty() {
                    return Some(index + character.len_utf8());
                }
            }
            '}' if brackets.pop() != Some('{') => return None,
            '}' => {}
            _ => {}
        }
    }
    None
}

fn expression_to_json(expr: &Expr) -> Result<Value> {
    match expr {
        Expr::Constant(constant) => constant_to_json(&constant.value),
        Expr::List(list) => list
            .elts
            .iter()
            .map(expression_to_json)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Expr::Tuple(tuple) => tuple
            .elts
            .iter()
            .map(expression_to_json)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Expr::Set(set) => set
            .elts
            .iter()
            .map(expression_to_json)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Expr::Dict(dict) => {
            let mut output = Map::with_capacity(dict.keys.len());
            for (key, value) in dict.keys.iter().zip(&dict.values) {
                let Some(key) = key else {
                    return Err(parsing_failed!(
                        "Lfm2 tool calls do not support dict unpacking"
                    ));
                };
                let Value::String(key) = expression_to_json(key)? else {
                    return Err(parsing_failed!("Lfm2 dictionary keys must be strings"));
                };
                output.insert(key, expression_to_json(value)?);
            }
            Ok(Value::Object(output))
        }
        Expr::Name(name) => match name.id.as_str() {
            "null" => Ok(Value::Null),
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            _ => Err(parsing_failed!(
                "unsupported Lfm2 name literal `{}`",
                name.id
            )),
        },
        Expr::UnaryOp(unary) => match (unary.op, unary.operand.as_ref()) {
            (rustpython_parser::ast::UnaryOp::USub, Expr::Constant(constant)) => {
                match &constant.value {
                    Constant::Int(value) => number_from_text(&format!("-{value}")),
                    Constant::Float(value) => Number::from_f64(-value)
                        .map(Value::Number)
                        .ok_or_else(|| parsing_failed!("invalid Lfm2 float literal")),
                    _ => Err(parsing_failed!("unsupported Lfm2 unary operand")),
                }
            }
            (rustpython_parser::ast::UnaryOp::UAdd, Expr::Constant(constant)) => {
                constant_to_json(&constant.value)
            }
            _ => Err(parsing_failed!("unsupported Lfm2 unary expression")),
        },
        _ => Err(parsing_failed!("unsupported Lfm2 argument expression")),
    }
}

fn constant_to_json(constant: &Constant) -> Result<Value> {
    match constant {
        Constant::None => Ok(Value::Null),
        Constant::Bool(value) => Ok(Value::Bool(*value)),
        Constant::Int(value) => number_from_text(&value.to_string()),
        Constant::Float(value) => Number::from_f64(*value)
            .map(Value::Number)
            .ok_or_else(|| parsing_failed!("invalid Lfm2 float literal")),
        Constant::Str(value) => Ok(Value::String(value.clone())),
        Constant::Tuple(values) => values
            .iter()
            .map(constant_to_json)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        _ => Err(parsing_failed!("unsupported Lfm2 literal")),
    }
}

fn number_from_text(text: &str) -> Result<Value> {
    if let Ok(value) = text.parse::<i64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = text.parse::<u64>() {
        return Ok(Value::Number(value.into()));
    }
    Err(parsing_failed!(
        "Lfm2 integer literal is outside the supported JSON range: `{text}`"
    ))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::Lfm2ToolParser;
    use crate::tool::{ToolParser, ToolParserTestExt as _};

    #[test]
    fn parses_list_wrapped_calls_and_literals() {
        let mut parser = Lfm2ToolParser::new();
        let output = parser
            .parse_complete(
                "before <|tool_call_start|>[weather.client.get(city='Paris', units=['c'], flags={'fast'}, active=true, note=null), ping()]<|tool_call_end|> after",
            )
            .unwrap();
        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "before  after",
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "weather.client.get",
                            ),
                            arguments: "{\"city\":\"Paris\",\"units\":[\"c\"],\"flags\":[\"fast\"],\"active\":true,\"note\":null}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "ping",
                            ),
                            arguments: "{}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn streams_across_split_markers_and_list_body() {
        let mut parser = Lfm2ToolParser::new();
        let mut output = Default::default();
        parser.parse_into("hello <|tool_call_", &mut output).unwrap();
        assert_eq!(output.normal_text(), "hello ");
        parser.parse_into("start|>[first(value=1), sec", &mut output).unwrap();
        assert!(output.calls().is_empty());
        parser.parse_into("ond(value=2)]<|tool_call_end|>", &mut output).unwrap();
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[1].name.as_deref(), Some("second"));
    }

    #[test]
    fn recovers_common_lfm2_python_syntax_errors() {
        let source =
            "<|tool_call_start|>[exec(command='cat\nx\0y', month=07, from=1)]<|tool_call_end|>";
        let mut parser = Lfm2ToolParser::new();
        let output = parser.parse_complete(source).unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            output.calls()[0].arguments,
            "{\"command\":\"cat\\nx\\u0000y\",\"month\":7,\"from\":1}"
        );
    }

    #[test]
    fn suppresses_echo_before_trailing_content() {
        let mut parser = Lfm2ToolParser::new();
        let output = parser
            .parse_complete(
                "<|tool_call_start|>[ping()]<|tool_call_end|>\n[ping()]<|tool_call_end|>\nDone.",
            )
            .unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.normal_text(), "\nDone.");
    }

    #[test]
    fn buffers_possible_echo_until_resolved() {
        let mut parser = Lfm2ToolParser::new();
        let mut output = Default::default();
        parser
            .parse_into(
                "<|tool_call_start|>[ping()]<|tool_call_end|>\n[ping()]",
                &mut output,
            )
            .unwrap();
        assert_eq!(output.calls().len(), 1);
        assert!(output.normal_text().is_empty());
        parser.parse_into("<|tool_call_end|>\nDone.", &mut output).unwrap();
        assert_eq!(output.normal_text(), "\nDone.");
    }

    #[test]
    fn empty_list_is_consumed_without_a_tool_call() {
        let mut parser = Lfm2ToolParser::new();
        let output = parser.parse_complete("<|tool_call_start|>[ ]<|tool_call_end|>").unwrap();
        assert!(output.calls().is_empty());
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn invalid_parallel_call_falls_back_atomically() {
        let source = "<|tool_call_start|>[ping(), not_a_call]<|tool_call_end|>";
        let mut parser = Lfm2ToolParser::new();
        let output = parser.parse_complete(source).unwrap();
        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), source);
    }

    #[test]
    fn invalid_wrapper_falls_back_to_text() {
        let source = "<|tool_call_start|>[not a call]<|tool_call_end|>";
        let mut parser = Lfm2ToolParser::new();
        let output = parser.parse_complete(source).unwrap();
        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), source);
    }

    #[test]
    fn preserves_special_tokens() {
        assert!(Lfm2ToolParser::new().preserve_special_tokens());
    }
}
