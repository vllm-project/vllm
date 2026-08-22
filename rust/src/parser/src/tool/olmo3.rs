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

const START: &str = "<function_calls>";
const END: &str = "</function_calls>";

/// Parser for Olmo 3 newline-separated Python-style function calls.
pub struct Olmo3ToolParser {
    buffer: String,
    mode: Olmo3Mode,
    next_tool_index: usize,
    emitted_tool_call: bool,
}

enum Olmo3Mode {
    Text,
    ToolBlock,
    Done,
}

impl Default for Olmo3ToolParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Olmo3ToolParser {
    /// Create an empty request-scoped parser.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            mode: Olmo3Mode::Text,
            next_tool_index: 0,
            emitted_tool_call: false,
        }
    }

    fn reset(&mut self) -> String {
        let buffered = match self.mode {
            Olmo3Mode::ToolBlock if !self.emitted_tool_call => {
                format!("{START}{}", self.buffer)
            }
            Olmo3Mode::ToolBlock => std::mem::take(&mut self.buffer),
            Olmo3Mode::Text | Olmo3Mode::Done => std::mem::take(&mut self.buffer),
        };
        self.buffer.clear();
        self.mode = Olmo3Mode::Text;
        self.next_tool_index = 0;
        self.emitted_tool_call = false;
        buffered
    }

    fn emit_complete_call(&mut self, output: &mut ToolParserOutput) -> Result<bool> {
        let Some(call_end) = complete_call_end(&self.buffer) else {
            return Ok(false);
        };
        let source = self.buffer[..call_end].trim();
        let module = parse(source, Mode::Expression, "<olmo3_tool_call>").map_err(|error| {
            parsing_failed!("invalid Olmo3 tool-call expression: {}", error.as_report())
        })?;
        let Mod::Expression(expression) = module else {
            return Err(parsing_failed!("expected an Olmo3 expression"));
        };
        let (name, arguments) = parse_call(*expression.body)?;

        output.push_call(ToolCallDelta {
            tool_index: self.next_tool_index,
            name: Some(name),
            arguments,
        });
        self.next_tool_index += 1;
        self.emitted_tool_call = true;
        self.buffer.drain(..call_end);
        Ok(true)
    }
}

impl ToolParser for Olmo3ToolParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new()))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);

        loop {
            match self.mode {
                Olmo3Mode::Text => {
                    if let Some(marker_start) = self.buffer.find(START) {
                        output.push_text(self.buffer[..marker_start].to_string());
                        self.buffer.drain(..marker_start + START.len());
                        self.mode = Olmo3Mode::ToolBlock;
                        continue;
                    }

                    let keep_len = partial_prefix_len(&self.buffer, START);
                    let emit_len = self.buffer.len() - keep_len;
                    if emit_len > 0 {
                        output.push_text(self.buffer.drain(..emit_len).collect::<String>());
                    }
                    return Ok(());
                }
                Olmo3Mode::ToolBlock => {
                    match self.emit_complete_call(output) {
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
                    self.mode = Olmo3Mode::Done;
                }
                Olmo3Mode::Done => {
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
        Olmo3ToolParser::reset(self)
    }
}

fn parse_call(expr: Expr) -> Result<(String, String)> {
    let Expr::Call(call) = expr else {
        return Err(parsing_failed!("expected an Olmo3 function call"));
    };
    if !call.args.is_empty() {
        return Err(parsing_failed!(
            "Olmo3 tool calls do not support positional arguments"
        ));
    }
    let function = callable_name(&call.func)?;

    let mut arguments = Map::with_capacity(call.keywords.len());
    for keyword in call.keywords {
        let Some(name) = keyword.arg else {
            return Err(parsing_failed!("Olmo3 tool calls do not support **kwargs"));
        };
        arguments.insert(name.to_string(), expression_to_json(&keyword.value)?);
    }
    let arguments = serde_json::to_string(&Value::Object(arguments)).map_err(|error| {
        parsing_failed!("failed to serialize Olmo3 arguments: {}", error.as_report())
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
            _ => return Err(parsing_failed!("unsupported Olmo3 function reference")),
        }
    }
}

fn complete_call_end(source: &str) -> Option<usize> {
    let mut brackets = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut saw_call = false;

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
            '(' => {
                brackets.push(character);
                saw_call = true;
            }
            '[' | '{' => brackets.push(character),
            ')' => {
                if brackets.pop() != Some('(') {
                    return None;
                }
                if saw_call && brackets.is_empty() {
                    return Some(index + character.len_utf8());
                }
            }
            ']' if brackets.pop() != Some('[') => return None,
            '}' if brackets.pop() != Some('{') => return None,
            ']' | '}' => {}
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
        Expr::Dict(dict) => {
            let mut output = Map::with_capacity(dict.keys.len());
            for (key, value) in dict.keys.iter().zip(&dict.values) {
                let Some(key) = key else {
                    return Err(parsing_failed!(
                        "Olmo3 tool calls do not support dict unpacking"
                    ));
                };
                let Value::String(key) = expression_to_json(key)? else {
                    return Err(parsing_failed!("Olmo3 dictionary keys must be strings"));
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
                "unsupported Olmo3 name literal `{}`",
                name.id
            )),
        },
        Expr::UnaryOp(unary) => match (unary.op, unary.operand.as_ref()) {
            (rustpython_parser::ast::UnaryOp::USub, Expr::Constant(constant)) => {
                match &constant.value {
                    Constant::Int(value) => number_from_text(&format!("-{value}")),
                    Constant::Float(value) => Number::from_f64(-value)
                        .map(Value::Number)
                        .ok_or_else(|| parsing_failed!("invalid Olmo3 float literal")),
                    _ => Err(parsing_failed!("unsupported Olmo3 unary operand")),
                }
            }
            (rustpython_parser::ast::UnaryOp::UAdd, Expr::Constant(constant)) => {
                constant_to_json(&constant.value)
            }
            _ => Err(parsing_failed!("unsupported Olmo3 unary expression")),
        },
        _ => Err(parsing_failed!("unsupported Olmo3 argument expression")),
    }
}

fn constant_to_json(constant: &Constant) -> Result<Value> {
    match constant {
        Constant::None => Ok(Value::Null),
        Constant::Bool(value) => Ok(Value::Bool(*value)),
        Constant::Int(value) => number_from_text(&value.to_string()),
        Constant::Float(value) => Number::from_f64(*value)
            .map(Value::Number)
            .ok_or_else(|| parsing_failed!("invalid Olmo3 float literal")),
        Constant::Str(value) => Ok(Value::String(value.clone())),
        Constant::Tuple(values) => values
            .iter()
            .map(constant_to_json)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        _ => Err(parsing_failed!("unsupported Olmo3 literal")),
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
        "Olmo3 integer literal is outside the supported JSON range: `{text}`"
    ))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::Olmo3ToolParser;
    use crate::tool::{ToolParser, ToolParserTestExt as _};

    #[test]
    fn parses_python_and_json_literals() {
        let mut parser = Olmo3ToolParser::new();
        let output = parser
            .parse_complete("<function_calls>register_user(name='John Doe', age=37, address={'city': 'San Francisco'}, role=null, passed_test=true, aliases=['John', 'Johnny'])</function_calls>")
            .unwrap();
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "register_user",
                            ),
                            arguments: "{\"name\":\"John Doe\",\"age\":37,\"address\":{\"city\":\"San Francisco\"},\"role\":null,\"passed_test\":true,\"aliases\":[\"John\",\"Johnny\"]}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn parses_parallel_calls_across_chunks() {
        let mut parser = Olmo3ToolParser::new();
        let mut output = Default::default();
        parser.parse_into("I will check. <function_", &mut output).unwrap();
        assert_eq!(output.normal_text(), "I will check. ");
        parser
            .parse_into(
                "calls>get_weather(city='San Francisco')\n\
                 do_something(steps=[])</function_calls> Done.",
                &mut output,
            )
            .unwrap();
        assert_eq!(output.normal_text(), "I will check.  Done.");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[1].arguments, "{\"steps\":[]}");
    }

    #[test]
    fn streams_complete_call_before_wrapper_end() {
        let mut parser = Olmo3ToolParser::new();
        let mut output = Default::default();
        parser
            .parse_into(
                "<function_calls>get_weather(city='San Francisco')",
                &mut output,
            )
            .unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, "{\"city\":\"San Francisco\"}");

        parser.parse_into("</function_calls>", &mut output).unwrap();
        assert_eq!(output.calls().len(), 1);
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn streams_each_parallel_call_when_complete() {
        let mut parser = Olmo3ToolParser::new();
        let mut output = Default::default();
        parser
            .parse_into("<function_calls>first(value=1)\nsecond(", &mut output)
            .unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("first"));

        parser.parse_into("value=2)</function_calls>", &mut output).unwrap();
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[1].name.as_deref(), Some("second"));
    }

    #[test]
    fn preserves_dotted_function_names() {
        let mut parser = Olmo3ToolParser::new();
        let output = parser
            .parse_complete("<function_calls>weather.client.get(city='Paris')</function_calls>")
            .unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            output.calls()[0].name.as_deref(),
            Some("weather.client.get")
        );
    }

    #[test]
    fn rejects_integer_outside_json_range_without_rounding() {
        let source = "<function_calls>lookup(id=18446744073709551617)</function_calls>";
        let mut parser = Olmo3ToolParser::new();
        let output = parser.parse_complete(source).unwrap();
        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), source);
    }

    #[test]
    fn passes_through_non_tool_text() {
        let mut parser = Olmo3ToolParser::new();
        let output = parser.parse_complete("How can I help?").unwrap();
        assert_eq!(output.normal_text(), "How can I help?");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn complete_call_without_wrapper_end_remains_a_tool_call() {
        let mut parser = Olmo3ToolParser::new();
        let output = parser.parse_complete("<function_calls>get_weather(city='Paris')").unwrap();
        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn invalid_wrapper_falls_back_to_text() {
        let mut parser = Olmo3ToolParser::new();
        let output = parser
            .parse_complete("Before <function_calls>not a call</function_calls> after")
            .unwrap();
        assert_eq!(
            output.normal_text(),
            "Before <function_calls>not a call</function_calls> after"
        );
        assert!(output.calls().is_empty());
    }
}
