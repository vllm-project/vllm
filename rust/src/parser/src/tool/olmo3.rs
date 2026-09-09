// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::{Map, Number, Value};
use thiserror_ext::AsReport as _;

use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::utils::partial_prefix_len;

const START: &str = "<function_calls>";
const END: &str = "</function_calls>";
const MAX_LITERAL_DEPTH: usize = 64;

/// Parser for Olmo 3 newline-separated Python-style function calls.
///
/// Only callable names and JSON-compatible Python literals are accepted;
/// arbitrary Python expressions are never evaluated.
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
        let (name, arguments) = parse_call(source)?;

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

fn parse_call(source: &str) -> Result<(String, String)> {
    let mut parser = PythonCallParser::new(source);
    let function = parser.parse_callable_name()?;
    parser.skip_whitespace();
    parser.expect('(')?;

    let mut arguments = Map::new();
    parser.skip_whitespace();
    if !parser.consume(')') {
        loop {
            let name = parser.parse_identifier()?.to_string();
            parser.skip_whitespace();
            parser.expect('=')?;
            let value = parser.parse_value(0)?;
            arguments.insert(name, value);

            parser.skip_whitespace();
            if parser.consume(')') {
                break;
            }
            parser.expect(',')?;
            parser.skip_whitespace();
            if parser.consume(')') {
                break;
            }
        }
    }
    parser.skip_whitespace();
    if !parser.is_empty() {
        return Err(parsing_failed!(
            "unexpected trailing Olmo3 tool-call content"
        ));
    }

    let arguments = serde_json::to_string(&Value::Object(arguments)).map_err(|error| {
        parsing_failed!("failed to serialize Olmo3 arguments: {}", error.as_report())
    })?;
    Ok((function, arguments))
}

struct PythonCallParser<'a> {
    source: &'a str,
    offset: usize,
}

impl<'a> PythonCallParser<'a> {
    fn new(source: &'a str) -> Self {
        Self { source, offset: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.source[self.offset..]
    }

    fn is_empty(&self) -> bool {
        self.offset == self.source.len()
    }

    fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn next(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.offset += character.len_utf8();
        Some(character)
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: char) -> Result<()> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(parsing_failed!("expected `{expected}` in Olmo3 tool call"))
        }
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.next();
        }
    }

    fn parse_identifier(&mut self) -> Result<&'a str> {
        self.skip_whitespace();
        let start = self.offset;
        let Some(first) = self.peek() else {
            return Err(parsing_failed!("expected an Olmo3 identifier"));
        };
        if first != '_' && !first.is_ascii_alphabetic() {
            return Err(parsing_failed!("expected an Olmo3 identifier"));
        }
        self.next();
        while self
            .peek()
            .is_some_and(|character| character == '_' || character.is_ascii_alphanumeric())
        {
            self.next();
        }
        Ok(&self.source[start..self.offset])
    }

    fn parse_callable_name(&mut self) -> Result<String> {
        let mut name = self.parse_identifier()?.to_string();
        loop {
            self.skip_whitespace();
            if !self.consume('.') {
                return Ok(name);
            }
            name.push('.');
            name.push_str(self.parse_identifier()?);
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value> {
        if depth >= MAX_LITERAL_DEPTH {
            return Err(parsing_failed!(
                "Olmo3 argument nesting exceeds {MAX_LITERAL_DEPTH} levels"
            ));
        }
        self.skip_whitespace();
        match self.peek() {
            Some('\'' | '"') => self.parse_string().map(Value::String),
            Some('[') => self.parse_sequence('[', ']', depth),
            Some('(') => self.parse_tuple(depth),
            Some('{') => self.parse_dict(depth),
            Some('+' | '-' | '.' | '0'..='9') => self.parse_number(),
            Some('_' | 'a'..='z' | 'A'..='Z') => self.parse_named_literal(),
            _ => Err(parsing_failed!("unsupported Olmo3 argument expression")),
        }
    }

    fn parse_named_literal(&mut self) -> Result<Value> {
        match self.parse_identifier()? {
            "None" | "null" => Ok(Value::Null),
            "True" | "true" => Ok(Value::Bool(true)),
            "False" | "false" => Ok(Value::Bool(false)),
            name => Err(parsing_failed!("unsupported Olmo3 name literal `{name}`")),
        }
    }

    fn parse_sequence(&mut self, open: char, close: char, depth: usize) -> Result<Value> {
        self.expect(open)?;
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.consume(close) {
            return Ok(Value::Array(values));
        }
        loop {
            values.push(self.parse_value(depth + 1)?);
            self.skip_whitespace();
            if self.consume(close) {
                return Ok(Value::Array(values));
            }
            self.expect(',')?;
            self.skip_whitespace();
            if self.consume(close) {
                return Ok(Value::Array(values));
            }
        }
    }

    fn parse_tuple(&mut self, depth: usize) -> Result<Value> {
        self.expect('(')?;
        self.skip_whitespace();
        if self.consume(')') {
            return Ok(Value::Array(Vec::new()));
        }

        let first = self.parse_value(depth + 1)?;
        self.skip_whitespace();
        if self.consume(')') {
            return Ok(first);
        }

        self.expect(',')?;
        let mut values = vec![first];
        loop {
            self.skip_whitespace();
            if self.consume(')') {
                return Ok(Value::Array(values));
            }
            values.push(self.parse_value(depth + 1)?);
            self.skip_whitespace();
            if self.consume(')') {
                return Ok(Value::Array(values));
            }
            self.expect(',')?;
        }
    }

    fn parse_dict(&mut self, depth: usize) -> Result<Value> {
        self.expect('{')?;
        let mut values = Map::new();
        self.skip_whitespace();
        if self.consume('}') {
            return Ok(Value::Object(values));
        }
        loop {
            let key = self.parse_value(depth + 1)?;
            let Value::String(key) = key else {
                return Err(parsing_failed!("Olmo3 dictionary keys must be strings"));
            };
            self.skip_whitespace();
            self.expect(':')?;
            values.insert(key, self.parse_value(depth + 1)?);
            self.skip_whitespace();
            if self.consume('}') {
                return Ok(Value::Object(values));
            }
            self.expect(',')?;
            self.skip_whitespace();
            if self.consume('}') {
                return Ok(Value::Object(values));
            }
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        let quote = self.next().ok_or_else(|| parsing_failed!("expected an Olmo3 string"))?;
        let mut output = String::new();
        loop {
            match self.next() {
                Some(character) if character == quote => return Ok(output),
                Some('\\') => self.parse_escape(&mut output)?,
                Some('\n' | '\r') | None => {
                    return Err(parsing_failed!("unterminated Olmo3 string literal"));
                }
                Some(character) => output.push(character),
            }
        }
    }

    fn parse_escape(&mut self, output: &mut String) -> Result<()> {
        let escaped =
            self.next().ok_or_else(|| parsing_failed!("unterminated Olmo3 string escape"))?;
        match escaped {
            '\n' => {}
            '\r' => {
                if self.peek() == Some('\n') {
                    self.next();
                }
            }
            '\\' => output.push('\\'),
            '\'' => output.push('\''),
            '"' => output.push('"'),
            'a' => output.push('\u{7}'),
            'b' => output.push('\u{8}'),
            'f' => output.push('\u{c}'),
            'n' => output.push('\n'),
            'r' => output.push('\r'),
            't' => output.push('\t'),
            'v' => output.push('\u{b}'),
            'x' => output.push(self.parse_unicode_escape(2)?),
            'u' => output.push(self.parse_unicode_escape(4)?),
            'U' => output.push(self.parse_unicode_escape(8)?),
            '0'..='7' => {
                let mut value = escaped.to_digit(8).unwrap_or_default();
                for _ in 0..2 {
                    let Some(character @ '0'..='7') = self.peek() else {
                        break;
                    };
                    self.next();
                    value = value * 8 + character.to_digit(8).unwrap_or_default();
                }
                let character = char::from_u32(value)
                    .ok_or_else(|| parsing_failed!("invalid Olmo3 octal string escape"))?;
                output.push(character);
            }
            other => {
                output.push('\\');
                output.push(other);
            }
        }
        Ok(())
    }

    fn parse_unicode_escape(&mut self, digits: usize) -> Result<char> {
        let mut value = 0;
        for _ in 0..digits {
            let character =
                self.next().ok_or_else(|| parsing_failed!("incomplete Olmo3 Unicode escape"))?;
            value = value * 16
                + character
                    .to_digit(16)
                    .ok_or_else(|| parsing_failed!("invalid Olmo3 Unicode escape"))?;
        }
        char::from_u32(value).ok_or_else(|| parsing_failed!("invalid Olmo3 Unicode scalar"))
    }

    fn parse_number(&mut self) -> Result<Value> {
        let start = self.offset;
        while self.peek().is_some_and(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.' | '_')
        }) {
            self.next();
        }
        number_from_text(&self.source[start..self.offset])
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

fn number_from_text(text: &str) -> Result<Value> {
    let normalized = text.replace('_', "");
    if let Ok(value) = normalized.parse::<i64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = normalized.parse::<u64>() {
        return Ok(Value::Number(value.into()));
    }
    if normalized.contains(['.', 'e', 'E'])
        && let Ok(value) = normalized.parse::<f64>()
    {
        return Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| parsing_failed!("invalid Olmo3 float literal"));
    }
    Err(parsing_failed!(
        "Olmo3 number literal is outside the supported JSON range: `{text}`"
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
    fn parses_python_literals_and_string_escapes() {
        let mut parser = Olmo3ToolParser::new();
        let output = parser
            .parse_complete(
                r#"<function_calls>configure(enabled=True, missing=None, ratio=-1.25, points=(1, 2), label='Martha\'s \u0056ineyard')</function_calls>"#,
            )
            .unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            output.calls()[0].arguments,
            r#"{"enabled":true,"missing":null,"ratio":-1.25,"points":[1,2],"label":"Martha's Vineyard"}"#
        );
    }

    #[test]
    fn rejects_executable_and_deeply_nested_expressions() {
        for source in [
            "<function_calls>run(value=other())</function_calls>".to_string(),
            format!(
                "<function_calls>run(value={}0{})</function_calls>",
                "[".repeat(65),
                "]".repeat(65)
            ),
        ] {
            let output = Olmo3ToolParser::new().parse_complete(&source).unwrap();
            assert!(output.calls().is_empty());
            assert_eq!(output.normal_text(), source);
        }
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
