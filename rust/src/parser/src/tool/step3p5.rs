// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use winnow::combinator::{alt, seq};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::literal;

use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len, take_until_marker};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

type Input<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Text,
    ToolCall { end_scan: MarkerScanState },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Event {
    Text(usize),
    ToolCallStart,
    ToolCallBody(String),
}

/// Tool parser for Step3.5 XML tool calls.
pub struct Step3p5ToolParser {
    buffer: String,
    mode: Mode,
    emitted_tool_count: usize,
    tool_schemas: ToolSchemas,
}

impl Step3p5ToolParser {
    fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: Mode::Text,
            emitted_tool_count: 0,
            tool_schemas: ToolSchemas::from_tools(tools),
        }
    }

    fn apply_event(&mut self, event: Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            Event::Text(len) => output.push_text(&self.buffer[..len]),
            Event::ToolCallStart => {
                self.mode = Mode::ToolCall {
                    end_scan: MarkerScanState::default(),
                };
            }
            Event::ToolCallBody(body) => {
                let (name, params) = parse_tool_call_body(&body)?;
                let arguments = self.tool_schemas.convert_params_with_schema(&name, params);
                let arguments = serde_json::to_string(&arguments)
                    .map_err(|error| parsing_failed!("failed to serialize arguments: {}", error))?;
                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: Some(name),
                    arguments,
                });
                self.emitted_tool_count += 1;
                self.mode = Mode::Text;
            }
        }
        Ok(())
    }
}

impl ToolParser for Step3p5ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(chunk);
        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        if !self.buffer.is_empty() {
            match self.mode {
                Mode::Text => output.push_text(&self.buffer),
                Mode::ToolCall { .. } => {
                    return Err(parsing_failed!("incomplete Step3p5 tool call"));
                }
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

fn parse_next_event(input: &mut Input<'_>, mode: &mut Mode) -> ModalResult<Event> {
    match mode {
        Mode::Text => alt((
            literal(TOOL_CALL_START).value(Event::ToolCallStart),
            safe_text_event,
        ))
        .parse_next(input),
        Mode::ToolCall { end_scan } => seq!(
            take_until_marker(TOOL_CALL_END, end_scan).map(str::to_string),
            _: literal(TOOL_CALL_END),
        )
        .map(|(body,)| Event::ToolCallBody(body))
        .parse_next(input),
    }
}

fn safe_text_event(input: &mut Input<'_>) -> ModalResult<Event> {
    safe_text_len(input, TOOL_CALL_START).map(Event::Text)
}

fn parse_tool_call_body(body: &str) -> Result<(String, Vec<(String, String)>)> {
    let mut remaining = body.trim_start();
    remaining = remaining
        .strip_prefix(FUNCTION_START)
        .ok_or_else(|| parsing_failed!("expected `{}`", FUNCTION_START))?;
    let name_end = remaining
        .find('>')
        .ok_or_else(|| parsing_failed!("incomplete Step3p5 function tag"))?;
    let name = remaining[..name_end].trim();
    if name.is_empty() {
        return Err(parsing_failed!("empty Step3p5 function name"));
    }
    remaining = &remaining[name_end + 1..];

    let mut params = Vec::new();
    loop {
        remaining = remaining.trim_start();
        if let Some(rest) = remaining.strip_prefix(FUNCTION_END) {
            if !rest.trim().is_empty() {
                return Err(parsing_failed!("unexpected text after Step3p5 function"));
            }
            return Ok((name.to_string(), params));
        }

        remaining = remaining
            .strip_prefix(PARAMETER_START)
            .ok_or_else(|| parsing_failed!("expected Step3p5 parameter or function end"))?;
        let name_end = remaining
            .find('>')
            .ok_or_else(|| parsing_failed!("incomplete Step3p5 parameter tag"))?;
        let parameter_name = remaining[..name_end].trim();
        if parameter_name.is_empty() {
            return Err(parsing_failed!("empty Step3p5 parameter name"));
        }
        remaining = &remaining[name_end + 1..];

        let value_end = [
            remaining.find(PARAMETER_END),
            remaining.find(PARAMETER_START),
            remaining.find(FUNCTION_END),
        ]
        .into_iter()
        .flatten()
        .min()
        .ok_or_else(|| parsing_failed!("incomplete Step3p5 parameter value"))?;
        params.push((
            parameter_name.to_string(),
            remaining[..value_end].trim().to_string(),
        ));
        remaining = &remaining[value_end..];
        if remaining.starts_with(PARAMETER_END) {
            remaining = &remaining[PARAMETER_END.len()..];
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::Step3p5ToolParser;
    use crate::tool::test_utils::{split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserEvent, ToolParserOutput, ToolParserTestExt as _};

    fn tool_call(name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(key, value)| format!("<parameter={key}>\n{value}\n</parameter>"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>")
    }

    #[test]
    fn preserves_text_without_tool_calls() {
        let mut parser = Step3p5ToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();
        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn converts_parameters_using_tool_schema() {
        let mut parser = Step3p5ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&tool_call(
                "convert",
                &[
                    ("whole", "1.5"),
                    ("flag", "true"),
                    ("payload", r#"{"city":"Paris"}"#),
                    ("items", "[1,2,3]"),
                ],
            ))
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "whole": 1.5,
                "flag": true,
                "payload": {"city": "Paris"},
                "items": [1, 2, 3],
            })
        );
    }

    #[test]
    fn streams_split_markers_and_mixed_content() {
        let input = format!(
            "hello{}middle{}done",
            tool_call("get_weather", &[("city", "Shanghai"), ("days", "3")]),
            tool_call("add", &[("x", "1"), ("y", "2")]),
        );
        let chunks = split_by_chars(&input, 3);
        let mut parser = Step3p5ToolParser::new(&test_tools());
        let mut streamed = ToolParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut streamed).unwrap();
        }
        streamed.append(parser.finish().unwrap());

        assert!(matches!(
            streamed.events.as_slice(),
            [
                ToolParserEvent::Text(prefix),
                ToolParserEvent::ToolCall(first),
                ToolParserEvent::Text(middle),
                ToolParserEvent::ToolCall(second),
                ToolParserEvent::Text(suffix),
            ] if prefix == "hello"
                && first.tool_index == 0
                && middle == "middle"
                && second.tool_index == 1
                && suffix == "done"
        ));

        let output = streamed.coalesce();
        assert_eq!(output.normal_text(), "hellomiddledone");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[1].name.as_deref(), Some("add"));
    }

    #[test]
    fn tolerates_missing_parameter_end() {
        let input = "<tool_call><function=get_weather><parameter=city>Dallas\n\
                     <parameter=state>TX</parameter></function></tool_call>";
        let mut parser = Step3p5ToolParser::new(&test_tools());
        let output = parser.parse_complete(input).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Dallas", "state": "TX"})
        );
    }

    #[test]
    fn rejects_incomplete_tool_call_at_finish() {
        let mut parser = Step3p5ToolParser::new(&test_tools());
        parser.parse_chunk("<tool_call><function=get_weather>").unwrap();
        assert_eq!(
            parser.finish().unwrap_err().as_report().to_string(),
            "tool parser parsing failed: incomplete Step3p5 tool call"
        );
    }
}
