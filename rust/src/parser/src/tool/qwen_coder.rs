// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, preceded, seq};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, take_until};

use super::parameters::ToolSchemas;
use super::utils::{parse_buffered_event, safe_text_len_mul};
use super::{
    Result, ToolCallDelta, ToolParser, ToolParserError, ToolParserEvent, ToolParserOutput,
};
use crate::tool::{StructuralTagBuilder, Tool};

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

/// Model-specific configuration for the shared Qwen Coder grammar.
///
/// Models may vary the tool-call wrapper and parameter newline framing;
/// the inner `<function=...>` / `<parameter=...>` tags are byte-identical.
/// Seed-OSS, for example, wraps the same body in
/// `<seed:tool_call>` / `</seed:tool_call>`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenCoderConfig {
    /// Human-readable parser name used in error messages.
    pub(crate) parser_name: &'static str,
    /// Marker that opens a tool-call block.
    pub(crate) tool_call_start: &'static str,
    pub(crate) first_tool_call_start: &'static str,
    pub(crate) next_tool_call_start: &'static str,
    /// Marker that closes a tool-call block.
    pub(crate) tool_call_end: &'static str,
    /// Consume one template-inserted newline at each parameter value boundary.
    pub(crate) trim_parameter_newlines: bool,
}

const QWEN_CODER_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "Qwen Coder",
    tool_call_start: TOOL_CALL_START,
    first_tool_call_start: "\n\n<tool_call>",
    next_tool_call_start: "\n<tool_call>",
    tool_call_end: TOOL_CALL_END,
    trim_parameter_newlines: true,
};

type QwenCoderInput<'i> = Partial<&'i str>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum QwenCoderMode {
    Text,
    FunctionName,
    Parameter,
    StringValue {
        at_start: bool,
        pending_newline: bool,
    },
    BufferedValue {
        name: String,
        text: String,
    },
    ToolCallEnd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum QwenCoderEvent {
    Text { len: usize },
    ToolCallStart,
    FunctionName(String),
    ParameterName(String),
    ValueText { len: usize },
    ParameterEnd,
    FunctionEnd,
    ToolCallEnd,
}

/// Tool parser for Qwen Coder XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=location>杭州</parameter>
/// </function>
/// </tool_call>
/// ```
///
/// String arguments stream incrementally; other values wait for their closing tag.
///
/// Note: parallel calls are represented as repeated
/// `<tool_call>...</tool_call>` blocks, not as multiple calls inside one tag.
pub struct Qwen3CoderToolParser {
    buffer: String,
    mode: QwenCoderMode,
    emitted_tool_count: usize,
    function_name: String,
    parameter_count: usize,
    streamed_arguments: String,
    tool_parameters: ToolSchemas,
    config: QwenCoderConfig,
}

impl Qwen3CoderToolParser {
    /// Create a Qwen Coder tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self::with_config(tools, QWEN_CODER_CONFIG)
    }

    /// Create a parser for a model that reuses the Qwen Coder grammar with a
    /// different tool-call wrapper (e.g. Seed-OSS).
    pub(crate) fn with_config(tools: &[Tool], config: QwenCoderConfig) -> Self {
        Self {
            buffer: String::new(),
            mode: QwenCoderMode::Text,
            emitted_tool_count: 0,
            function_name: String::new(),
            parameter_count: 0,
            streamed_arguments: String::new(),
            tool_parameters: ToolSchemas::from_tools(tools),
            config,
        }
    }

    /// Apply one parsed Qwen Coder event to parser state and output.
    fn apply_event(&mut self, event: QwenCoderEvent, output: &mut ToolParserOutput) -> Result<()> {
        let tool_index = self.emitted_tool_count;
        let call_finished = matches!(event, QwenCoderEvent::ToolCallEnd);
        let mut name = None;
        let mut arguments = String::new();
        match event {
            QwenCoderEvent::Text { len } => output.push_text(&self.buffer[..len]),
            QwenCoderEvent::ToolCallStart => self.mode = QwenCoderMode::FunctionName,
            QwenCoderEvent::FunctionName(function_name) => {
                self.function_name.clone_from(&function_name);
                self.streamed_arguments.clear();
                name = Some(function_name);
                arguments.push('{');
                self.parameter_count = 0;
                self.mode = QwenCoderMode::Parameter;
            }
            QwenCoderEvent::ParameterName(parameter_name) => {
                if self.parameter_count > 0 {
                    arguments.push(',');
                }
                arguments.push_str(&serialize_argument(&parameter_name)?);
                arguments.push(':');
                let is_string =
                    self.tool_parameters.is_string_param(&self.function_name, &parameter_name);
                self.mode = if is_string {
                    arguments.push('"');
                    QwenCoderMode::StringValue {
                        at_start: true,
                        pending_newline: false,
                    }
                } else {
                    QwenCoderMode::BufferedValue {
                        name: parameter_name,
                        text: String::new(),
                    }
                };
                self.parameter_count += 1;
            }
            QwenCoderEvent::ValueText { len } => match &mut self.mode {
                QwenCoderMode::BufferedValue { text, .. } => {
                    text.push_str(&self.buffer[..len]);
                }
                QwenCoderMode::StringValue {
                    at_start,
                    pending_newline,
                } => {
                    let mut text = self.buffer[..len].to_owned();
                    if self.config.trim_parameter_newlines {
                        if *at_start && text.starts_with('\n') {
                            text.remove(0);
                        }
                        if *pending_newline {
                            text.insert(0, '\n');
                        }
                        *pending_newline = text.ends_with('\n');
                        if *pending_newline {
                            text.pop();
                        }
                    }
                    *at_start = false;
                    if !text.is_empty() {
                        let encoded = serialize_argument(&text)?;
                        arguments.push_str(&encoded[1..encoded.len() - 1]);
                    }
                }
                _ => unreachable!("value text requires an open parameter"),
            },
            QwenCoderEvent::ParameterEnd => {
                match &mut self.mode {
                    QwenCoderMode::BufferedValue { name, text } => {
                        let text = std::mem::take(text);
                        let text = if self.config.trim_parameter_newlines {
                            trim_one_wrapping_newline(&text).to_owned()
                        } else {
                            text
                        };
                        let value = self.tool_parameters.convert_param_with_schema(
                            &self.function_name,
                            name,
                            text,
                        );
                        arguments = serialize_argument(&value)?;
                    }
                    QwenCoderMode::StringValue { .. } => arguments.push('"'),
                    _ => unreachable!("parameter end requires an open parameter"),
                }
                self.mode = QwenCoderMode::Parameter;
            }
            QwenCoderEvent::FunctionEnd => self.mode = QwenCoderMode::ToolCallEnd,
            QwenCoderEvent::ToolCallEnd => {
                arguments.push('}');
                self.mode = QwenCoderMode::Text;
                self.emitted_tool_count += 1;
            }
        }
        if !arguments.is_empty() {
            self.streamed_arguments.push_str(&arguments);
            output.push_call(ToolCallDelta {
                tool_index,
                name,
                arguments,
            });
        }
        if call_finished {
            let arguments: serde_json::Value = serde_json::from_str(&self.streamed_arguments)
                .map_err(|error| parsing_failed!("invalid completed arguments: {}", error))?;
            output.events.push(ToolParserEvent::ToolCallEnd {
                tool_index,
                arguments: serialize_argument(&arguments)?,
            });
            self.streamed_arguments.clear();
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        let mut remaining = match std::mem::replace(&mut self.mode, QwenCoderMode::Text) {
            QwenCoderMode::BufferedValue { text, .. } => text,
            QwenCoderMode::StringValue {
                pending_newline: true,
                ..
            } => "\n".to_owned(),
            _ => String::new(),
        };
        remaining.push_str(&std::mem::take(&mut self.buffer));
        self.emitted_tool_count = 0;
        self.function_name.clear();
        self.parameter_count = 0;
        self.streamed_arguments.clear();
        remaining
    }
}

impl ToolParser for Qwen3CoderToolParser {
    fn create(tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn reports_tool_call_end(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Qwen3Coder.builder())
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(&chunk.text);
        let config = self.config;

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_qwen_coder_event(input, &self.mode, config, self.emitted_tool_count > 0)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        if self.mode != QwenCoderMode::Text || self.buffer.starts_with(self.config.tool_call_start)
        {
            return Err(ToolParserError::IncompleteToolCall {
                parser: self.config.parser_name,
            });
        }
        output.push_text(&self.buffer);
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        Qwen3CoderToolParser::reset(self)
    }
}

/// Parse a Qwen Coder event for the current parser mode.
fn parse_next_qwen_coder_event(
    input: &mut QwenCoderInput<'_>,
    mode: &QwenCoderMode,
    config: QwenCoderConfig,
    after_tool_call: bool,
) -> ModalResult<QwenCoderEvent> {
    match mode {
        QwenCoderMode::Text => parse_text_event(input, config, after_tool_call),
        QwenCoderMode::FunctionName => preceded(ws0, |input: &mut QwenCoderInput<'_>| {
            tag_name(input, FUNCTION_START).map(QwenCoderEvent::FunctionName)
        })
        .parse_next(input),
        QwenCoderMode::Parameter => preceded(
            ws0,
            alt((
                literal(FUNCTION_END).value(QwenCoderEvent::FunctionEnd),
                |input: &mut QwenCoderInput<'_>| {
                    tag_name(input, PARAMETER_START).map(QwenCoderEvent::ParameterName)
                },
            )),
        )
        .parse_next(input),
        QwenCoderMode::StringValue { .. } | QwenCoderMode::BufferedValue { .. } => alt((
            literal(PARAMETER_END).value(QwenCoderEvent::ParameterEnd),
            |input: &mut QwenCoderInput<'_>| {
                safe_text_len_mul(input, &[PARAMETER_END, config.tool_call_end])
                    .map(|len| QwenCoderEvent::ValueText { len })
            },
        ))
        .parse_next(input),
        QwenCoderMode::ToolCallEnd => preceded(ws0, literal(config.tool_call_end))
            .value(QwenCoderEvent::ToolCallEnd)
            .parse_next(input),
    }
}

/// Parse a text-mode Qwen Coder event.
fn parse_text_event(
    input: &mut QwenCoderInput<'_>,
    config: QwenCoderConfig,
    after_tool_call: bool,
) -> ModalResult<QwenCoderEvent> {
    let framed_start = if after_tool_call {
        config.next_tool_call_start
    } else {
        config.first_tool_call_start
    };
    alt((
        alt((literal(framed_start), literal(config.tool_call_start)))
            .value(QwenCoderEvent::ToolCallStart),
        |input: &mut QwenCoderInput<'_>| {
            safe_text_len_mul(input, &[framed_start, config.tool_call_start])
                .map(|len| QwenCoderEvent::Text { len })
        },
    ))
    .parse_next(input)
}

/// Parse the name in a function or parameter opening tag.
fn tag_name(input: &mut QwenCoderInput<'_>, start: &'static str) -> ModalResult<String> {
    let (name,) = seq!(
        _: literal(start),
        take_until(1.., ">"),
        _: ">",
    )
    .parse_next(input)?;
    Ok(name.to_owned())
}

/// Trim a single leading and trailing newline from a parameter value.
fn trim_one_wrapping_newline(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

fn serialize_argument(value: &impl serde::Serialize) -> Result<String> {
    serde_json::to_string(value)
        .map_err(|error| parsing_failed!("failed to serialize argument: {}", error))
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{Qwen3CoderToolParser, Tool, ToolParser};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::tests::assert_tool_framing_preserves_body_whitespace;
    use crate::tool::{ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<parameter={name}>{value}</parameter>"))
            .collect::<Vec<_>>()
            .join("\n");
        format!("<tool_call>\n<function={function_name}>\n{params}\n</function>\n</tool_call>")
    }

    #[test]
    fn qwen_coder_exposes_structural_tag_builder() {
        let parser = Qwen3CoderToolParser::new(&test_tools());

        assert!(parser.structural_tag_builder().is_some());
    }

    #[test]
    fn qwen_coder_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn qwen_coder_parse_complete_extracts_single_tool_call() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2026-04-29")],
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "location": "SF",
                "date": "2026-04-29"
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_preserves_prefix_text() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = format!(
            "Thinking... {}",
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text(), "Thinking... ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn qwen_coder_parse_complete_converts_schema_types() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "convert",
                &[
                    ("whole", "5.0"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "42"),
                ],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_extracts_empty_arguments() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser.parse_complete(&build_tool_call("get_weather", &[])).unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({})
        );
    }

    #[test]
    fn qwen_coder_parse_complete_handles_upstream_multiline_typed_params() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<tool_call>\n\
                 <function=calculate_area>\n\
                 <parameter=shape>\n\
                 rectangle\n\
                 </parameter>\n\
                 <parameter=dimensions>\n\
                 {\"width\": 10,\n\
                  \"height\": 20}\n\
                 </parameter>\n\
                 <parameter=precision>\n\
                 2\n\
                 </parameter>\n\
                 </function>\n\
                 </tool_call>",
            )
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("calculate_area"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "shape": "rectangle",
                "dimensions": { "width": 10, "height": 20 },
                "precision": 2,
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_handles_nested_json_parameter() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "convert",
                &[(
                    "payload",
                    r#"{"nested":{"value":[1,2,3],"child":{"enabled":true}}}"#,
                )],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "payload": {
                    "nested": {
                        "value": [1, 2, 3],
                        "child": { "enabled": true },
                    },
                },
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_preserves_xml_like_parameter_values() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "process",
                &[
                    (
                        "html_content",
                        r#"<div class="test"><span>Hello</span></div>"#,
                    ),
                    ("xml_snippet", r#"<root><child attr="value"/></root>"#),
                ],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "html_content": r#"<div class="test"><span>Hello</span></div>"#,
                "xml_snippet": r#"<root><child attr="value"/></root>"#,
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_preserves_raw_closing_tag_text_in_parameter_value() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[
                    (
                        "location",
                        "杭州 &lt;/parameter&gt;&lt;/function&gt;&lt;/tool_call&gt;",
                    ),
                    ("date", "2026-05-08"),
                ],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "location": "杭州 &lt;/parameter&gt;&lt;/function&gt;&lt;/tool_call&gt;",
                "date": "2026-05-08",
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_does_not_double_encode_anyof_object() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "update_record",
                &[("data", r#"{"key":"value","count":42}"#)],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "data": { "key": "value", "count": 42 },
            })
        );
    }

    #[test]
    fn qwen_coder_parse_complete_resolves_schema_references() {
        let tools = [Tool {
            name: "plan_trip".to_string(),
            description: None,
            parameters: json!({
                "$defs": {
                    "Place": { "type": "object", "properties": { "city": { "type": "string" } } },
                    "Days": { "type": "integer" }
                },
                "type": "object",
                "properties": {
                    "place": { "$ref": "#/$defs/Place" },
                    "days": { "$ref": "#/$defs/Days" }
                }
            }),
            strict: None,
            defer_loading: None,
        }];
        let mut parser = Qwen3CoderToolParser::new(&tools);
        let output = parser
            .parse_complete(&build_tool_call(
                "plan_trip",
                &[("place", "\n{\"city\": \"Paris\"}\n"), ("days", "\n3\n")],
            ))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "place": { "city": "Paris" }, "days": 3 })
        );
    }

    #[test]
    fn qwen_coder_streaming_extracts_single_tool_call() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "<tool_call>\n",
                "<function=get_weather>\n",
                "<parameter=location>SF</parameter>\n",
                "</function>\n",
                "</tool_call>",
            ],
        );

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn qwen_coder_streaming_preserves_prefix_text() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "Thinking... ",
                "<tool_call>\n",
                "<function=get_weather>\n",
                "<parameter=location>SF</parameter>\n",
                "</function>\n",
                "</tool_call>",
            ],
        );

        assert_eq!(output.normal_text(), "Thinking... ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn qwen_coder_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &["Hello, ", "world!"]);

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn qwen_coder_streaming_extracts_multiple_tool_calls_in_order() {
        let text = format!(
            "{}\n{}",
            build_tool_call("get_weather", &[("location", "SF")]),
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &[&text]);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[1].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn qwen_coder_streaming_preserves_text_between_tool_calls() {
        let text = format!(
            "I'll check two cities.{}Between calls.{}Done.",
            build_tool_call("get_weather", &[("city", "Dallas"), ("state", "TX")]),
            build_tool_call("get_weather", &[("city", "Orlando"), ("state", "FL")])
        );
        let chunks = split_by_chars(&text, 5);
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(
            output.normal_text(),
            "I'll check two cities.Between calls.Done."
        );
        assert_eq!(output.calls().len(), 2);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Dallas", "state": "TX" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "city": "Orlando", "state": "FL" })
        );
    }

    #[test]
    fn qwen_coder_streaming_handles_start_token_split_across_chunks() {
        let text = build_tool_call("get_weather", &[("location", "SF")]);
        let chunks = split_by_chars(&text, 3);
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn qwen_coder_streaming_handles_end_token_split_across_chunks() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_chunk(
                "<tool_call>\n\
                 <function=get_weather>\n\
                 <parameter=location>SF</parameter>\n\
                 </function>\n\
                 </tool",
            )
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(
            output.clone().coalesce().calls()[0].arguments,
            r#"{"location":"SF""#
        );

        let mut output = output;
        output.append(parser.parse_chunk("_call>").unwrap());
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn qwen_coder_streaming_emits_string_before_parameter_end() {
        let long_location = format!("SF-{}", "x".repeat(8192));
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let mut output = parser
            .parse_chunk("<tool_call>\n<function=get_weather>\n<parameter=location>")
            .unwrap();

        for chunk in split_by_chars(&long_location, 31) {
            let delta = parser.parse_chunk(chunk).unwrap();
            assert!(delta.calls().iter().all(|call| call.tool_index == 0));
            let arguments: String =
                delta.calls().iter().map(|call| call.arguments.as_str()).collect();
            assert_eq!(arguments, chunk);
            output.append(delta);
        }
        output.append(parser.parse_chunk("</parameter>\n</function>\n</tool_call>").unwrap());
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": long_location })
        );
    }

    #[test]
    fn qwen_coder_streaming_keeps_incomplete_tool_arguments_open() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_chunk("<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>")
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(
            output.coalesce().calls()[0].arguments,
            r#"{"location":"SF""#
        );
    }

    #[test]
    fn qwen_coder_streaming_waits_for_typed_value_before_conversion() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let mut output =
            parser.parse_chunk("<tool_call><function=get_weather><parameter=days>").unwrap();
        for chunk in ["1", "2"] {
            assert!(parser.parse_chunk(chunk).unwrap().calls().is_empty());
        }
        output.append(parser.parse_chunk("</parameter></function></tool_call>").unwrap());
        output.append(parser.finish().unwrap());
        assert_eq!(
            serde_json::from_str::<Value>(&output.coalesce().calls()[0].arguments).unwrap(),
            json!({"days": 12})
        );
    }

    #[test]
    fn qwen_coder_streaming_preserves_values_at_every_chunk_size() {
        let mut tools = test_tools();
        tools[0].parameters["$defs"] = json!({"text": {"type": "string"}});
        tools[0].parameters["properties"]["location"] = json!({"$ref": "#/$defs/text"});
        tools[0].parameters["properties"]["maybe"] = json!({"type": ["string", "null"]});
        let value = concat!(
            "\n\n杭州",
            r#"\""#,
            "\t\r\n</function><parameter=literal>&lt;tag&gt;\n\n"
        );
        let text = build_tool_call(
            "get_weather",
            &[
                ("location", value),
                ("days", "12"),
                ("maybe", "None"),
                ("unknown", "true"),
            ],
        );
        for size in 1..=text.chars().count() {
            let mut parser = Qwen3CoderToolParser::new(&tools);
            let output = collect_stream(&mut parser, &split_by_chars(&text, size));
            assert_eq!(
                serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
                json!({
                    "location": concat!("\n杭州", r#"\""#, "\t\r\n</function><parameter=literal>&lt;tag&gt;\n"),
                    "days": 12, "maybe": null, "unknown": "true",
                }),
                "chunk size {size}",
            );
        }
    }

    #[test]
    fn qwen_coder_streaming_repeated_parameters_keep_last_value() {
        let text = build_tool_call(
            "get_weather",
            &[
                ("location", "SF"),
                ("days", "3"),
                ("location", "NYC"),
                ("days", "unknown"),
            ],
        );
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        for chunk in split_by_chars(&text, 1) {
            output.append(parser.parse_chunk(chunk).unwrap());
        }
        output.append(parser.finish().unwrap());
        let streamed: String = output.calls().iter().map(|call| call.arguments.as_str()).collect();
        assert_eq!(
            streamed,
            r#"{"location":"SF","days":3,"location":"NYC","days":"unknown"}"#
        );
        let output = output.coalesce();
        assert_eq!(
            output.calls()[0].arguments,
            r#"{"location":"NYC","days":"unknown"}"#
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"location": "NYC", "days": "unknown"}),
        );
    }

    #[test]
    fn qwen_coder_reset_recovers_uncommitted_value_after_malformed_close() {
        for (name, value, remaining) in [("days", "\n12\n", "\n12\n"), ("location", "\nSF\n", "\n")]
        {
            let mut parser = Qwen3CoderToolParser::new(&test_tools());
            parser
                .parse_chunk(&format!(
                    "<tool_call><function=get_weather><parameter={name}>"
                ))
                .unwrap();
            parser.parse_chunk(value).unwrap();
            assert!(parser.parse_chunk("</tool_call>").is_err());
            assert_eq!(parser.reset(), format!("{remaining}</tool_call>"));
            let output = parser.parse_complete(&build_tool_call("get_weather", &[])).unwrap();
            assert_eq!(output.calls()[0].tool_index, 0);
            assert_eq!(output.calls()[0].arguments, "{}");
        }
    }

    #[test]
    fn qwen_coder_finish_fails_incomplete_tool_call() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        parser
            .parse_chunk("<tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>")
            .unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn qwen_coder_malformed_tool_call_fails_fast() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let error = parser.parse_chunk("<tool_call>\n<bad>\n</tool_call>").unwrap_err();

        expect![[r#"tool parser parsing failed: near "\n<bad>\n</tool_call>": "#]]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn qwen_coder_missing_parameter_end_fails_fast_after_function_end() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let error = parser
            .parse_chunk(
                "<tool_call>\n<function=get_weather>\n<parameter=location>SF</function>\n</tool_call>",
            )
            .unwrap_err();

        expect![[r#"tool parser parsing failed: near "</tool_call>": "#]]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn qwen_coder_parse_function_body_trims_one_wrapping_newline() {
        let mut parser = Qwen3CoderToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<tool_call>\n<function=get_weather>\n<parameter=location>\nHangzhou\n</parameter>\n</function>\n</tool_call>",
            )
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "Hangzhou" })
        );
    }

    #[test]
    fn tool_framing_preserves_body_whitespace_across_chunk_boundaries() {
        assert_tool_framing_preserves_body_whitespace::<Qwen3CoderToolParser>(
            "\n\n",
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
        );
    }
}
