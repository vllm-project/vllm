// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};
use winnow::ascii::{multispace0 as ws0, multispace1 as ws1};
use winnow::combinator::{alt, delimited, seq};
use winnow::error::{ContextError, ErrMode};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest, take_until};
use xgrammar_structural_tag::format::Format;

use super::parameters::ToolSchemas;
use super::utils::{MarkerScanState, parse_buffered_event, safe_text_len_mul, take_until_marker};
use super::{JsonStyle, OpenToolCall, Result, ToolParser, ToolParserError, ToolParserOutput};
use crate::output_grammar::{self, OutputGrammarContext, OutputGrammarError};
use crate::tool::{StructuralTagBuilder, Tool};

const TOOL_CALL_START: &str = "<minimax:tool_call>";
const FRAMED_TOOL_CALL_START: &str = "\n<minimax:tool_call>";
const TOOL_CALL_END: &str = "</minimax:tool_call>";
const INVOKE_START: &str = "<invoke";
const INVOKE_END: &str = "</invoke>";
const PARAMETER_START: &str = "<parameter";
const PARAMETER_END: &str = "</parameter>";

type MinimaxM2Input<'i> = Partial<&'i str>;

#[derive(Debug)]
enum MinimaxM2Mode {
    Text,
    /// Inside the tool block, between invokes.
    ToolBlock,
    /// Inside an invoke.
    Invoke {
        call: OpenToolCall,
        body: InvokeBody,
    },
    Done,
}

impl MinimaxM2Mode {
    /// The open invoke and the position inside it, for an event inside an
    /// invoke.
    fn invoke_mut(&mut self) -> Result<(&mut OpenToolCall, &mut InvokeBody)> {
        match self {
            MinimaxM2Mode::Invoke { call, body } => Ok((call, body)),
            _ => Err(parsing_failed!("MiniMax M2 invoke event outside an invoke")),
        }
    }
}

/// Where the parser is inside an invoke.
#[derive(Debug)]
enum InvokeBody {
    /// Between parameters.
    Parameters,
    /// Inside a parameter whose text streams as it arrives.
    StringParameter,
    /// Inside a parameter that is converted through the tool's schema once it
    /// closes.
    BufferedParameter {
        name: String,
        value_end_scan: MarkerScanState,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MinimaxM2Event {
    Text {
        len: usize,
    },
    ToolBlockStart,
    InvokeStart {
        name: String,
    },
    ParameterStart {
        name: String,
    },
    /// The first `len` buffered bytes continue a streamed parameter value.
    ValueText {
        len: usize,
    },
    ParameterEnd,
    /// The complete value of a buffered parameter.
    BufferedValue {
        value: String,
    },
    InvokeEnd,
    ToolBlockEnd,
    IgnoredRest,
}

/// Tool parser for MiniMax M2 XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <minimax:tool_call><invoke name="get_weather">
/// <parameter name="city">Seattle</parameter>
/// </invoke></minimax:tool_call>
/// ```
///
/// The invoke name goes out when the invoke opens. A parameter whose schema
/// type is exactly `string` streams as its text arrives; any other parameter
/// is converted through the schema once its `</parameter>` arrives.
pub struct MinimaxM2ToolParser {
    buffer: String,
    mode: MinimaxM2Mode,
    emitted_tool_count: usize,
    tool_parameters: ToolSchemas,
}

impl MinimaxM2ToolParser {
    /// Create a MiniMax M2 tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            mode: MinimaxM2Mode::Text,
            emitted_tool_count: 0,
            tool_parameters: ToolSchemas::from_tools(tools),
        }
    }

    /// Apply one parsed MiniMax M2 event to parser state and output.
    fn apply_event(&mut self, event: MinimaxM2Event, output: &mut ToolParserOutput) -> Result<()> {
        match event {
            MinimaxM2Event::Text { len: consumed_len } => {
                output.push_text(&self.buffer[..consumed_len]);
            }
            MinimaxM2Event::ToolBlockStart => self.mode = MinimaxM2Mode::ToolBlock,
            MinimaxM2Event::InvokeStart { name } => {
                // The template renders JSON values with Python separators.
                let call = OpenToolCall::new(name, self.emitted_tool_count, JsonStyle::Spaced);
                self.emitted_tool_count += 1;
                self.mode = MinimaxM2Mode::Invoke {
                    call,
                    body: InvokeBody::Parameters,
                };
            }
            MinimaxM2Event::ParameterStart { name } => {
                let (call, body) = self.mode.invoke_mut()?;
                *body = if self.tool_parameters.is_verbatim_string(&call.name, &name) {
                    call.arguments.open_string(&name);
                    InvokeBody::StringParameter
                } else {
                    InvokeBody::BufferedParameter {
                        name,
                        value_end_scan: MarkerScanState::default(),
                    }
                };
            }
            MinimaxM2Event::ValueText { len } => {
                let (call, _) = self.mode.invoke_mut()?;
                call.arguments.push(&self.buffer[..len]);
            }
            MinimaxM2Event::ParameterEnd => {
                let (call, body) = self.mode.invoke_mut()?;
                call.arguments.close();
                *body = InvokeBody::Parameters;
            }
            MinimaxM2Event::BufferedValue { value } => {
                let (call, body) = self.mode.invoke_mut()?;
                let InvokeBody::BufferedParameter { name, .. } =
                    std::mem::replace(body, InvokeBody::Parameters)
                else {
                    return Err(parsing_failed!(
                        "MiniMax M2 buffered value outside a buffered parameter"
                    ));
                };
                let value =
                    self.tool_parameters.convert_param_with_schema(&call.name, &name, value);
                call.arguments.write_value(&name, &value);
            }
            MinimaxM2Event::InvokeEnd => {
                let MinimaxM2Mode::Invoke { call, .. } =
                    std::mem::replace(&mut self.mode, MinimaxM2Mode::ToolBlock)
                else {
                    return Err(parsing_failed!("MiniMax M2 invoke end outside an invoke"));
                };
                output.push_call(call.finish());
            }
            MinimaxM2Event::ToolBlockEnd => self.mode = MinimaxM2Mode::Done,
            MinimaxM2Event::IgnoredRest => {}
        }
        if let MinimaxM2Mode::Invoke { call, .. } = &mut self.mode
            && let Some(delta) = call.take_delta()
        {
            output.push_call(delta);
        }
        Ok(())
    }

    fn reset(&mut self) -> String {
        self.mode = MinimaxM2Mode::Text;
        self.emitted_tool_count = 0;
        std::mem::take(&mut self.buffer)
    }
}

impl ToolParser for MinimaxM2ToolParser {
    fn create(tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::Minimax.builder())
    }

    fn build_call_format(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<Format>> {
        let Some(format) =
            output_grammar::visible_format_from_builder(self.structural_tag_builder(), ctx)?
        else {
            return Ok(None);
        };

        // The MiniMax builder's reasoning=false form retains the empty
        // thinking close and separator used by its chat template. Strip that
        // fixed prefix so the initialized reasoning parser can supply the
        // actual reasoning phase.
        let Format::Sequence(sequence) = format else {
            return Err(OutputGrammarError::UnexpectedBuilderFormat { builder: "minimax" });
        };
        let [
            Format::ConstString(empty_think),
            Format::ConstString(separator),
            visible,
        ] = sequence.elements.as_slice()
        else {
            return Err(OutputGrammarError::UnexpectedBuilderFormat { builder: "minimax" });
        };
        if empty_think.value != "\n</think>\n\n" || separator.value != "\n\n" {
            return Err(OutputGrammarError::UnexpectedBuilderFormat { builder: "minimax" });
        }

        Ok(Some(visible.clone()))
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.push_str(&chunk.text);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer, |input| {
            parse_next_minimax_m2_event(input, &mut self.mode)
        })? {
            self.apply_event(event, output)?;
            self.buffer.drain(..consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            MinimaxM2Mode::Text => {
                output.push_text(&self.buffer);
            }
            MinimaxM2Mode::ToolBlock | MinimaxM2Mode::Invoke { .. } => {
                return Err(ToolParserError::IncompleteToolCall {
                    parser: "MiniMax M2",
                });
            }
            MinimaxM2Mode::Done => {}
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        MinimaxM2ToolParser::reset(self)
    }
}

/// Parse a MiniMax M2 event for the current parser mode.
fn parse_next_minimax_m2_event(
    input: &mut MinimaxM2Input<'_>,
    mode: &mut MinimaxM2Mode,
) -> ModalResult<MinimaxM2Event> {
    match mode {
        MinimaxM2Mode::Text => parse_text_event(input),
        MinimaxM2Mode::ToolBlock => parse_tool_block_event(input),
        MinimaxM2Mode::Invoke { body, .. } => match body {
            InvokeBody::Parameters => parse_invoke_event(input),
            InvokeBody::StringParameter => string_value_event(input),
            InvokeBody::BufferedParameter { value_end_scan, .. } => {
                buffered_value_event(input, value_end_scan)
            }
        },
        MinimaxM2Mode::Done => ignored_rest_event(input),
    }
}

/// Parse a text-mode MiniMax M2 event.
fn parse_text_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    alt((tool_block_start_event, safe_text_event)).parse_next(input)
}

/// Parse a MiniMax M2 tool-block start marker.
fn tool_block_start_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    alt((literal(FRAMED_TOOL_CALL_START), literal(TOOL_CALL_START)))
        .value(MinimaxM2Event::ToolBlockStart)
        .parse_next(input)
}

/// Parse a safe text run before the next MiniMax M2 marker.
fn safe_text_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    safe_text_len_mul(input, &[FRAMED_TOOL_CALL_START, TOOL_CALL_START])
        .map(|len| MinimaxM2Event::Text { len })
}

/// Parse one event inside a MiniMax M2 tool block, between invokes.
fn parse_tool_block_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    alt((tool_block_end_event, invoke_start_event)).parse_next(input)
}

/// Parse a MiniMax M2 tool-block end marker.
fn tool_block_end_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    (ws0, literal(TOOL_CALL_END))
        .value(MinimaxM2Event::ToolBlockEnd)
        .parse_next(input)
}

/// Parse a MiniMax M2 invoke start tag.
fn invoke_start_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    let (name,) = seq!(
        _: ws0,
        _: literal(INVOKE_START),
        _: (ws1, literal("name=")),
        partial_attr_value,
        _: literal(">"),
    )
    .parse_next(input)?;

    Ok(MinimaxM2Event::InvokeStart {
        name: name.trim().to_string(),
    })
}

/// Parse one event inside a MiniMax M2 invoke, between parameters: the invoke
/// end or the next parameter's start tag.
fn parse_invoke_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    ws0.void().parse_next(input)?;
    alt((
        literal(INVOKE_END).value(MinimaxM2Event::InvokeEnd),
        parameter_start_event,
    ))
    .parse_next(input)
}

/// Parse a MiniMax M2 parameter start tag.
fn parameter_start_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    let (name,) = seq!(
        _: literal(PARAMETER_START),
        _: (ws1, literal("name=")),
        partial_attr_value,
        _: literal(">"),
    )
    .parse_next(input)?;

    Ok(MinimaxM2Event::ParameterStart {
        name: name.trim().to_string(),
    })
}

/// Parse a streamed value event inside a string parameter.
///
/// The value runs to the first `</parameter>`. An `</invoke>` before it leaves
/// the parameter unclosed, which is malformed.
fn string_value_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    alt((
        literal(PARAMETER_END).value(MinimaxM2Event::ParameterEnd),
        |input: &mut MinimaxM2Input<'_>| {
            safe_text_len_mul(input, &[PARAMETER_END, INVOKE_END])
                .map(|len| MinimaxM2Event::ValueText { len })
        },
    ))
    .parse_next(input)
}

/// Parse the complete value of a buffered parameter, through its
/// `</parameter>`.
fn buffered_value_event(
    input: &mut MinimaxM2Input<'_>,
    value_end_scan: &mut MarkerScanState,
) -> ModalResult<MinimaxM2Event> {
    let (value,) = seq!(
        take_until_marker(PARAMETER_END, value_end_scan),
        _: literal(PARAMETER_END),
    )
    .parse_next(input)?;
    // As for a string parameter, the invoke cannot end inside a value.
    if value.contains(INVOKE_END) {
        return Err(ErrMode::Cut(ContextError::new()));
    }

    Ok(MinimaxM2Event::BufferedValue {
        value: value.to_string(),
    })
}

/// Parse a quoted or unquoted XML attribute value from partial streaming input.
fn partial_attr_value<'i>(input: &mut MinimaxM2Input<'i>) -> ModalResult<&'i str> {
    alt((
        delimited(literal("\""), take_until(1.., "\""), literal("\"")),
        delimited(literal("'"), take_until(1.., "'"), literal("'")),
        take_until(1.., ">"),
    ))
    .parse_next(input)
}

/// Parse ignored rest after the MiniMax M2 tool block ends.
fn ignored_rest_event(input: &mut MinimaxM2Input<'_>) -> ModalResult<MinimaxM2Event> {
    rest.value(MinimaxM2Event::IgnoredRest).parse_next(input)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::{
        MinimaxM2ToolParser, TOOL_CALL_END, TOOL_CALL_START, ToolParser, ToolParserOutput,
    };
    use crate::tool::ToolParserTestExt as _;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::tests::assert_tool_framing_preserves_body_whitespace;

    fn build_tool_block(invokes: &[(&str, Vec<(&str, &str)>)]) -> String {
        let invokes = invokes
            .iter()
            .map(|(function_name, params)| {
                let params = params
                    .iter()
                    .map(|(name, value)| format!(r#"<parameter name="{name}">{value}</parameter>"#))
                    .collect::<Vec<_>>()
                    .join("");
                format!(r#"<invoke name="{function_name}">{params}</invoke>"#)
            })
            .collect::<String>();
        format!("{TOOL_CALL_START}{invokes}{TOOL_CALL_END}")
    }

    #[test]
    fn minimax_m2_parse_complete_without_tool_call_keeps_text() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m2_parse_complete_extracts_single_tool_call() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                vec![("city", "Seattle"), ("days", "5")],
            )]))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle", "days": 5 })
        );
    }

    #[test]
    fn minimax_m2_parse_complete_preserves_prefix_and_ignores_trailing_text() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = format!(
            "Let me check. {} This trailing text is ignored.",
            build_tool_block(&[("get_weather", vec![("city", "Seattle")])])
        );
        let output = parser.parse_complete(&output).unwrap();

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m2_parse_complete_extracts_multiple_invokes() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[
                ("get_weather", vec![("city", "Seattle")]),
                ("get_weather", vec![("city", "NYC")]),
            ]))
            .unwrap();

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "city": "NYC" })
        );
    }

    #[test]
    fn minimax_m2_parse_complete_converts_schema_types() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "convert",
                vec![
                    ("whole", "5.0"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "42"),
                ],
            )]))
            .unwrap();

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
    fn minimax_m2_parse_complete_preserves_raw_entities_in_parameter_value() {
        // The MiniMax-M2 chat template renders string parameter values RAW (no
        // XML escaping), so a value the user wants to be the literal text
        // "Tom &amp; Jerry &lt;3" is emitted verbatim. The parser must preserve
        // it; xml_unescape currently decodes it, corrupting the bytes.
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                vec![("city", "Tom &amp; Jerry &lt;3")],
            )]))
            .unwrap();
        let args: Value = serde_json::from_str(&output.calls()[0].arguments).unwrap();
        assert_eq!(args["city"], json!("Tom &amp; Jerry &lt;3"));
    }

    #[test]
    fn minimax_m2_parse_complete_preserves_raw_closing_tag_text_in_parameter_value() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                vec![
                    (
                        "city",
                        "Seattle &lt;/parameter&gt;&lt;/invoke&gt;&lt;/minimax:tool_call&gt;",
                    ),
                    ("days", "5"),
                ],
            )]))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "city": "Seattle &lt;/parameter&gt;&lt;/invoke&gt;&lt;/minimax:tool_call&gt;",
                "days": 5,
            })
        );
    }

    #[test]
    fn minimax_m2_parse_complete_handles_multiline_parameters() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(
                "<minimax:tool_call>\
                 <invoke name=\"calculate_area\">\
                 <parameter name=\"shape\">\nrectangle\n</parameter>\
                 <parameter name=\"dimensions\">{\"width\":10,\n\"height\":20}</parameter>\
                 <parameter name=\"precision\">2</parameter>\
                 </invoke>\
                 </minimax:tool_call>",
            )
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "shape": "\nrectangle\n",
                "dimensions": { "width": 10, "height": 20 },
                "precision": 2,
            })
        );
    }

    #[test]
    fn minimax_m2_streaming_extracts_single_tool_call() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "<minimax:tool_call>",
                r#"<invoke name="get_weather">"#,
                r#"<parameter name="city">Seattle</parameter>"#,
                "</invoke></minimax:tool_call>",
            ],
        );

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "city": "Seattle" })
        );
    }

    #[test]
    fn minimax_m2_streaming_preserves_prefix_text() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "Let me check. ",
                "<minimax:tool_call>",
                r#"<invoke name="get_weather"><parameter name="city">Seattle</parameter></invoke>"#,
                "</minimax:tool_call>",
            ],
        );

        assert_eq!(output.normal_text(), "Let me check. ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m2_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &["Hello, ", "world!"]);

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn minimax_m2_streaming_handles_marker_split_across_chunks() {
        let text = build_tool_block(&[("get_weather", vec![("city", "Seattle")])]);
        let chunks = split_by_chars(&text, 3);
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn minimax_m2_streaming_extracts_multiple_invokes_in_order() {
        let text = build_tool_block(&[
            ("get_weather", vec![("city", "Seattle")]),
            ("get_weather", vec![("city", "NYC")]),
        ]);
        let chunks = split_by_chars(&text, 7);
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
    }

    #[test]
    fn minimax_m2_streaming_handles_template_whitespace_and_split_parameters() {
        let text = concat!(
            "I will call the tools.\n",
            "<minimax:tool_call>\n",
            "<invoke name=\"get_weather\">\n",
            "<parameter name=\"city\">Seattle</parameter>\n",
            "</invoke>\n",
            "<invoke name=\"get_weather\">\n",
            "<parameter name=\"city\">NYC</parameter>\n",
            "</invoke>\n",
            "</minimax:tool_call>",
        );
        let chunks = split_by_chars(text, 7);
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text(), "I will call the tools.");
        assert_eq!(result.calls().len(), 2);
        assert_eq!(result.calls()[0].tool_index, 0);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls()[1].tool_index, 1);
        assert_eq!(result.calls()[1].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn minimax_m2_streaming_ignores_text_after_tool_block() {
        let text = format!(
            "{} ignored",
            build_tool_block(&[("get_weather", vec![("city", "Seattle")])])
        );
        let chunks = split_by_chars(&text, 5);
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn minimax_m2_streaming_emits_the_name_when_the_invoke_opens() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output =
            parser.parse_chunk(r#"<minimax:tool_call><invoke name="get_weather">"#).unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert!(output.calls()[0].arguments.is_empty());
    }

    #[test]
    fn minimax_m2_streams_string_parameters_before_the_invoke_closes() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();

        // The call and its arguments so far, after each chunk.
        let mut progress = Vec::new();
        for chunk in [
            "<minimax:tool_call>\n<invoke name=\"get_weather\">\n",
            "<parameter name=\"city\">San Fran",
            "cisco \"CA\"</para",
            "meter>\n<parameter name=\"days\">1",
            "2</parameter>\n<parameter name=\"unit\">",
            "celsius</parameter>\n</invoke>\n</minimax:tool_call>",
        ] {
            output.append(parser.parse_chunk(chunk).unwrap());
            progress.push(match output.clone().coalesce().calls().first() {
                Some(call) => format!("{}({})", call.name.as_deref().unwrap(), call.arguments),
                None => "no call".to_string(),
            });
        }

        // `city` and `unit` are strings and stream; `days` is an integer and
        // waits for its `</parameter>`. A partial close tag never leaks.
        expect![[r#"
            get_weather()
            get_weather({"city": "San Fran)
            get_weather({"city": "San Francisco \"CA\")
            get_weather({"city": "San Francisco \"CA\"")
            get_weather({"city": "San Francisco \"CA\"", "days": 12, "unit": ")
            get_weather({"city": "San Francisco \"CA\"", "days": 12, "unit": "celsius"})"#]]
        .assert_eq(&progress.join("\n"));
    }

    #[test]
    fn minimax_m2_streaming_matches_parse_complete_at_every_chunk_size() {
        let text = concat!(
            "Checking.\n<minimax:tool_call>\n",
            "<invoke name=\"calculate_area\">\n",
            "<parameter name=\"shape\">\n  rect\\angle \"wide\" 🍎\n</parameter>\n",
            "<parameter name=\"dimensions\">{\"width\": 10,\n\"height\": 20}</parameter>\n",
            "<parameter name=\"precision\">2</parameter>\n",
            "</invoke>\n",
            "<invoke name='get_weather'><parameter name=city>a <b> </parameter></invoke>\n",
            "<invoke name=\"no_schema\"><parameter name=\"payload\">{\"a\": 1}</parameter>",
            "<parameter name=\"count\">5</parameter></invoke>\n",
            "</minimax:tool_call>",
        );
        let complete = MinimaxM2ToolParser::new(&test_tools()).parse_complete(text).unwrap();
        expect![[r#"
            Checking.
            0 calculate_area {"shape": "\n  rect\\angle \"wide\" 🍎\n", "dimensions": {"width": 10, "height": 20}, "precision": 2}
            1 get_weather {"city": "a <b> "}
            2 no_schema {"payload": "{\"a\": 1}", "count": "5"}"#]]
        .assert_eq(
            &std::iter::once(complete.normal_text())
                .chain(complete.calls().iter().map(|call| {
                    format!(
                        "{} {} {}",
                        call.tool_index,
                        call.name.as_deref().unwrap(),
                        call.arguments
                    )
                }))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        for size in 1..=8 {
            let mut parser = MinimaxM2ToolParser::new(&test_tools());
            let streamed = collect_stream(&mut parser, &split_by_chars(text, size));
            assert_eq!(streamed, complete, "chunk size {size}");
        }
    }

    #[test]
    fn minimax_m2_keeps_the_first_value_of_a_duplicate_parameter() {
        // The first value may already be on the wire, so a later duplicate
        // cannot replace it.
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_block(&[(
                "get_weather",
                vec![
                    ("city", "Seattle"),
                    ("days", "5"),
                    ("city", "NYC"),
                    ("days", "7"),
                ],
            )]))
            .unwrap();

        assert_eq!(
            output.calls()[0].arguments,
            r#"{"city": "Seattle", "days": 5}"#
        );
    }

    #[test]
    fn minimax_m2_invoke_end_inside_a_parameter_is_malformed() {
        for (name, value) in [("city", "Seattle"), ("days", "5")] {
            // Without and with a later `</parameter>` that could close the value.
            for rest in [
                "",
                r#"<invoke name="get_weather"><parameter name="city">NYC</parameter></invoke>"#,
            ] {
                let mut parser = MinimaxM2ToolParser::new(&test_tools());
                let text = format!(
                    r#"<minimax:tool_call><invoke name="get_weather"><parameter name="{name}">{value}</invoke>{rest}</minimax:tool_call>"#
                );

                assert!(parser.parse_complete(&text).is_err(), "{name}: {rest}");
            }
        }
    }

    #[test]
    fn minimax_m2_finish_fails_incomplete_tool_call() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        parser.parse_chunk(r#"<minimax:tool_call><invoke name="get_weather">"#).unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m2_finish_fails_after_bare_tool_block_start() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        parser.parse_chunk("<minimax:tool_call>").unwrap();

        assert!(parser.finish().is_err());
    }

    #[test]
    fn minimax_m2_malformed_tool_call_fails_fast() {
        let mut parser = MinimaxM2ToolParser::new(&test_tools());
        let error = parser.parse_chunk("<minimax:tool_call><bad></minimax:tool_call>").unwrap_err();

        expect![[r#"tool parser parsing failed: near "<bad></minimax:tool_call>": "#]]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn tool_framing_preserves_body_whitespace_across_chunk_boundaries() {
        assert_tool_framing_preserves_body_whitespace::<MinimaxM2ToolParser>(
            "\n",
            "<minimax:tool_call><invoke name=\"get_weather\"></invoke></minimax:tool_call>",
        );
    }
}
