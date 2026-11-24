// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::alt;
use winnow::prelude::*;
use winnow::stream::{Partial, Stream};

use super::json::MistralJsonArrayToolParser;
use super::utils::{
    Attributed, AttributedExt, JsonObjectScanState, Marker, MarkerScanState, MarkerStream, Scan,
    SpecialToken, incomplete, parse_buffered_event_attributed, safe_text_len, scan_markers,
    take_json_object, take_until_marker,
};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput};

const TOOL_CALLS: &str = "[TOOL_CALLS]";
const ARGS: &str = "[ARGS]";
const CALL_ID: &str = "[CALL_ID]";

type MistralInput<'i> = Attributed<'i, MistralMarkers>;

/// The control tokens of a v11+ Mistral tokenizer.
#[derive(Debug)]
struct MistralMarkers {
    tool_calls: Marker,
    args: Marker,
    /// Separates the name from a call ID; only the v11 encoder writes one.
    call_id: Option<Marker>,
}

impl MistralMarkers {
    /// Resolve both markers, or `None` when the tokenizer predates `[ARGS]`.
    fn new(tokenizer: &dyn Tokenizer) -> Option<Self> {
        let special = |text: &str| {
            tokenizer.token_to_id(text).map(|id| SpecialToken {
                text: text.to_string(),
                id,
            })
        };
        Some(Self {
            tool_calls: Marker::special(&special(TOOL_CALLS)?),
            args: Marker::special(&special(ARGS)?),
            call_id: special(CALL_ID).map(|token| Marker::special(&token)),
        })
    }
}

#[derive(Debug)]
enum MistralMode {
    Text,
    Name,
    /// After `name[CALL_ID]`, with the name held back until its ID is known.
    CallId {
        name: String,
        args_scan: MarkerScanState,
    },
    Arguments {
        json_scan: JsonObjectScanState,
    },
    /// After a call's arguments, where only the next call may follow.
    BetweenCalls,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum MistralEvent {
    Text {
        len: usize,
    },
    ToolCallStart,
    /// A name ended by `[ARGS]`, or by `[CALL_ID]` when `call_id_follows`.
    Name {
        name: String,
        call_id_follows: bool,
    },
    CallId {
        name: String,
        id: String,
    },
    Arguments {
        len: usize,
    },
}

/// Tool parser for the v11+ Mistral format, where each call is
/// `[TOOL_CALLS]name[ARGS]` followed by its JSON arguments; the v11 encoder
/// also writes a call ID as `name[CALL_ID]id[ARGS]`. A tokenizer without
/// `[ARGS]` (before v11) gets [`MistralJsonArrayToolParser`] instead, like
/// Python's `MistralParser`.
///
/// Example tool call content:
///
/// ```text
/// [TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}[TOOL_CALLS]add[ARGS]{"x": 1, "y": 2}
/// ```
///
/// Both markers are control tokens, so they count only where those tokens
/// produced them. Arguments are already OpenAI-style JSON text, so they are
/// streamed as raw argument deltas.
pub struct MistralToolParser {
    buffer: DecodedText,
    mode: MistralMode,
    emitted_tool_count: usize,
    call_ids: BTreeMap<usize, String>,
    markers: MistralMarkers,
}

impl MistralToolParser {
    /// Create a parser for the v11+ markers.
    fn new(markers: MistralMarkers) -> Self {
        Self {
            buffer: DecodedText::default(),
            mode: MistralMode::Text,
            emitted_tool_count: 0,
            call_ids: BTreeMap::new(),
            markers,
        }
    }

    /// Start the current call with its final name.
    fn start_call(&mut self, name: String, output: &mut ToolParserOutput) {
        output.push_call(ToolCallDelta {
            tool_index: self.emitted_tool_count,
            name: Some(name),
            arguments: String::new(),
        });
        self.mode = MistralMode::Arguments {
            json_scan: JsonObjectScanState::default(),
        };
    }

    /// Apply one parsed event to parser state and output.
    fn apply_event(&mut self, event: MistralEvent, output: &mut ToolParserOutput) {
        match event {
            MistralEvent::Text { len } => output.push_text(&self.buffer.text[..len]),
            MistralEvent::ToolCallStart => self.mode = MistralMode::Name,
            MistralEvent::Name {
                name,
                call_id_follows: true,
            } => {
                self.mode = MistralMode::CallId {
                    name,
                    args_scan: MarkerScanState::default(),
                };
            }
            MistralEvent::Name {
                name,
                call_id_follows: false,
            } => self.start_call(name, output),
            MistralEvent::CallId { name, id } => {
                self.call_ids.insert(self.emitted_tool_count, id);
                self.start_call(name, output);
            }
            MistralEvent::Arguments { len } => {
                output.push_call(ToolCallDelta {
                    tool_index: self.emitted_tool_count,
                    name: None,
                    arguments: self.buffer.text[..len].to_string(),
                });
                if let MistralMode::Arguments { json_scan } = &self.mode
                    && json_scan.complete()
                {
                    self.emitted_tool_count += 1;
                    self.mode = MistralMode::BetweenCalls;
                }
            }
        }
    }
}

impl ToolParser for MistralToolParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        match MistralMarkers::new(tokenizer.as_ref()) {
            Some(markers) => Ok(Box::new(Self::new(markers))),
            None => MistralJsonArrayToolParser::create(tools, tokenizer),
        }
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.call_ids.get(&tool_index).map(String::as_str)
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.buffer.append(chunk);

        while let Some((event, consumed_len)) =
            parse_buffered_event_attributed(&self.buffer, &self.markers, |input| {
                parse_next_mistral_event(input, &mut self.mode)
            })?
        {
            self.apply_event(event, output);
            let _ = self.buffer.drain_prefix(consumed_len);
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        match self.mode {
            MistralMode::Text => output.push_text(&self.buffer.text),
            // Only whitespace can be left over after a complete call.
            MistralMode::BetweenCalls => {}
            MistralMode::Name | MistralMode::CallId { .. } | MistralMode::Arguments { .. } => {
                return Err(ToolParserError::IncompleteToolCall { parser: "Mistral" });
            }
        }
        let _ = self.reset();
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = MistralMode::Text;
        self.emitted_tool_count = 0;
        self.call_ids.clear();
        std::mem::take(&mut self.buffer).text
    }
}

/// Parse a Mistral event for the current parser mode.
fn parse_next_mistral_event(
    input: &mut MistralInput<'_>,
    mode: &mut MistralMode,
) -> ModalResult<MistralEvent> {
    match mode {
        MistralMode::Text => alt((tool_call_start_event, safe_text_event)).parse_next(input),
        MistralMode::Name => name_event(input),
        MistralMode::CallId { name, args_scan } => call_id_event(input, name, args_scan),
        MistralMode::Arguments { json_scan } => arguments_event(input, json_scan),
        MistralMode::BetweenCalls => {
            (ws0, tool_call_start_event).map(|(_, event)| event).parse_next(input)
        }
    }
}

/// Parse a `[TOOL_CALLS]` marker.
fn tool_call_start_event(input: &mut MistralInput<'_>) -> ModalResult<MistralEvent> {
    let markers = input.markers();
    (&markers.tool_calls).value(MistralEvent::ToolCallStart).parse_next(input)
}

/// Parse a safe text run before the next `[TOOL_CALLS]` marker.
fn safe_text_event(input: &mut MistralInput<'_>) -> ModalResult<MistralEvent> {
    let markers = input.markers();
    safe_text_len(input, &markers.tool_calls).map(|len| MistralEvent::Text { len })
}

/// Parse a tool name up to and including the `[ARGS]` or `[CALL_ID]` marker
/// that ends it.
fn name_event(input: &mut MistralInput<'_>) -> ModalResult<MistralEvent> {
    let markers = input.markers();
    let ends: Vec<&Marker> = std::iter::once(&markers.args).chain(&markers.call_id).collect();
    let start = input.offset();
    let Scan::Found { at, index } = scan_markers(input, &ends, start)? else {
        return incomplete();
    };
    let name = input.next_slice(at - start).to_string();
    let mut end = ends[index];
    end.parse_next(input)?;
    Ok(MistralEvent::Name {
        name,
        call_id_follows: index == 1,
    })
}

/// Parse a call ID up to and including its `[ARGS]` marker.
fn call_id_event(
    input: &mut MistralInput<'_>,
    name: &mut String,
    args_scan: &mut MarkerScanState,
) -> ModalResult<MistralEvent> {
    let markers = input.markers();
    let (id, _) = (take_until_marker(&markers.args, args_scan), &markers.args).parse_next(input)?;
    Ok(MistralEvent::CallId {
        name: std::mem::take(name),
        id: id.to_string(),
    })
}

/// Parse a raw JSON arguments delta.
fn arguments_event(
    input: &mut MistralInput<'_>,
    json_scan: &mut JsonObjectScanState,
) -> ModalResult<MistralEvent> {
    let len = take_json_object(&mut Partial::new(input.remaining()), json_scan)?;
    input.next_slice(len);
    Ok(MistralEvent::Arguments { len })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use thiserror_ext::AsReport;
    use vllm_tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::{ARGS, CALL_ID, MistralToolParser, TOOL_CALLS};
    use crate::tool::test_utils::{
        collect_attributed_stream, decode_ids, split_by_chars, test_tools,
    };
    use crate::tool::{ToolParser, ToolParserOutput};

    /// A v11+ tokenizer, where the markers are control tokens.
    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_special_token(TOOL_CALLS, 256)
            .with_special_token(ARGS, 257)
            .with_special_token(CALL_ID, 258)
    }

    fn parser() -> Box<dyn ToolParser> {
        MistralToolParser::create(&test_tools(), Arc::new(tokenizer())).unwrap()
    }

    fn call(name: &str, arguments: &str) -> String {
        format!("{TOOL_CALLS}{name}{ARGS}{arguments}")
    }

    fn parse_complete(text: &str) -> ToolParserOutput {
        collect_attributed_stream(parser().as_mut(), &tokenizer(), &[text])
    }

    #[test]
    fn mistral_parses_args_calls_after_content() {
        let output = parse_complete(&format!(
            "Let me check.{}{}",
            call("get_weather", r#"{"city": "Paris"}"#),
            call("add", r#"{"x": 1, "y": 2}"#),
        ));

        assert_eq!(output.normal_text(), "Let me check.");
        let calls = output.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (calls[0].tool_index, calls[0].name.as_deref()),
            (0, Some("get_weather"))
        );
        assert_eq!(calls[0].arguments, r#"{"city": "Paris"}"#);
        assert_eq!(
            (calls[1].tool_index, calls[1].name.as_deref()),
            (1, Some("add"))
        );
        assert_eq!(calls[1].arguments, r#"{"x": 1, "y": 2}"#);
    }

    #[test]
    fn mistral_streaming_matches_complete_parse() {
        let text = format!(
            "Checking.{}{TOOL_CALLS}add{CALL_ID}abc123XYZ{ARGS}{}",
            call("get_weather", r#"{"city": "Paris", "days": [1, 2]}"#),
            r#"{"x": "}{", "y": 2}"#,
        );
        let complete = parse_complete(&text);
        assert_eq!(complete.normal_text(), "Checking.");
        let calls = complete.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments, r#"{"city": "Paris", "days": [1, 2]}"#);
        assert_eq!(
            (calls[1].name.as_deref(), calls[1].arguments.as_str()),
            (Some("add"), r#"{"x": "}{", "y": 2}"#)
        );

        for chunk_chars in [1, 2, 3, 7] {
            let streamed = collect_attributed_stream(
                parser().as_mut(),
                &tokenizer(),
                &split_by_chars(&text, chunk_chars),
            );
            assert_eq!(streamed, complete, "chunk_chars={chunk_chars}");
        }
    }

    #[test]
    fn mistral_reports_v11_call_ids() {
        let text = format!(
            "{TOOL_CALLS}get_weather{CALL_ID}abc123XYZ{ARGS}{}{}",
            r#"{"city": "Paris"}"#,
            call("add", r#"{"x": 1}"#),
        );
        let mut parser = parser();
        let mut output = ToolParserOutput::default();
        let ids = tokenizer().encode(&text, false).unwrap();
        parser.parse_into(decode_ids(&tokenizer(), &ids), &mut output).unwrap();

        let output = output.coalesce();
        let calls = output.calls();
        assert_eq!(
            (calls[0].name.as_deref(), calls[0].arguments.as_str()),
            (Some("get_weather"), r#"{"city": "Paris"}"#)
        );
        assert_eq!(calls[1].name.as_deref(), Some("add"));
        assert_eq!(parser.tool_call_id(0), Some("abc123XYZ"));
        assert_eq!(parser.tool_call_id(1), None);
    }

    #[test]
    fn mistral_streams_name_before_arguments() {
        let mut parser = parser();
        let mut output = ToolParserOutput::default();
        let mut decoded = decode_ids(
            &tokenizer(),
            &tokenizer().encode(&call("add", r#"{"x": 1}"#), false).unwrap(),
        );
        let head = decoded.drain_prefix(format!("{TOOL_CALLS}add{ARGS}{{\"x\"").len());
        parser.parse_into(head, &mut output).unwrap();

        let calls = output.calls();
        assert_eq!(calls[0].name.as_deref(), Some("add"));
        assert_eq!(
            output.calls().iter().map(|call| call.arguments.as_str()).collect::<String>(),
            r#"{"x""#
        );

        parser.parse_into(decoded, &mut output).unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.coalesce().calls()[0].arguments, r#"{"x": 1}"#);
    }

    #[test]
    fn mistral_marker_spelled_by_ordinary_tokens_stays_text() {
        let text = call("add", r#"{"x": 1}"#);
        let mut parser = parser();
        let mut output = ToolParserOutput::default();
        let ordinary = tokenizer().encode_ordinary(&text).unwrap();
        parser.parse_into(decode_ids(&tokenizer(), &ordinary), &mut output).unwrap();
        output.append(parser.finish().unwrap());

        assert_eq!(output.normal_text(), text);
        assert!(output.calls().is_empty());
    }

    #[test]
    fn mistral_truncated_call_is_incomplete() {
        let mut parser = parser();
        let mut output = ToolParserOutput::default();
        let text = format!("{TOOL_CALLS}add{ARGS}{{\"x\": 1");
        let ids = tokenizer().encode(&text, false).unwrap();
        parser.parse_into(decode_ids(&tokenizer(), &ids), &mut output).unwrap();

        let error = parser.finish().unwrap_err();
        assert_eq!(error.to_report_string(), "incomplete Mistral tool call");
    }

    #[test]
    fn mistral_preserves_special_tokens() {
        assert!(parser().preserve_special_tokens());
    }

    #[test]
    fn mistral_without_args_token_parses_json_arrays() {
        let tokenizer = TestTokenizer::new().with_special_token(TOOL_CALLS, 256);
        let mut parser = MistralToolParser::create(&test_tools(), Arc::new(tokenizer)).unwrap();
        let mut output = ToolParserOutput::default();
        let text = r#"[TOOL_CALLS] [{"name": "add", "arguments": {"x": 1}}]"#;
        parser
            .parse_into(vllm_tokenizer::DecodedText::unattributed(text), &mut output)
            .unwrap();
        output.append(parser.finish().unwrap());

        let output = output.coalesce();
        assert_eq!(output.calls()[0].name.as_deref(), Some("add"));
        assert_eq!(output.calls()[0].arguments, r#"{"x": 1}"#);
    }
}
