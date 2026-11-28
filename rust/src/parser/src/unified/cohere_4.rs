// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Native unified parser for Cohere Command cmd4 models
//!
//! Limitations:
//! * citations are not parsed yet!
//! * no structural-tag/guided decoding.

use vllm_tokenizer::{DecodedText, DynTokenizer};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, seq};
use winnow::error::{ModalResult, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::Partial;
use winnow::token::{literal, rest};

use super::{Result, UnifiedParser, UnifiedParserOutput, token_id};
use crate::reasoning::last_reasoning_boundary;
use crate::tool::{Tool, ToolCallDelta};
use crate::unified::parsing_failed;
use crate::utils::{
    JsonObjectScanState, json_str, parse_buffered_event, safe_text_len_mul, take_json_object,
};

const START_THINKING: &str = "<|START_THINKING|>";
const END_THINKING: &str = "<|END_THINKING|>";
const START_TEXT: &str = "<|START_TEXT|>";
const END_TEXT: &str = "<|END_TEXT|>";
const START_RESPONSE: &str = "<|START_RESPONSE|>";
const END_RESPONSE: &str = "<|END_RESPONSE|>";
const START_ACTION: &str = "<|START_ACTION|>";
const END_ACTION: &str = "<|END_ACTION|>";

/// Markers that end a run of reasoning text.
const REASONING_MARKERS: &[&str] = &[END_THINKING];
/// Markers that open the block after thinking.
const IDLE_MARKERS: &[&str] = &[START_TEXT, START_RESPONSE, START_ACTION];
/// Markers that end a run of answer text.
const ANSWER_MARKERS: &[&str] = &[END_TEXT, END_RESPONSE];
/// Markers that end a run of filler between tool-call objects.
const ACTION_FILLER_MARKERS: &[&str] = &[END_ACTION, "{"];

type CohereInput<'i> = Partial<&'i str>;

/// One recognized span of Cohere renderer output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CohereEvent<T> {
    /// Reasoning text.
    Reasoning,
    /// Answer text.
    Text,
    /// Marker or action-array filler: consumed and dropped.
    Skip,
    ReasoningClose,
    AnswerOpen,
    AnswerClose,
    ActionOpen,
    ActionClose,
    /// The head of one tool-call object up to and including `"parameters":`.
    ToolCallStart {
        id: String,
        name: String,
    },
    /// One renderer-specific tool-argument delta.
    ToolCallArguments(T),
    /// The end of one tool call.
    ToolCallEnd,
}

/// Shared outer state for Cohere renderers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CohereMode<A> {
    /// Inside the thinking block.
    Reasoning,
    /// Between the thinking block and the next block opener; also the start
    /// state when thinking is off. Stray text here is visible content.
    Idle,
    /// Inside the answer block.
    Answer,
    /// Inside the renderer-specific action block.
    Action(A),
    /// After the answer or action block closed: ignore the rest.
    Done,
}

type Cohere4Event = CohereEvent<()>;
type Cohere4Mode = CohereMode<Option<JsonObjectScanState>>;

/// Cohere Command cmd4 parser.
pub struct Cohere4UnifiedParser {
    buffer: DecodedText,
    mode: Cohere4Mode,
    tokenizer: DynTokenizer,
    start_token_id: u32,
    end_token_id: u32,
    /// Model-emitted tool call ids; the position is the `tool_index`.
    tool_call_ids: Vec<String>,
}

impl Cohere4UnifiedParser {
    /// Create a cmd4 parser. Tool definitions are not needed: the action
    /// block is self-describing.
    pub fn new(_tools: &[Tool], tokenizer: DynTokenizer) -> Result<Self> {
        let start_token_id = token_id(tokenizer.as_ref(), START_THINKING)?;
        let end_token_id = token_id(tokenizer.as_ref(), END_THINKING)?;
        Ok(Self {
            buffer: DecodedText::default(),
            mode: CohereMode::Reasoning,
            tokenizer,
            start_token_id,
            end_token_id,
            tool_call_ids: Vec::new(),
        })
    }

    /// Derive the start state from the prompt. The cmd4 template ends the
    /// generation prompt with `<|START_THINKING|>` (thinking on) or with
    /// `<|START_THINKING|><|END_THINKING|>` (thinking off); with no boundary
    /// at all the parser starts in thinking, as the Python parser does.
    fn initialize_mode(&mut self, prompt_token_ids: &[u32]) {
        self.mode = match last_reasoning_boundary(
            prompt_token_ids,
            self.start_token_id,
            self.end_token_id,
            self.tokenizer.as_ref(),
        ) {
            Some(false) => CohereMode::Idle,
            Some(true) | None => CohereMode::Reasoning,
        };
    }

    fn apply_event(
        &mut self,
        event: Cohere4Event,
        piece: DecodedText,
        output: &mut UnifiedParserOutput,
    ) -> Result<()> {
        match event {
            CohereEvent::Reasoning => output.push_reasoning(piece),
            CohereEvent::Text => output.push_text(piece.text),
            CohereEvent::Skip => {}
            CohereEvent::ReasoningClose => self.mode = CohereMode::Idle,
            CohereEvent::AnswerOpen => self.mode = CohereMode::Answer,
            CohereEvent::AnswerClose | CohereEvent::ActionClose => {
                self.mode = CohereMode::Done;
            }
            CohereEvent::ActionOpen => self.mode = CohereMode::Action(None),
            CohereEvent::ToolCallStart { id, name } => {
                let CohereMode::Action(call) = &mut self.mode else {
                    return Err(parsing_failed!("Cohere tool call outside an action block"));
                };
                *call = Some(JsonObjectScanState::default());
                let tool_index = self.tool_call_ids.len();
                self.tool_call_ids.push(id);
                output.push_call(ToolCallDelta {
                    tool_index,
                    name: Some(name),
                    arguments: String::new(),
                });
            }
            CohereEvent::ToolCallArguments(()) => output.push_call(ToolCallDelta {
                tool_index: self.tool_call_ids.len() - 1,
                name: None,
                arguments: piece.text,
            }),
            CohereEvent::ToolCallEnd => self.mode = CohereMode::Action(None),
        }
        Ok(())
    }
}

impl UnifiedParser for Cohere4UnifiedParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn UnifiedParser>>
    where
        Self: Sized + 'static,
    {
        Self::new(tools, tokenizer).map(|parser| Box::new(parser) as Box<dyn UnifiedParser>)
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.buffer.clear();
        self.tool_call_ids.clear();
        self.initialize_mode(prompt_token_ids);
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.tool_call_ids
            .get(tool_index)
            .map(String::as_str)
            .filter(|id| !id.is_empty())
    }

    fn parse_into(&mut self, delta: DecodedText, output: &mut UnifiedParserOutput) -> Result<()> {
        self.buffer.append(delta);

        while let Some((event, consumed_len)) = parse_buffered_event(&self.buffer.text, |input| {
            parse_next_cohere_event(input, &mut self.mode)
        })? {
            let piece = self.buffer.drain_prefix(consumed_len);
            self.apply_event(event, piece, output)?;
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();

        match &self.mode {
            CohereMode::Reasoning => output.push_reasoning(self.buffer.take()),
            CohereMode::Idle | CohereMode::Answer => {
                output.push_text(self.buffer.take().text);
            }
            // A call cut off by the length limit is dropped: its streamed
            // argument deltas already reached the client, and partial JSON is
            // never surfaced as content (the Python parser does the same).
            CohereMode::Action(_) | CohereMode::Done => self.buffer.clear(),
        }

        self.mode = CohereMode::Done;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.mode = CohereMode::Done;
        self.tool_call_ids.clear();
        self.buffer.take().text
    }
}

/// Parse one cmd4 event from buffered streaming input.
fn parse_next_cohere_event(
    input: &mut CohereInput<'_>,
    mode: &mut Cohere4Mode,
) -> ModalResult<Cohere4Event> {
    match mode {
        CohereMode::Reasoning => parse_reasoning_event(input),
        CohereMode::Idle => parse_idle_event(input),
        CohereMode::Answer => parse_answer_event(input),
        CohereMode::Action(None) => parse_action_event(input),
        CohereMode::Action(Some(scan)) => parse_call_event(input, scan),
        CohereMode::Done => rest.value(CohereEvent::Skip).parse_next(input),
    }
}

/// Parse an event inside the thinking block.
fn parse_reasoning_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    alt((
        literal(END_THINKING).value(CohereEvent::ReasoningClose),
        safe_reasoning_event,
    ))
    .parse_next(input)
}

/// Parse an event between the thinking block and the next block opener.
fn parse_idle_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    alt((
        literal(START_TEXT).value(CohereEvent::AnswerOpen),
        literal(START_RESPONSE).value(CohereEvent::AnswerOpen),
        literal(START_ACTION).value(CohereEvent::ActionOpen),
        |i: &mut CohereInput<'_>| safe_text_len_mul(i, IDLE_MARKERS).map(|_| CohereEvent::Text),
    ))
    .parse_next(input)
}

/// Parse an event inside the answer block.
fn parse_answer_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    alt((
        literal(END_TEXT).value(CohereEvent::AnswerClose),
        literal(END_RESPONSE).value(CohereEvent::AnswerClose),
        safe_answer_text_event,
    ))
    .parse_next(input)
}

/// Parse an event between tool-call objects in the action array.
fn parse_action_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    alt((
        literal(END_ACTION).value(CohereEvent::ActionClose),
        call_head_event,
        action_filler_event,
    ))
    .parse_next(input)
}

/// Skip bytes before the next tool-call object or the end marker.
///
/// The array syntax is never validated, matching the Python parser, which
/// discards everything before the next `"tool_call_id"` key. A `{` that does not open
/// a well-formed call head is skipped so the scan can move past it.
fn action_filler_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    alt(('{'.value(CohereEvent::Skip), |i: &mut CohereInput<'_>| {
        safe_text_len_mul(i, ACTION_FILLER_MARKERS).map(|_| CohereEvent::Skip)
    }))
    .parse_next(input)
}

/// Parse a configured JSON object key (same as the JSON tool parser's).
fn json_key(input: &mut CohereInput<'_>, key: &'static str) -> ModalResult<()> {
    seq!(
        _: literal("\""),
        _: literal(key).context(StrContext::Expected(StrContextValue::StringLiteral(key))),
        _: literal("\""),
    )
    .void()
    .parse_next(input)
}

/// Parse the head of a tool-call object in template key order.
fn call_head_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    seq!(CohereEvent::ToolCallStart {
        _: ('{', ws0, |i: &mut CohereInput<'_>| json_key(i, "tool_call_id"), ws0, ':', ws0),
        id: json_str,
        _: (ws0, ',', ws0, |i: &mut CohereInput<'_>| json_key(i, "tool_name"), ws0, ':', ws0),
        name: json_str,
        _: (ws0, ',', ws0, |i: &mut CohereInput<'_>| json_key(i, "parameters"), ws0, ':', ws0),
    })
    .parse_next(input)
}

/// Parse an event inside one tool-call object after its head.
fn parse_call_event(
    input: &mut CohereInput<'_>,
    scan: &mut JsonObjectScanState,
) -> ModalResult<Cohere4Event> {
    if scan.complete() {
        (ws0, '}').value(CohereEvent::ToolCallEnd).parse_next(input)
    } else {
        take_json_object(input, scan).map(|_| CohereEvent::ToolCallArguments(()))
    }
}

/// Parse safe reasoning text before the next marker.
fn safe_reasoning_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    safe_text_len_mul(input, REASONING_MARKERS).map(|_| CohereEvent::Reasoning)
}

/// Parse safe answer text before the next marker.
fn safe_answer_text_event(input: &mut CohereInput<'_>) -> ModalResult<Cohere4Event> {
    safe_text_len_mul(input, ANSWER_MARKERS).map(|_| CohereEvent::Text)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::{DecodedText, test_utils::TestTokenizer};

    use super::{Cohere4UnifiedParser, END_THINKING, START_THINKING};
    use crate::tool::ToolCallDelta;
    use crate::tool::test_utils::{split_by_chars, test_tools};
    use crate::unified::{UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

    /// Token ids of the Cohere thinking markers in `cohere_test_tokenizer`.
    const COHERE_START_THINKING_ID: u32 = 256;
    const COHERE_END_THINKING_ID: u32 = 257;

    /// A tokenizer knowing the six Cohere Command cmd4 markers as special tokens.
    fn cohere_test_tokenizer() -> TestTokenizer {
        TestTokenizer::new()
            .with_special_token("<|START_THINKING|>", COHERE_START_THINKING_ID)
            .with_special_token("<|END_THINKING|>", COHERE_END_THINKING_ID)
            .with_special_token("<|START_TEXT|>", 258)
            .with_special_token("<|END_TEXT|>", 259)
            .with_special_token("<|START_ACTION|>", 260)
            .with_special_token("<|END_ACTION|>", 261)
    }

    /// Push chunks through `parser`, finish, and return every event.
    fn collect_stream(parser: &mut dyn UnifiedParser, chunks: &[&str]) -> UnifiedParserOutput {
        let mut output = UnifiedParserOutput::default();
        for chunk in chunks {
            parser.parse_into(DecodedText::unattributed(*chunk), &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());
        output
    }

    fn normal_text(output: &UnifiedParserOutput) -> String {
        output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning_text(output: &UnifiedParserOutput) -> String {
        output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Every string the stream would surface to a client: text, reasoning,
    /// tool names and argument deltas.
    fn streamed_strings(output: &UnifiedParserOutput) -> Vec<&str> {
        output
            .events
            .iter()
            .flat_map(|event| match event {
                UnifiedParserEvent::Text(text) => vec![text.as_str()],
                UnifiedParserEvent::Reasoning(piece) => vec![piece.text.as_str()],
                UnifiedParserEvent::ToolCall(call) => {
                    let mut fields = vec![call.arguments.as_str()];
                    fields.extend(call.name.as_deref());
                    fields
                }
            })
            .collect()
    }

    fn calls(output: &UnifiedParserOutput) -> Vec<ToolCallDelta> {
        output
            .events
            .iter()
            .filter_map(|event| match event {
                UnifiedParserEvent::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect()
    }

    /// Accumulate streamed tool-call deltas into `(index, name, arguments)`.
    fn accumulated_calls(output: &UnifiedParserOutput) -> Vec<(usize, String, String)> {
        let mut accumulated: Vec<(usize, String, String)> = Vec::new();
        for call in calls(output) {
            match accumulated.iter_mut().find(|(index, ..)| *index == call.tool_index) {
                Some((_, _, arguments)) => arguments.push_str(&call.arguments),
                None => accumulated.push((
                    call.tool_index,
                    call.name.clone().expect("first delta carries the name"),
                    call.arguments,
                )),
            }
        }
        accumulated
    }

    fn parser() -> Cohere4UnifiedParser {
        let mut parser =
            Cohere4UnifiedParser::new(&test_tools(), Arc::new(cohere_test_tokenizer())).unwrap();
        parser.initialize(&[COHERE_START_THINKING_ID]).unwrap();
        parser
    }

    const ANSWER: &str = "I think about it.<|END_THINKING|><|START_TEXT|>Hello there!<|END_TEXT|>";
    const ACTION: &str = "I will call a tool.<|END_THINKING|><|START_ACTION|>[\n    {\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}},\n    {\"tool_call_id\": \"1\", \"tool_name\": \"get_weather\", \"parameters\": {\"city\": \"Tokyo\", \"days\": [1, 2]}}\n]<|END_ACTION|>";

    // Multi-hop wires from the Python parser's tests. `<|START_THINKING|>` is
    // omitted because the parser learns it from the prompt; Python behaviours
    // that cmd4 models do not need are pinned as divergences below.
    const MULTI_HOP_ACTION: &str = "plan1<|END_THINKING|>\n<|START_ACTION|>\n[\n    {\n         \"tool_call_id\": \"2\",\n        \"tool_name\": \"internet_search\",\n        \"parameters\": {\n            \"query\": \"query1\"\n        }\n    }\n]<|END_ACTION|>";
    const MULTI_HOP_CITED_THINKING: &str = "This is a rainbow <co>emoji: 🌈</co: 0:[1]><|END_THINKING|>\n<|START_TEXT|>foo <co>bar</co: 0:[1,2],1:[3,4]><|END_TEXT|>";
    const MULTI_HOP_NO_THINKING: &str =
        "<|END_THINKING|><|START_TEXT|>foo <co>bar</co: 0:[1]><|END_TEXT|>";

    #[test]
    fn multi_hop_indented_action_streams_raw_parameters() {
        let mut parser = parser();
        let output = collect_stream(&mut parser, &[MULTI_HOP_ACTION]);
        assert_eq!(reasoning_text(&output), "plan1");
        // Divergence: the Python parser left-trims the newline after `<|END_THINKING|>`;
        // the target models emit the markers back to back, so the native
        // parser passes stray bytes between blocks through as content.
        assert_eq!(normal_text(&output), "\n");
        let calls = accumulated_calls(&output);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "internet_search");
        assert_eq!(parser.tool_call_id(0), Some("2"));
        // Divergence: the Python parser strips up to eight spaces of indentation
        // per line from the raw parameters; the native parser streams the bytes
        // verbatim, which parses to the same JSON.
        assert_eq!(
            calls[0].2,
            "{\n            \"query\": \"query1\"\n        }"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[0].2).unwrap(),
            serde_json::json!({"query": "query1"})
        );
    }

    #[test]
    fn multi_hop_citation_markup_passes_through_as_text() {
        let output = collect_stream(&mut parser(), &[MULTI_HOP_CITED_THINKING]);
        // Divergence: the Python parser strips the markup and reports
        // citations; the native parser does not parse citations yet.
        assert_eq!(
            reasoning_text(&output),
            "This is a rainbow <co>emoji: 🌈</co: 0:[1]>"
        );
        assert_eq!(normal_text(&output), "\nfoo <co>bar</co: 0:[1,2],1:[3,4]>");
        assert!(calls(&output).is_empty());
    }

    #[test]
    fn multi_hop_end_thinking_first_yields_no_reasoning() {
        let output = collect_stream(&mut parser(), &[MULTI_HOP_NO_THINKING]);
        assert_eq!(reasoning_text(&output), "");
        assert_eq!(normal_text(&output), "foo <co>bar</co: 0:[1]>");
        assert!(calls(&output).is_empty());
    }

    // Ported from the Python cmd4 streaming regressions in
    // `tests/parser/cohere`: scheduler steps that batch the reasoning end
    // with the answer, and chunk edges that look like a marker prefix.
    #[test]
    fn batched_reasoning_end_and_answer_in_one_delta_preserves_content() {
        let unbatched = [
            "I think D",
            END_THINKING,
            "<|START_TEXT|>",
            "Answer: D",
            "<|END_TEXT|>",
        ];
        let batched = [
            "I think D",
            "<|END_THINKING|><|START_TEXT|>Answer: D<|END_TEXT|>",
        ];
        for chunks in [&unbatched[..], &batched[..]] {
            let output = collect_stream(&mut parser(), chunks);
            assert_eq!(reasoning_text(&output), "I think D");
            assert_eq!(normal_text(&output), "Answer: D");
        }
    }

    #[test]
    fn batched_handoff_preserves_reasoning_tail_before_answer() {
        let output = collect_stream(
            &mut parser(),
            &[
                "The",
                " rest of the reasoning<|END_THINKING|><|START_TEXT|>Final: 69372<|END_TEXT|>",
            ],
        );
        assert_eq!(reasoning_text(&output), "The rest of the reasoning");
        assert_eq!(normal_text(&output), "Final: 69372");
    }

    #[test]
    fn batched_reasoning_end_keeps_held_back_tail() {
        for (tail, next, expected) in [
            (
                ["import sys", "\n"],
                "from array import array",
                "import sys\nfrom array import array",
            ),
            (
                ["def f():", "\n\t    "],
                "return 1",
                "def f():\n\t    return 1",
            ),
            (["#include", " <"], "vector>", "#include <vector>"),
            (["std::vector", "<c"], "har> v;", "std::vector<char> v;"),
        ] {
            let step = format!("<|END_THINKING|><|START_TEXT|>{}{}", tail[0], tail[1]);
            let output = collect_stream(&mut parser(), &["I think D", &step, next, "<|END_TEXT|>"]);
            assert_eq!(reasoning_text(&output), "I think D", "{expected:?}");
            assert_eq!(normal_text(&output), expected, "{expected:?}");
        }
    }

    #[test]
    fn chunk_ending_in_lt_before_a_marker_is_kept() {
        let output = collect_stream(
            &mut parser(),
            &[
                "I think a <",
                END_THINKING,
                "<|START_TEXT|>",
                "ans",
                "<|END_TEXT|>",
            ],
        );
        assert_eq!(reasoning_text(&output), "I think a <");
        assert_eq!(normal_text(&output), "ans");

        let output = collect_stream(
            &mut parser(),
            &[
                "I think D",
                END_THINKING,
                "<|START_TEXT|>",
                "x <",
                "<|END_TEXT|>",
            ],
        );
        assert_eq!(reasoning_text(&output), "I think D");
        assert_eq!(normal_text(&output), "x <");
    }

    const SPEC_DECODE_ACTION: &str = " i will call foo with query1<|END_THINKING|><|START_ACTION|>\n[\n    {\"tool_call_id\": \"0\", \"tool_name\": \"foo\", \"parameters\": {\"query\": \"query1\"}}\n]\n<|END_ACTION|>";

    #[test]
    fn multi_token_deltas_never_leak_markers_and_still_yield_the_call() {
        for size in [2, 3, 4, 6] {
            let mut parser = parser();
            let output = collect_stream(&mut parser, &split_by_chars(SPEC_DECODE_ACTION, size));
            for text in streamed_strings(&output) {
                assert!(
                    !text.contains("<|"),
                    "size {size}: marker leaked into {text:?}"
                );
            }
            let calls = accumulated_calls(&output);
            assert_eq!(calls.len(), 1, "size {size}");
            assert_eq!(calls[0].1, "foo");
            assert_eq!(parser.tool_call_id(0), Some("0"));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&calls[0].2).unwrap(),
                serde_json::json!({"query": "query1"})
            );
        }
    }

    #[test]
    fn missing_marker_token_fails() {
        let tokenizer = TestTokenizer::new().with_special_token(START_THINKING, 256);
        assert!(Cohere4UnifiedParser::new(&[], Arc::new(tokenizer)).is_err());
    }

    #[test]
    fn answer_is_split_into_reasoning_and_text() {
        let output = collect_stream(&mut parser(), &[ANSWER]);
        assert_eq!(reasoning_text(&output), "I think about it.");
        assert_eq!(normal_text(&output), "Hello there!");
        assert!(calls(&output).is_empty());
    }

    #[test]
    fn streaming_is_chunk_invariant() {
        for size in [1, 3, 7] {
            for wire in [ANSWER, ACTION, MULTI_HOP_ACTION, MULTI_HOP_CITED_THINKING] {
                let streamed = collect_stream(&mut parser(), &split_by_chars(wire, size));
                let whole = collect_stream(&mut parser(), &[wire]);
                assert_eq!(
                    reasoning_text(&streamed),
                    reasoning_text(&whole),
                    "size {size}"
                );
                assert_eq!(normal_text(&streamed), normal_text(&whole), "size {size}");
                assert_eq!(
                    accumulated_calls(&streamed),
                    accumulated_calls(&whole),
                    "size {size}"
                );
            }
        }
    }

    #[test]
    fn every_split_of_the_action_wire_parses_identically() {
        let expected = accumulated_calls(&collect_stream(&mut parser(), &[ACTION]));
        for split in ACTION.char_indices().map(|(i, _)| i).chain([ACTION.len()]) {
            let output = collect_stream(&mut parser(), &[&ACTION[..split], &ACTION[split..]]);
            assert_eq!(accumulated_calls(&output), expected, "split at {split}");
            assert_eq!(
                reasoning_text(&output),
                "I will call a tool.",
                "split at {split}"
            );
        }
    }

    #[test]
    fn parallel_tool_calls_carry_ids_and_json_arguments() {
        let mut parser = parser();
        let output = collect_stream(&mut parser, &[ACTION]);
        let calls = accumulated_calls(&output);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].1, "get_weather");
        assert_eq!(calls[0].2, "{\"city\": \"Paris\"}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&calls[1].2).unwrap(),
            serde_json::json!({"city": "Tokyo", "days": [1, 2]})
        );
        assert_eq!(parser.tool_call_id(0), Some("0"));
        assert_eq!(parser.tool_call_id(1), Some("1"));
        assert_eq!(normal_text(&output), "");
    }

    #[test]
    fn malformed_action_array_syntax_is_skipped() {
        let wire = "t<|END_THINKING|><|START_ACTION|>junk {\"not\": \"a call\"} \
            {\"tool_call_id\": \"0\", \"tool_name\": \"a\", \"parameters\": {}} ;; \
            {\"tool_call_id\": \"1\", \"tool_name\": \"b\", \"parameters\": {}} trailing<|END_ACTION|>";
        for chunks in [vec![wire], split_by_chars(wire, 1)] {
            let output = collect_stream(&mut parser(), &chunks);
            let calls = accumulated_calls(&output);
            let names: Vec<&str> = calls.iter().map(|c| c.1.as_str()).collect();
            assert_eq!(names, ["a", "b"]);
            assert_eq!(normal_text(&output), "");
        }
    }

    #[test]
    fn arguments_stream_before_the_call_closes() {
        let mut parser = parser();
        let head = "x<|END_THINKING|><|START_ACTION|>[{\"tool_call_id\": \"0\", \"tool_name\": \"get_weather\", \"parameters\": {\"city\": \"Par";
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(DecodedText::unattributed(head), &mut output).unwrap();
        let calls = calls(&output);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls[1].arguments, "{\"city\": \"Par");
    }

    #[test]
    fn prompt_closing_thinking_starts_in_answer() {
        let mut parser = parser();
        parser.initialize(&[COHERE_START_THINKING_ID, COHERE_END_THINKING_ID]).unwrap();
        let output = collect_stream(&mut parser, &["<|START_TEXT|>Hi<|END_TEXT|>"]);
        assert_eq!(reasoning_text(&output), "");
        assert_eq!(normal_text(&output), "Hi");
    }

    #[test]
    fn legacy_response_delimiters_are_supported() {
        let wire = "plan<|END_THINKING|><|START_RESPONSE|>answer<|END_RESPONSE|>junk";
        for size in [1, 4, wire.len()] {
            let output = collect_stream(&mut parser(), &split_by_chars(wire, size));
            assert_eq!(reasoning_text(&output), "plan", "size {size}");
            assert_eq!(normal_text(&output), "answer", "size {size}");
        }
    }

    #[test]
    fn text_between_end_thinking_and_start_text_is_content() {
        let output = collect_stream(
            &mut parser(),
            &["t<|END_THINKING|>\n<|START_TEXT|>a<|END_TEXT|>"],
        );
        assert_eq!(reasoning_text(&output), "t");
        assert_eq!(normal_text(&output), "\na");
    }

    #[test]
    fn text_after_end_text_is_ignored() {
        let output = collect_stream(
            &mut parser(),
            &["t<|END_THINKING|><|START_TEXT|>a<|END_TEXT|>junk"],
        );
        assert_eq!(normal_text(&output), "a");
    }

    #[test]
    fn reset_returns_buffered_text() {
        let mut parser = parser();
        let mut output = UnifiedParserOutput::default();
        parser
            .parse_into(DecodedText::unattributed("thinking <|END_TH"), &mut output)
            .unwrap();
        assert_eq!(reasoning_text(&output), "thinking ");
        assert_eq!(parser.reset(), "<|END_TH");
    }
}
