// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};
use crate::utils::partial_prefix_len;

/// Marker opening the final answer in ERNIE 4.5 thinking-model output.
const RESPONSE_START: &str = "<response>";
/// Marker closing the final answer in ERNIE 4.5 thinking-model output.
const RESPONSE_END: &str = "</response>";

/// Reasoning parser for the ERNIE 4.5 thinking models.
///
/// ERNIE 4.5 uses standard `<think>`/`</think>` delimiters with a `\n` on each
/// side of both markers, and the `ERNIE-4.5-21B-A3B-Thinking` template wraps the
/// final answer in `<response>`/`</response>` before any `<tool_call>` blocks:
///
/// ```text
/// <think>
/// {reasoning}
/// </think>
/// <response>
/// {content}
/// </response>
///
///
/// <tool_call>
/// {"name": ..., "arguments": {...}}
/// </tool_call>
/// ```
///
/// The `<think>` framing is handled by the shared delimited state machine. This
/// parser adds the `<response>` wrapper on top: the markers and the newlines
/// framing them are dropped from content, and everything after `</response>`
/// is passed through verbatim so the tool calls that follow the answer reach
/// the tool parser untouched. Content that does not open with `<response>`,
/// such as `abc\n</think>\ndef` or a tools-only turn, is passed through as is.
pub struct Ernie45ReasoningParser {
    inner: DelimitedReasoningParser,
    /// Strips the `<response>`/`</response>` framing from content pieces.
    response_frame: ResponseFrameFilter,
}

impl Ernie45ReasoningParser {
    /// Create an ERNIE 4.5 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
                .with_after_start("\n")
                .with_before_end("\n")
                .with_after_end("\n")
                .build()?,
            response_frame: ResponseFrameFilter::default(),
        })
    }

    /// Filter the `<response>` framing out of one delta's content.
    ///
    /// Reasoning deltas leave the filter untouched, so the first content of the
    /// stream always reaches it in its opening state, however reasoning ended.
    fn filter_content(&mut self, delta: ReasoningDelta, finishing: bool) -> ReasoningDelta {
        let mut content = match delta.content {
            Some(piece) => self.response_frame.push(piece),
            None => DecodedText::default(),
        };
        if finishing {
            content.append(self.response_frame.finish());
        }

        ReasoningDelta {
            reasoning: delta.reasoning,
            content: (!content.is_empty()).then_some(content),
        }
    }
}

// TODO: implement `wrap_visible_format` once the `<response>` framing is
// modeled, so strict tool calling can constrain the stream from the first token
// instead of the final output only.
impl ReasoningParser for Ernie45ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)?;
        self.response_frame = ResponseFrameFilter::default();
        Ok(())
    }

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
        let inner_delta = self.inner.push(delta);
        Ok(self.filter_content(inner_delta, false))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let inner_delta = self.inner.finish();
        Ok(self.filter_content(inner_delta, true))
    }
}

/// Where the filter is within the content stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum FrameState {
    /// Before any content: the newlines after `</think>` are framing, and
    /// `<response>` may open the answer.
    #[default]
    Opening,
    /// `<response>` was consumed; the single newline the template renders after
    /// it is framing.
    Opened,
    /// Inside the wrapped answer, watching for `</response>`.
    Answer,
    /// Right after `</response>`: newlines are framing until other text
    /// arrives.
    Trailing,
    /// Content without a wrapper, or everything after the answer such as tool
    /// calls, is passed through verbatim.
    PassThrough,
}

/// Incremental filter that removes the `<response>`/`</response>` answer framing
/// from ERNIE 4.5 content.
///
/// Markers and their framing newlines may be split across pushes, so the filter
/// holds back any text that could still complete a marker, plus the single
/// trailing `\n` that could turn out to be the framing newline right before
/// `</response>`. Only the newlines the template renders around the markers are
/// dropped; newlines inside the answer are kept. The runs of newlines after
/// `</think>` and `</response>` are dropped whole, since the model does not
/// always reproduce the template's single newline there.
#[derive(Default)]
struct ResponseFrameFilter {
    state: FrameState,
    /// Content held back until the filter can tell framing from answer text.
    pending: DecodedText,
}

impl ResponseFrameFilter {
    /// Feed one content piece and return the content that is safe to emit.
    fn push(&mut self, content: DecodedText) -> DecodedText {
        self.pending.append(content);
        self.filter(false)
    }

    /// Flush held-back content at end of stream.
    fn finish(&mut self) -> DecodedText {
        self.filter(true)
    }

    fn filter(&mut self, finishing: bool) -> DecodedText {
        let mut emitted = DecodedText::default();
        loop {
            match self.state {
                FrameState::Opening => {
                    self.drop_leading_newlines();
                    let text = self.pending.text.as_str();
                    if text.starts_with(RESPONSE_START) {
                        let _ = self.pending.drain_prefix(RESPONSE_START.len());
                        self.state = FrameState::Opened;
                    } else if text.is_empty() || (!finishing && RESPONSE_START.starts_with(text)) {
                        // Nothing to decide on yet, or the marker may still complete.
                        break;
                    } else {
                        self.state = FrameState::PassThrough;
                    }
                }
                FrameState::Opened => {
                    if self.pending.text.starts_with('\n') {
                        let _ = self.pending.drain_prefix(1);
                    } else if self.pending.text.is_empty() && !finishing {
                        break;
                    }
                    self.state = FrameState::Answer;
                }
                FrameState::Answer => {
                    let text = self.pending.text.as_str();
                    if let Some(index) = text.find(RESPONSE_END) {
                        // The single newline right before `</response>` is framing.
                        let body_len = index - usize::from(text[..index].ends_with('\n'));
                        let marker_len = index - body_len + RESPONSE_END.len();
                        emitted.append(self.pending.drain_prefix(body_len));
                        let _ = self.pending.drain_prefix(marker_len);
                        self.state = FrameState::Trailing;
                    } else {
                        let keep_len = if finishing { 0 } else { held_suffix_len(text) };
                        let emit_len = text.len() - keep_len;
                        emitted.append(self.pending.drain_prefix(emit_len));
                        break;
                    }
                }
                FrameState::Trailing => {
                    self.drop_leading_newlines();
                    if self.pending.text.is_empty() {
                        break;
                    }
                    self.state = FrameState::PassThrough;
                }
                FrameState::PassThrough => {
                    emitted.append(self.pending.take());
                    break;
                }
            }
        }
        // At end of stream only zero-width token records can still be pending;
        // they are content like any other piece.
        if finishing {
            emitted.append(self.pending.take());
        }
        emitted
    }

    /// Drop the run of newlines at the start of the pending content.
    fn drop_leading_newlines(&mut self) {
        let text = &self.pending.text;
        let framing_len = text.len() - text.trim_start_matches('\n').len();
        if framing_len > 0 {
            let _ = self.pending.drain_prefix(framing_len);
        }
    }
}

/// Return the length of the trailing text that may still turn out to be the
/// `</response>` framing: a partial marker, plus the single newline before it.
fn held_suffix_len(text: &str) -> usize {
    let partial = partial_prefix_len(text, RESPONSE_END);
    partial + usize::from(text[..text.len() - partial].ends_with('\n'))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};

    use super::Ernie45ReasoningParser;
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{
        THINK_END_ID, THINK_START_ID, content_str, fake_tokenizer, push_str, reasoning_str,
    };

    #[test]
    fn finish_flushes_zero_width_only_content() {
        let mut parser = parser_after_reasoning();

        // A filtered special token decodes to no text but keeps its record.
        let delta = parser
            .push(DecodedText {
                text: String::new(),
                attributions: [TokenAttribution {
                    token_id: 42,
                    anchor: TokenAnchor::ZeroWidth { byte_offset: 0 },
                }]
                .into_iter()
                .collect(),
            })
            .unwrap();
        assert!(delta.is_empty());

        // The filter cannot tell framing from answer text yet, so the record is
        // held back; at end of stream it is content like any other piece.
        let flushed = parser.finish().unwrap();
        let content = flushed.content.expect("zero-width content is kept");
        assert_eq!(content.text, "");
        assert_eq!(
            content.attributions.iter().map(|attr| attr.token_id).collect::<Vec<_>>(),
            [42]
        );
    }

    /// A parser whose prompt already opened reasoning, as the ERNIE templates do.
    fn parser_in_reasoning() -> Ernie45ReasoningParser {
        let mut parser = Ernie45ReasoningParser::new(Arc::new(fake_tokenizer())).unwrap();
        parser.initialize(&[THINK_START_ID, u32::from(b'\n')]).unwrap();
        parser
    }

    /// A parser whose prompt did not open reasoning, so the output carries the
    /// explicit `<think>` marker.
    fn parser_before_reasoning() -> Ernie45ReasoningParser {
        Ernie45ReasoningParser::new(Arc::new(fake_tokenizer())).unwrap()
    }

    /// A parser whose prompt already closed reasoning with `</think>`.
    fn parser_after_reasoning() -> Ernie45ReasoningParser {
        let mut parser = Ernie45ReasoningParser::new(Arc::new(fake_tokenizer())).unwrap();
        parser.initialize(&[THINK_START_ID, THINK_END_ID]).unwrap();
        parser
    }

    /// Push each chunk and concatenate the reasoning/content parts, including
    /// whatever `finish()` flushes.
    fn collect(parser: &mut Ernie45ReasoningParser, chunks: &[&str]) -> (String, String) {
        let mut reasoning = String::new();
        let mut content = String::new();
        for chunk in chunks {
            let delta = push_str(parser, chunk);
            reasoning.push_str(reasoning_str(&delta).unwrap_or_default());
            content.push_str(content_str(&delta).unwrap_or_default());
        }
        let delta = parser.finish().unwrap();
        reasoning.push_str(reasoning_str(&delta).unwrap_or_default());
        content.push_str(content_str(&delta).unwrap_or_default());
        (reasoning, content)
    }

    /// Assert that `wire` parses to `expected` whether it arrives in one piece,
    /// split in two at every byte, or one byte at a time.
    fn assert_split_invariant(
        make_parser: impl Fn() -> Ernie45ReasoningParser,
        wire: &str,
        expected: (&str, &str),
    ) {
        let expected = (expected.0.to_string(), expected.1.to_string());
        for split in 0..=wire.len() {
            assert_eq!(
                collect(&mut make_parser(), &[&wire[..split], &wire[split..]]),
                expected,
                "wire {wire:?}, split at {split}"
            );
        }
        let bytes = wire
            .as_bytes()
            .chunks(1)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            collect(&mut make_parser(), &bytes),
            expected,
            "wire {wire:?}, bytewise"
        );
    }

    const THINKING_OUTPUT: &str =
        "Need compute 2 + 2 directly.\n</think>\n<response>\nThe answer is 4.\n</response>\n";

    #[test]
    fn picks_up_prompt_start_boundary() {
        let mut parser = parser_in_reasoning();

        let delta = push_str(&mut parser, "implicit reasoning</think>answer");
        assert_eq!(reasoning_str(&delta), Some("implicit reasoning"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn respects_prompt_end_boundary() {
        let mut parser = parser_after_reasoning();

        let delta = push_str(&mut parser, "answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn strips_response_framing_after_prompt_end_boundary() {
        assert_split_invariant(
            parser_after_reasoning,
            "<response>\nanswer\n</response>\n",
            ("", "answer"),
        );
    }

    #[test]
    fn strips_think_and_response_framing_in_single_push() {
        let mut parser = parser_in_reasoning();

        let (reasoning, content) = collect(&mut parser, &[THINKING_OUTPUT]);
        assert_eq!(reasoning, "Need compute 2 + 2 directly.");
        assert_eq!(content, "The answer is 4.");
    }

    #[test]
    fn strips_framing_at_every_split() {
        assert_split_invariant(
            parser_in_reasoning,
            THINKING_OUTPUT,
            ("Need compute 2 + 2 directly.", "The answer is 4."),
        );
    }

    #[test]
    fn strips_extra_newlines_before_response_start() {
        // The model does not always reproduce the template's single newline
        // between `</think>` and `<response>`.
        assert_split_invariant(
            parser_in_reasoning,
            "reason\n</think>\n\n\n<response>\nanswer\n</response>\n",
            ("reason", "answer"),
        );
    }

    #[test]
    fn keeps_tool_calls_after_response_framing() {
        // The template renders tool calls after `</response>`; they must reach
        // the content stream (and thus the tool parser) untouched.
        let wire = "Need call the tools.\n</think>\n<response>\nI will call the tools.\n</response>\n\n\n<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1}}\n</tool_call>\n";
        assert_split_invariant(
            parser_in_reasoning,
            wire,
            (
                "Need call the tools.",
                "I will call the tools.<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1}}\n</tool_call>\n",
            ),
        );
    }

    #[test]
    fn passes_through_markers_inside_tool_call_arguments() {
        // Once the answer is closed, `<response>` literals in a tool call's JSON
        // are arguments, not framing.
        let wire = "reason\n</think>\n<response>\nanswer\n</response>\n\n\n<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"payload\": \"<response>hello</response>\"}}\n</tool_call>\n";
        assert_split_invariant(
            parser_in_reasoning,
            wire,
            (
                "reason",
                "answer<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"payload\": \"<response>hello</response>\"}}\n</tool_call>\n",
            ),
        );
    }

    #[test]
    fn passes_through_tools_only_turn_untouched() {
        // Without an answer, the template renders `</think>\n\n<tool_call>`, and
        // nothing in the tool call is framing: not even response markers.
        let wire = "reason\n</think>\n\n<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"payload\": \"<response>hello</response>\"}}\n</tool_call>\n";
        assert_split_invariant(
            parser_in_reasoning,
            wire,
            (
                "reason",
                "<tool_call>\n{\"name\": \"echo\", \"arguments\": {\"payload\": \"<response>hello</response>\"}}\n</tool_call>\n",
            ),
        );
    }

    #[test]
    fn handles_answer_without_response_framing() {
        // ERNIE also emits `abc\n</think>\ndef` without the response wrapper;
        // such content is passed through, response markers included.
        assert_split_invariant(
            parser_in_reasoning,
            "abc\n</think>\ndef\nDEF</response>",
            ("abc", "def\nDEF</response>"),
        );
    }

    #[test]
    fn drops_all_framing_newlines_after_think_end() {
        // The ERNIE-4.5-VL template renders `</think>\n\n` before content.
        let mut parser = parser_in_reasoning();

        let (reasoning, content) = collect(&mut parser, &["abc\n</think>\n\n", "def"]);
        assert_eq!(reasoning, "abc");
        assert_eq!(content, "def");
    }

    #[test]
    fn preserves_newlines_inside_reasoning_and_content() {
        // Only the newlines the template renders around the markers are
        // dropped: the single `\n` before `</think>` and `</response>`, the
        // single `\n` after `<response>`, and the runs after `</think>` and
        // `</response>`. The blank lines inside the answer survive.
        assert_split_invariant(
            parser_in_reasoning,
            "line1\n\nline2\n\n</think>\n<response>\n\npara1\n\npara2\n\n</response>\n",
            ("line1\n\nline2\n", "\npara1\n\npara2\n"),
        );
    }

    #[test]
    fn handles_empty_reasoning_at_every_split() {
        // A `<think>\n</think>` round-trip inside one chunk leaves no reasoning
        // text behind; the framing after it must still be recognized.
        assert_split_invariant(
            parser_before_reasoning,
            "<think>\n</think>\n\n<response>\nanswer\n</response>\n",
            ("", "answer"),
        );
        assert_split_invariant(
            parser_before_reasoning,
            "<think>\n</think>\n\nanswer",
            ("", "answer"),
        );
    }

    #[test]
    fn holds_trailing_newline_until_response_end_is_ruled_out() {
        let mut parser = parser_in_reasoning();

        let first = push_str(&mut parser, "reason\n</think>\n<response>\nanswer\n");
        assert_eq!(reasoning_str(&first), Some("reason"));
        assert_eq!(content_str(&first), Some("answer"));

        // The held `\n` is replayed once `</response>` does not follow.
        let second = push_str(&mut parser, "more answer");
        assert_eq!(content_str(&second), Some("\nmore answer"));
    }

    #[test]
    fn finish_flushes_partial_markers_as_content() {
        let mut parser = parser_in_reasoning();

        let pushed = push_str(&mut parser, "reason\n</think>\n<response>\nanswer\n</resp");
        assert_eq!(reasoning_str(&pushed), Some("reason"));
        assert_eq!(content_str(&pushed), Some("answer"));

        // Held-back text that never became a marker is literal content.
        let flushed = parser.finish().unwrap();
        assert_eq!(flushed.reasoning, None);
        assert_eq!(content_str(&flushed), Some("\n</resp"));
    }

    #[test]
    fn finish_flushes_partial_response_start_as_content() {
        let mut parser = parser_in_reasoning();

        let pushed = push_str(&mut parser, "reason\n</think>\n<resp");
        assert_eq!(reasoning_str(&pushed), Some("reason"));
        assert_eq!(pushed.content, None);

        let flushed = parser.finish().unwrap();
        assert_eq!(content_str(&flushed), Some("<resp"));
    }

    #[test]
    fn drops_newlines_after_response_end_at_finish() {
        // `</response>` is followed only by framing before `<|im_end|>`.
        assert_split_invariant(
            parser_in_reasoning,
            "reason\n</think>\n<response>\nanswer\n</response>\n\n",
            ("reason", "answer"),
        );
    }

    #[test]
    fn handles_empty_response_and_empty_input() {
        let mut parser = parser_in_reasoning();

        assert!(push_str(&mut parser, "").is_empty());
        let (reasoning, content) = collect(
            &mut parser,
            &["reason\n</think>\n<response>\n</response>\n"],
        );
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "");
    }

    #[test]
    fn handles_explicit_start_token() {
        // An explicit start delimiter must not leak into reasoning text.
        assert_split_invariant(
            parser_before_reasoning,
            "<think>\nreason\n</think>\n<response>\nanswer\n</response>",
            ("reason", "answer"),
        );
    }

    #[test]
    fn initialize_resets_the_response_frame() {
        let mut parser = parser_in_reasoning();
        let (_, content) = collect(&mut parser, &["reason\n</think>\n<response>\nanswer"]);
        assert_eq!(content, "answer");

        // A new stream must recognize the wrapper again.
        parser.initialize(&[THINK_START_ID, u32::from(b'\n')]).unwrap();
        let (reasoning, content) = collect(&mut parser, &[THINKING_OUTPUT]);
        assert_eq!(reasoning, "Need compute 2 + 2 directly.");
        assert_eq!(content, "The answer is 4.");
    }
}
