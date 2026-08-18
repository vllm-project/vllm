// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};

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
/// <tool_call>
/// {"name": ..., "arguments": {...}}
/// </tool_call>
/// ```
///
/// The `<think>` framing is handled by the shared delimited state machine. This
/// parser adds the `<response>` wrapper on top: the markers and the newlines
/// framing them are dropped from content, while anything after `</response>` is
/// passed through, so tool calls that follow the answer still reach the tool
/// parser.
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
    fn filter_content(
        &mut self,
        delta: ReasoningDelta,
        was_in_reasoning: bool,
        now_in_reasoning: bool,
    ) -> ReasoningDelta {
        // A `<think>...</think>` round-trip in one push still counts as a
        // transition: the inner emits reasoning while ending in content mode.
        let transitioned = !now_in_reasoning && (was_in_reasoning || delta.reasoning.is_some());

        // Templates may emit more newlines between `</think>` and `<response>`
        // than the single one the state machine already consumed.
        if transitioned {
            self.response_frame.expect_framing_newlines();
        }

        let content = delta.content.map(|content| self.response_frame.push(content));

        ReasoningDelta {
            reasoning: delta.reasoning,
            content: content.filter(|piece| !piece.is_empty()),
        }
    }
}

impl ReasoningParser for Ernie45ReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn initialize(&mut self, prompt_token_ids: &[u32]) -> Result<()> {
        self.inner.initialize(prompt_token_ids)
    }

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
        let was = self.inner.in_reasoning();
        let inner_delta = self.inner.push(delta);
        let now = self.inner.in_reasoning();
        Ok(self.filter_content(inner_delta, was, now))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let was = self.inner.in_reasoning();
        let inner_delta = self.inner.finish();
        let now = self.inner.in_reasoning();
        let mut delta = self.filter_content(inner_delta, was, now);

        // Held-back text that never completed a marker is literal content.
        delta.push_content(self.response_frame.finish());
        Ok(delta)
    }
}

/// Incremental filter that removes the `<response>`/`</response>` answer framing
/// from ERNIE 4.5 content.
///
/// Markers and their framing newlines may be split across pushes, so the filter
/// holds back any text that could still complete a marker, plus a single
/// trailing `\n` that could turn out to be the framing newline right before
/// `</response>`.
#[derive(Default)]
struct ResponseFrameFilter {
    /// Content held back until it can no longer complete a marker or the
    /// framing newline before `</response>`.
    pending: DecodedText,
    /// Whether newlines at the start of the next content are framing (right
    /// after `</think>`, `<response>`, or `</response>`) and should be dropped.
    strip_leading_newlines: bool,
}

impl ResponseFrameFilter {
    /// Treat newlines at the start of the upcoming content as framing.
    fn expect_framing_newlines(&mut self) {
        self.strip_leading_newlines = true;
    }

    /// Feed one content piece and return the content that is safe to emit.
    fn push(&mut self, content: DecodedText) -> DecodedText {
        self.pending.append(content);
        let mut emitted = DecodedText::default();

        loop {
            if self.strip_leading_newlines {
                let framing_len =
                    self.pending.text.len() - self.pending.text.trim_start_matches('\n').len();
                let _ = self.pending.drain_prefix(framing_len);
                if self.pending.text.is_empty() {
                    break;
                }
            }

            match earliest_marker(&self.pending.text) {
                Some((marker_start, marker)) => {
                    let mut before_len = marker_start;
                    if marker == RESPONSE_END && self.pending.text[..before_len].ends_with('\n') {
                        // The single newline right before `</response>` is framing.
                        before_len -= 1;
                    }
                    self.emit(before_len, &mut emitted);
                    let _ = self.pending.drain_prefix(marker_start - before_len + marker.len());
                    self.strip_leading_newlines = true;
                }
                None => {
                    // Keep back a possible partial marker, plus the `\n` right
                    // before it (or at the very end) that may precede
                    // `</response>`.
                    let mut stable_len =
                        self.pending.text.len() - partial_marker_suffix_len(&self.pending.text);
                    if self.pending.text[..stable_len].ends_with('\n') {
                        stable_len -= 1;
                    }
                    self.emit(stable_len, &mut emitted);
                    break;
                }
            }
        }

        emitted
    }

    /// Flush held-back content at end of stream.
    fn finish(&mut self) -> DecodedText {
        if self.strip_leading_newlines {
            let framing_len =
                self.pending.text.len() - self.pending.text.trim_start_matches('\n').len();
            let _ = self.pending.drain_prefix(framing_len);
        }
        self.strip_leading_newlines = false;
        self.pending.take()
    }

    /// Move the first `len` bytes of pending content into `emitted`, ending
    /// framing-newline stripping once any content has been emitted.
    fn emit(&mut self, len: usize, emitted: &mut DecodedText) {
        if len == 0 {
            return;
        }
        self.strip_leading_newlines = false;
        emitted.append(self.pending.drain_prefix(len));
    }
}

/// Find the earliest `<response>` or `</response>` marker in `text`.
fn earliest_marker(text: &str) -> Option<(usize, &'static str)> {
    [RESPONSE_START, RESPONSE_END]
        .into_iter()
        .filter_map(|marker| text.find(marker).map(|start| (start, marker)))
        .min_by_key(|(start, _)| *start)
}

/// Return the length of the longest trailing suffix of `text` that could still
/// complete a `<response>` or `</response>` marker.
fn partial_marker_suffix_len(text: &str) -> usize {
    text.char_indices()
        .map(|(idx, _)| &text[idx..])
        .filter(|suffix| {
            [RESPONSE_START, RESPONSE_END]
                .iter()
                .any(|marker| marker.len() > suffix.len() && marker.starts_with(suffix))
        })
        .map(str::len)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Ernie45ReasoningParser;
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{
        THINK_END_ID, THINK_START_ID, content_str, fake_tokenizer, push_str, reasoning_str,
    };
    use crate::tool::test_utils::split_by_chars;

    /// A parser whose prompt already opened reasoning, as the ERNIE templates do.
    fn parser_in_reasoning() -> Ernie45ReasoningParser {
        let mut parser = Ernie45ReasoningParser::new(Arc::new(fake_tokenizer())).unwrap();
        parser.initialize(&[THINK_START_ID, u32::from(b'\n')]).unwrap();
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
        let mut parser = Ernie45ReasoningParser::new(Arc::new(fake_tokenizer())).unwrap();
        // Prompt already closed reasoning with `</think>`.
        parser.initialize(&[THINK_START_ID, THINK_END_ID]).unwrap();

        let delta = push_str(&mut parser, "answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn strips_think_and_response_framing_in_single_push() {
        let mut parser = parser_in_reasoning();

        let (reasoning, content) = collect(&mut parser, &[THINKING_OUTPUT]);
        assert_eq!(reasoning, "Need compute 2 + 2 directly.");
        assert_eq!(content, "The answer is 4.");
    }

    #[test]
    fn strips_framing_across_arbitrary_chunk_boundaries() {
        for chunk_chars in 1..=12 {
            let mut parser = parser_in_reasoning();

            let (reasoning, content) =
                collect(&mut parser, &split_by_chars(THINKING_OUTPUT, chunk_chars));
            assert_eq!(
                reasoning, "Need compute 2 + 2 directly.",
                "chunk size {chunk_chars}"
            );
            assert_eq!(content, "The answer is 4.", "chunk size {chunk_chars}");
        }
    }

    #[test]
    fn keeps_tool_calls_after_response_framing() {
        // The template renders tool calls after `</response>`; they must reach
        // the content stream (and thus the tool parser) untouched.
        let output = "Need call the tools.\n</think>\n<response>\nI will call the tools.\n</response>\n\n\n<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1}}\n</tool_call>\n";
        for chunk_chars in [1, 3, 7, output.len()] {
            let mut parser = parser_in_reasoning();

            let (reasoning, content) = collect(&mut parser, &split_by_chars(output, chunk_chars));
            assert_eq!(
                reasoning, "Need call the tools.",
                "chunk size {chunk_chars}"
            );
            assert_eq!(
                content,
                "I will call the tools.<tool_call>\n{\"name\": \"add\", \"arguments\": {\"x\": 1}}\n</tool_call>\n",
                "chunk size {chunk_chars}"
            );
        }
    }

    #[test]
    fn handles_answer_without_response_framing() {
        // ERNIE also emits `abc\n</think>\ndef` without the response wrapper.
        let mut parser = parser_in_reasoning();

        let (reasoning, content) = collect(&mut parser, &["abc\n</think>\ndef\nDEF"]);
        assert_eq!(reasoning, "abc");
        assert_eq!(content, "def\nDEF");
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
        let mut parser = parser_in_reasoning();

        let (reasoning, content) = collect(
            &mut parser,
            &["line1\n\nline2\n\n</think>\n<response>\n\npara1\n\npara2\n\n</response>\n"],
        );
        // Only the framing newlines around the markers are dropped: the single
        // `\n` right before `</think>` / `</response>`, and the run of newlines
        // right after `</think>` / `<response>` / `</response>`.
        assert_eq!(reasoning, "line1\n\nline2\n");
        assert_eq!(content, "para1\n\npara2\n");
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
        let mut parser = Ernie45ReasoningParser::new(Arc::new(fake_tokenizer())).unwrap();

        // An explicit start delimiter must not leak into reasoning text.
        let (reasoning, content) = collect(
            &mut parser,
            &["<think>\nreason\n</think>\n<response>\nanswer\n</response>"],
        );
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer");
    }
}
