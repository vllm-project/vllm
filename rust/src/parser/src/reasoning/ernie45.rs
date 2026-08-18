// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Marker opening the final answer in ERNIE 4.5 thinking-model output.
const RESPONSE_START: &str = "<response>";
/// Marker closing the final answer in ERNIE 4.5 thinking-model output.
const RESPONSE_END: &str = "</response>";

/// Reasoning parser for the ERNIE 4.5 thinking models.
///
/// ERNIE 4.5 uses standard `<think>`/`</think>` delimiters. Its chat templates
/// prefill `<think>\n` at the end of the generation prompt, so a stream usually
/// starts inside reasoning and the model only emits the closing `</think>`;
/// the no-boundary fallback therefore defaults to `in_reasoning = true`.
///
/// The templates frame both sections with newlines, and the
/// `ERNIE-4.5-21B-A3B-Thinking` template additionally wraps the final answer in
/// `<response>`/`</response>` before any `<tool_call>` blocks:
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
/// This parser strips exactly that framing so that only the reasoning body and
/// the answer text (plus any trailing tool-call blocks) reach the caller:
///
/// - the single `\n` right before `</think>` is dropped from reasoning;
/// - newlines directly after `</think>` are dropped from content;
/// - a `<response>` marker and the newlines directly after it are dropped;
/// - a `</response>` marker, the single `\n` right before it, and the newlines
///   directly after it are dropped.
///
/// Anything after `</response>` is passed through as content, so tool calls
/// that follow the answer are still visible to the tool parser.
pub struct Ernie45ReasoningParser {
    inner: DelimitedReasoningParser,
    /// `\n` at the end of the last reasoning delta, held in case `</think>`
    /// follows.
    pending_reasoning_newline: bool,
    /// Strips the `<response>`/`</response>` framing from content deltas.
    response_frame: ResponseFrameFilter,
}

impl Ernie45ReasoningParser {
    /// Create an ERNIE 4.5 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", true)?,
            pending_reasoning_newline: false,
            response_frame: ResponseFrameFilter::default(),
        })
    }

    /// Trim the reasoning framing newline and filter the content framing.
    fn process(
        &mut self,
        mut inner_delta: ReasoningDelta,
        was_in_reasoning: bool,
        now_in_reasoning: bool,
    ) -> ReasoningDelta {
        // A `<think>...</think>` round-trip in one push still counts as a
        // transition: the inner emits reasoning while ending in content mode.
        let transitioned =
            !now_in_reasoning && (was_in_reasoning || inner_delta.reasoning.is_some());

        // Replay or drop a previously-held trailing reasoning newline.
        if self.pending_reasoning_newline {
            if let Some(reasoning) = inner_delta.reasoning.as_mut() {
                reasoning.insert(0, '\n');
                self.pending_reasoning_newline = false;
            } else if transitioned {
                // The held `\n` was the one right before `</think>`: drop it.
                self.pending_reasoning_newline = false;
            }
        }

        // Hold back a trailing reasoning `\n` until we know if `</think>` follows.
        if let Some(reasoning) = inner_delta.reasoning.as_mut()
            && reasoning.ends_with('\n')
        {
            reasoning.pop();
            if !transitioned {
                self.pending_reasoning_newline = true;
            }
        }

        // Content right after `</think>` starts with framing newlines.
        if transitioned {
            self.response_frame.expect_framing_newlines();
        }
        let content =
            inner_delta.content.as_deref().map(|content| self.response_frame.push(content));

        ReasoningDelta {
            reasoning: inner_delta.reasoning.filter(|text| !text.is_empty()),
            content: content.filter(|text| !text.is_empty()),
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
        self.inner.initialize(prompt_token_ids);
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        let was = self.inner.in_reasoning();
        let inner_delta = self.inner.push(delta);
        let now = self.inner.in_reasoning();
        Ok(self.process(inner_delta, was, now))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        let was = self.inner.in_reasoning();
        let inner_delta = self.inner.finish();
        let now = self.inner.in_reasoning();
        let mut delta = self.process(inner_delta, was, now);

        // Emit a still-held newline rather than silently dropping it.
        if self.pending_reasoning_newline {
            match delta.reasoning.as_mut() {
                Some(existing) => existing.push('\n'),
                None => delta.reasoning = Some("\n".to_string()),
            }
            self.pending_reasoning_newline = false;
        }

        let content = self.response_frame.finish();
        if !content.is_empty() {
            delta.push_content(&content);
        }

        Ok(delta)
    }
}

/// Incremental filter that removes the `<response>`/`</response>` answer
/// framing from ERNIE 4.5 content.
///
/// Markers and their framing newlines may be split across pushes, so the filter
/// holds back any text that could still complete a marker, plus a single
/// trailing `\n` that could turn out to be the framing newline right before
/// `</response>`.
#[derive(Default)]
struct ResponseFrameFilter {
    /// Content held back until it can no longer complete a marker or the
    /// framing newline before `</response>`.
    pending: String,
    /// Whether newlines at the start of the next content are framing (right
    /// after `</think>`, `<response>`, or `</response>`) and should be dropped.
    strip_leading_newlines: bool,
}

impl ResponseFrameFilter {
    /// Treat newlines at the start of the upcoming content as framing.
    fn expect_framing_newlines(&mut self) {
        self.strip_leading_newlines = true;
    }

    /// Feed one content delta and return the content that is safe to emit.
    fn push(&mut self, content: &str) -> String {
        self.pending.push_str(content);
        let mut emitted = String::new();

        loop {
            if self.strip_leading_newlines {
                let framing_len = self.pending.len() - self.pending.trim_start_matches('\n').len();
                self.pending.drain(..framing_len);
                if self.pending.is_empty() {
                    break;
                }
            }

            match earliest_marker(&self.pending) {
                Some((marker_start, marker)) => {
                    let mut before_len = marker_start;
                    if marker == RESPONSE_END && self.pending[..before_len].ends_with('\n') {
                        // The single newline right before `</response>` is framing.
                        before_len -= 1;
                    }
                    self.emit(before_len, &mut emitted);
                    self.pending.drain(..marker_start + marker.len());
                    self.strip_leading_newlines = true;
                }
                None => {
                    // Keep back a possible partial marker, plus the `\n` right
                    // before it (or at the very end) that may precede
                    // `</response>`.
                    let mut stable_len =
                        self.pending.len() - partial_marker_suffix_len(&self.pending);
                    if self.pending[..stable_len].ends_with('\n') {
                        stable_len -= 1;
                    }
                    self.emit(stable_len, &mut emitted);
                    self.pending.drain(..stable_len);
                    break;
                }
            }
        }

        emitted
    }

    /// Flush held-back content at end of stream.
    fn finish(&mut self) -> String {
        if self.strip_leading_newlines {
            let framing_len = self.pending.len() - self.pending.trim_start_matches('\n').len();
            self.pending.drain(..framing_len);
        }
        let mut emitted = String::new();
        self.emit(self.pending.len(), &mut emitted);
        self.pending.clear();
        self.strip_leading_newlines = false;
        emitted
    }

    /// Append the first `len` bytes of pending content to `emitted`, ending
    /// framing-newline stripping once any content has been emitted.
    fn emit(&mut self, len: usize, emitted: &mut String) {
        if len == 0 {
            return;
        }
        self.strip_leading_newlines = false;
        emitted.push_str(&self.pending[..len]);
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
    use crate::reasoning::tests::{THINK_END_ID, THINK_START_ID, fake_tokenizer};
    use crate::reasoning::{ReasoningDelta, ReasoningParser};
    use crate::tool::test_utils::split_by_chars;

    fn parser() -> Ernie45ReasoningParser {
        Ernie45ReasoningParser::new(Arc::new(fake_tokenizer())).unwrap()
    }

    /// Push each chunk and concatenate the reasoning/content parts, including
    /// whatever `finish()` flushes.
    fn collect(parser: &mut Ernie45ReasoningParser, chunks: &[&str]) -> (String, String) {
        let mut reasoning = String::new();
        let mut content = String::new();
        let mut absorb = |delta: ReasoningDelta| {
            reasoning.push_str(delta.reasoning.as_deref().unwrap_or_default());
            content.push_str(delta.content.as_deref().unwrap_or_default());
        };
        for chunk in chunks {
            absorb(parser.push(chunk).unwrap());
        }
        absorb(parser.finish().unwrap());
        (reasoning, content)
    }

    const THINKING_OUTPUT: &str =
        "Need compute 2 + 2 directly.\n</think>\n<response>\nThe answer is 4.\n</response>\n";

    #[test]
    fn defaults_to_reasoning_without_prompt_markers() {
        // The ERNIE templates prefill `<think>`, so the model only emits `</think>`.
        let mut parser = parser();

        let delta = parser.push("implicit reasoning</think>answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("implicit reasoning"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn respects_prompt_end_boundary() {
        let mut parser = parser();
        // Prompt already closed reasoning with `</think>`.
        parser.initialize(&[THINK_START_ID, THINK_END_ID]).unwrap();

        let delta = parser.push("answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn strips_think_and_response_framing_in_single_push() {
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

        let (reasoning, content) = collect(&mut parser, &[THINKING_OUTPUT]);
        assert_eq!(reasoning, "Need compute 2 + 2 directly.");
        assert_eq!(content, "The answer is 4.");
    }

    #[test]
    fn strips_framing_across_arbitrary_chunk_boundaries() {
        for chunk_chars in 1..=12 {
            let mut parser = parser();
            parser.initialize(&[THINK_START_ID]).unwrap();

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
            let mut parser = parser();
            parser.initialize(&[THINK_START_ID]).unwrap();

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
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

        let (reasoning, content) = collect(&mut parser, &["abc\n</think>\ndef\nDEF"]);
        assert_eq!(reasoning, "abc");
        assert_eq!(content, "def\nDEF");
    }

    #[test]
    fn drops_all_framing_newlines_after_think_end() {
        // The ERNIE-4.5-VL template renders `</think>\n\n` before content.
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

        let (reasoning, content) = collect(&mut parser, &["abc\n</think>\n\n", "def"]);
        assert_eq!(reasoning, "abc");
        assert_eq!(content, "def");
    }

    #[test]
    fn preserves_newlines_inside_reasoning_and_content() {
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

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
    fn replays_held_newlines_when_more_text_follows() {
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

        let first = parser.push("reason\n").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));
        let second = parser.push("more\n</think>\n<response>\nanswer\n").unwrap();
        assert_eq!(second.reasoning.as_deref(), Some("\nmore"));
        assert_eq!(second.content.as_deref(), Some("answer"));

        // A trailing content newline is held until we know `</response>` does
        // not follow, then replayed.
        let third = parser.push("more answer").unwrap();
        assert_eq!(third.content.as_deref(), Some("\nmore answer"));
    }

    #[test]
    fn finish_flushes_held_newlines_and_partial_markers() {
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

        let pushed = parser.push("reason\n</think>\n<response>\nanswer\n</resp").unwrap();
        assert_eq!(pushed.reasoning.as_deref(), Some("reason"));
        assert_eq!(pushed.content.as_deref(), Some("answer"));

        // Held back text that never became a marker is literal content.
        let flushed = parser.finish().unwrap();
        assert_eq!(flushed.reasoning, None);
        assert_eq!(flushed.content.as_deref(), Some("\n</resp"));
    }

    #[test]
    fn finish_flushes_held_reasoning_newline_in_unterminated_stream() {
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

        let pushed = parser.push("reason\n").unwrap();
        assert_eq!(pushed.reasoning.as_deref(), Some("reason"));

        let flushed = parser.finish().unwrap();
        assert_eq!(flushed.reasoning.as_deref(), Some("\n"));
        assert_eq!(flushed.content, None);
    }

    #[test]
    fn handles_empty_response_and_empty_input() {
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID]).unwrap();

        assert!(parser.push("").unwrap().is_empty());
        let (reasoning, content) = collect(
            &mut parser,
            &["reason\n</think>\n<response>\n</response>\n"],
        );
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "");
    }

    #[test]
    fn handles_explicit_start_token() {
        let mut parser = parser();
        parser.initialize(&[THINK_START_ID, THINK_END_ID]).unwrap();

        // An explicit start delimiter must not leak into reasoning text.
        let (reasoning, content) = collect(
            &mut parser,
            &["<think>reason\n</think>\n<response>\nanswer\n</response>"],
        );
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer");
    }
}
