// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for Step3p5 outputs.
///
/// Step3p5 uses standard `<think>`/`</think>` delimiters but emits a `\n`
/// immediately before and/or after `</think>`. The parser drops these framing
/// newlines on both sides of the boundary, holding a trailing `\n` from
/// reasoning across pushes until either more reasoning text or `</think>`
/// arrives, and dropping a leading `\n` from the first content delta after
/// the boundary.
pub struct Step3p5ReasoningParser {
    inner: DelimitedReasoningParser,
    /// `\n` at end of last reasoning delta, held in case `</think>` follows.
    pending_reasoning_newline: bool,
    /// Last push ended on `</think>` without emitting content; the next
    /// content delta's leading `\n` should be dropped.
    just_ended_reasoning: bool,
}

impl Step3p5ReasoningParser {
    /// Create a Step3p5 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false)?,
            pending_reasoning_newline: false,
            just_ended_reasoning: false,
        })
    }

    /// Drop framing newlines around `</think>` and track held-newline state.
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

        // Drop a leading `\n` of content emitted right after `</think>`.
        if let Some(content) = inner_delta.content.as_mut()
            && (transitioned || self.just_ended_reasoning)
            && content.starts_with('\n')
        {
            content.remove(0);
        }

        self.just_ended_reasoning = transitioned && inner_delta.content.is_none();

        if inner_delta.reasoning.as_deref() == Some("") {
            inner_delta.reasoning = None;
        }
        if inner_delta.content.as_deref() == Some("") {
            inner_delta.content = None;
        }

        inner_delta
    }
}

impl ReasoningParser for Step3p5ReasoningParser {
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

        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Step3p5ReasoningParser;
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{THINK_START_ID, fake_tokenizer};

    #[test]
    fn picks_up_prompt_start_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();
        // Prompt prefills `<think>`, opening reasoning before the stream.
        parser.initialize(&[THINK_START_ID]).unwrap();

        let delta = parser.push("This is a reasoning section</think>This is the rest").unwrap();
        assert_eq!(
            delta.reasoning.as_deref(),
            Some("This is a reasoning section")
        );
        assert_eq!(delta.content.as_deref(), Some("This is the rest"));
    }

    #[test]
    fn handles_unterminated_reasoning() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let pushed = parser.push("<think>reason without end").unwrap();
        assert_eq!(pushed.reasoning.as_deref(), Some("reason without end"));
        assert_eq!(pushed.content, None);

        let flushed = parser.finish().unwrap();
        assert!(flushed.is_empty());
    }

    #[test]
    fn handles_empty_input() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let pushed = parser.push("").unwrap();
        assert!(pushed.is_empty());
        let flushed = parser.finish().unwrap();
        assert!(flushed.is_empty());
    }

    #[test]
    fn complex_newline_pattern_trims_only_single_framing_newline_each_side() {
        // Only the immediately-adjacent framing `\n` is dropped on each side of
        // `</think>`; surrounding newlines remain part of reasoning/content.
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[THINK_START_ID]).unwrap();

        let delta = parser
            .push("\n This is a \n reasoning section\n\n\n</think>\n\nThis is the rest")
            .unwrap();
        assert_eq!(
            delta.reasoning.as_deref(),
            Some("\n This is a \n reasoning section\n\n")
        );
        assert_eq!(delta.content.as_deref(), Some("\nThis is the rest"));
    }

    #[test]
    fn drops_framing_newlines_in_single_push() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("<think>reason\n</think>\nanswer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn drops_framing_newlines_across_pushes() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        // The trailing `\n` from the first push is held until we know whether
        // `</think>` follows.
        let first = parser.push("<think>reason\n").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));
        assert_eq!(first.content, None);

        // `</think>` arrives standalone; the held newline should be dropped.
        let second = parser.push("</think>").unwrap();
        assert!(second.is_empty());

        // The leading newline of the first content delta is dropped.
        let third = parser.push("\nanswer").unwrap();
        assert_eq!(third.reasoning, None);
        assert_eq!(third.content.as_deref(), Some("answer"));
    }

    #[test]
    fn replays_held_newline_when_more_reasoning_follows() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let first = parser.push("<think>reason\n").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));

        let second = parser.push("more reason").unwrap();
        assert_eq!(second.reasoning.as_deref(), Some("\nmore reason"));
        assert_eq!(second.content, None);
    }

    #[test]
    fn finish_flushes_held_newline_in_unterminated_stream() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let first = parser.push("<think>reason\n").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));

        let flushed = parser.finish().unwrap();
        assert_eq!(flushed.reasoning.as_deref(), Some("\n"));
        assert_eq!(flushed.content, None);
    }

    #[test]
    fn preserves_inner_newlines_in_reasoning() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("<think>line1\nline2</think>tail").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("line1\nline2"));
        assert_eq!(delta.content.as_deref(), Some("tail"));
    }

    #[test]
    fn trims_only_one_trailing_reasoning_newline() {
        // Only the single framing newline immediately before `</think>` is
        // dropped; earlier newlines in the reasoning body are preserved.
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("<think>reason\n\n</think>answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason\n"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn drops_only_first_content_newline_after_transition() {
        // The leading-`\n` drop applies only to the first content delta after
        // `</think>`; later deltas pass through untouched.
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let first = parser.push("<think>reason</think>").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));
        assert_eq!(first.content, None);

        let second = parser.push("\nfirst").unwrap();
        assert_eq!(second.reasoning, None);
        assert_eq!(second.content.as_deref(), Some("first"));

        // A `\n` arriving in a later content delta must NOT be dropped.
        let third = parser.push("\nsecond").unwrap();
        assert_eq!(third.reasoning, None);
        assert_eq!(third.content.as_deref(), Some("\nsecond"));
    }

    #[test]
    fn passes_through_clean_boundary_without_framing_newlines() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("<think>reason</think>tail").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("tail"));
    }

    #[test]
    fn handles_empty_reasoning_section() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("<think></think>answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }
}
