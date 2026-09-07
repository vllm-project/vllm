// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

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
    pending_reasoning_newline: Option<DecodedText>,
    /// Last push ended on `</think>` without emitting content; the next
    /// content delta's leading `\n` should be dropped.
    just_ended_reasoning: bool,
}

impl Step3p5ReasoningParser {
    /// Create a Step3p5 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false)?,
            pending_reasoning_newline: None,
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
        if self.pending_reasoning_newline.is_some() {
            if let Some(reasoning) = inner_delta.reasoning.as_mut() {
                let mut held = self.pending_reasoning_newline.take().unwrap();
                held.append(std::mem::take(reasoning));
                *reasoning = held;
            } else if transitioned {
                // The held `\n` was the one right before `</think>`: drop it.
                self.pending_reasoning_newline = None;
            }
        }

        // Hold back a trailing reasoning `\n` until we know if `</think>` follows.
        if let Some(reasoning) = inner_delta.reasoning.as_mut()
            && reasoning.text.ends_with('\n')
        {
            let kept = reasoning.drain_prefix(reasoning.text.len() - 1);
            let newline = std::mem::replace(reasoning, kept);
            if !transitioned {
                self.pending_reasoning_newline = Some(newline);
            }
        }

        // Drop a leading `\n` of content emitted right after `</think>`.
        if let Some(content) = inner_delta.content.as_mut()
            && (transitioned || self.just_ended_reasoning)
            && content.text.starts_with('\n')
        {
            let _ = content.drain_prefix(1);
        }

        self.just_ended_reasoning = transitioned && inner_delta.content.is_none();

        if inner_delta.reasoning.as_ref().is_some_and(DecodedText::is_empty) {
            inner_delta.reasoning = None;
        }
        if inner_delta.content.as_ref().is_some_and(DecodedText::is_empty) {
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

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
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
        if let Some(held) = self.pending_reasoning_newline.take() {
            match delta.reasoning.as_mut() {
                Some(existing) => existing.append(held),
                None => delta.reasoning = Some(held),
            }
        }

        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Step3p5ReasoningParser;
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{
        THINK_START_ID, content_str, fake_tokenizer, push_str, reasoning_str,
    };

    #[test]
    fn picks_up_prompt_start_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();
        // Prompt prefills `<think>`, opening reasoning before the stream.
        parser.initialize(&[THINK_START_ID]).unwrap();

        let delta = push_str(
            &mut parser,
            "This is a reasoning section</think>This is the rest",
        );
        assert_eq!(reasoning_str(&delta), Some("This is a reasoning section"));
        assert_eq!(content_str(&delta), Some("This is the rest"));
    }

    #[test]
    fn handles_unterminated_reasoning() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let pushed = push_str(&mut parser, "<think>reason without end");
        assert_eq!(reasoning_str(&pushed), Some("reason without end"));
        assert_eq!(pushed.content, None);

        let flushed = parser.finish().unwrap();
        assert!(flushed.is_empty());
    }

    #[test]
    fn handles_empty_input() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let pushed = push_str(&mut parser, "");
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

        let delta = push_str(
            &mut parser,
            "\n This is a \n reasoning section\n\n\n</think>\n\nThis is the rest",
        );
        assert_eq!(
            reasoning_str(&delta),
            Some("\n This is a \n reasoning section\n\n")
        );
        assert_eq!(content_str(&delta), Some("\nThis is the rest"));
    }

    #[test]
    fn drops_framing_newlines_in_single_push() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "<think>reason\n</think>\nanswer");
        assert_eq!(reasoning_str(&delta), Some("reason"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn drops_framing_newlines_across_pushes() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        // The trailing `\n` from the first push is held until we know whether
        // `</think>` follows.
        let first = push_str(&mut parser, "<think>reason\n");
        assert_eq!(reasoning_str(&first), Some("reason"));
        assert_eq!(first.content, None);

        // `</think>` arrives standalone; the held newline should be dropped.
        let second = push_str(&mut parser, "</think>");
        assert!(second.is_empty());

        // The leading newline of the first content delta is dropped.
        let third = push_str(&mut parser, "\nanswer");
        assert_eq!(third.reasoning, None);
        assert_eq!(content_str(&third), Some("answer"));
    }

    #[test]
    fn replays_held_newline_when_more_reasoning_follows() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let first = push_str(&mut parser, "<think>reason\n");
        assert_eq!(reasoning_str(&first), Some("reason"));

        let second = push_str(&mut parser, "more reason");
        assert_eq!(reasoning_str(&second), Some("\nmore reason"));
        assert_eq!(second.content, None);
    }

    #[test]
    fn finish_flushes_held_newline_in_unterminated_stream() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let first = push_str(&mut parser, "<think>reason\n");
        assert_eq!(reasoning_str(&first), Some("reason"));

        let flushed = parser.finish().unwrap();
        assert_eq!(reasoning_str(&flushed), Some("\n"));
        assert_eq!(flushed.content, None);
    }

    #[test]
    fn preserves_inner_newlines_in_reasoning() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "<think>line1\nline2</think>tail");
        assert_eq!(reasoning_str(&delta), Some("line1\nline2"));
        assert_eq!(content_str(&delta), Some("tail"));
    }

    #[test]
    fn trims_only_one_trailing_reasoning_newline() {
        // Only the single framing newline immediately before `</think>` is
        // dropped; earlier newlines in the reasoning body are preserved.
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "<think>reason\n\n</think>answer");
        assert_eq!(reasoning_str(&delta), Some("reason\n"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn drops_only_first_content_newline_after_transition() {
        // The leading-`\n` drop applies only to the first content delta after
        // `</think>`; later deltas pass through untouched.
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let first = push_str(&mut parser, "<think>reason</think>");
        assert_eq!(reasoning_str(&first), Some("reason"));
        assert_eq!(first.content, None);

        let second = push_str(&mut parser, "\nfirst");
        assert_eq!(second.reasoning, None);
        assert_eq!(content_str(&second), Some("first"));

        // A `\n` arriving in a later content delta must NOT be dropped.
        let third = push_str(&mut parser, "\nsecond");
        assert_eq!(third.reasoning, None);
        assert_eq!(content_str(&third), Some("\nsecond"));
    }

    #[test]
    fn passes_through_clean_boundary_without_framing_newlines() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "<think>reason</think>tail");
        assert_eq!(reasoning_str(&delta), Some("reason"));
        assert_eq!(content_str(&delta), Some("tail"));
    }

    #[test]
    fn handles_empty_reasoning_section() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "<think></think>answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }
}
