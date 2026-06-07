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
