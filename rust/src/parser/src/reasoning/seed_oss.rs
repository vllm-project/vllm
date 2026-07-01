use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for SeedOSS models using `<seed:think>`/`</seed:think>`
/// delimiters.
pub struct SeedOssReasoningParser {
    inner: DelimitedReasoningParser,
}

impl SeedOssReasoningParser {
    /// Create a SeedOSS parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(
                tokenizer,
                "<seed:think>",
                "</seed:think>",
                false,
            )?,
        })
    }
}

impl ReasoningParser for SeedOssReasoningParser {
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
        Ok(self.inner.push(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::SeedOssReasoningParser;
    use crate::reasoning::{ReasoningParser, tests::FakeTokenizer};

    #[test]
    fn without_prompt_markers_expects_start_token() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("implicit reasoning</seed:think>answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(
            delta.content.as_deref(),
            Some("implicit reasoning</seed:think>answer")
        );
    }

    #[test]
    fn picks_up_prompt_start_boundary() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
        // Prompt prefills `<seed:think>` (id 10), opening reasoning before the stream.
        parser.initialize(&[10]).unwrap();

        let delta = parser.push("reason</seed:think>answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn respects_prompt_end_boundary() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
        // Prompt already closed reasoning with `</seed:think>` (id 11).
        parser.initialize(&[11]).unwrap();

        let delta = parser.push("answer").unwrap();
        assert_eq!(delta.reasoning, None);
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn handles_explicit_start_token() {
        // An explicit start delimiter must not leak into reasoning text.
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

        let delta = parser.push("<seed:think>reason</seed:think>answer").unwrap();
        assert_eq!(delta.reasoning.as_deref(), Some("reason"));
        assert_eq!(delta.content.as_deref(), Some("answer"));
    }

    #[test]
    fn streams_explicit_start_token_across_pushes() {
        // Start token, reasoning body, end token, and content arrive in separate
        // streaming deltas.
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

        let mut reasoning = String::new();
        let mut content = String::new();
        for delta_str in [
            "<seed:think>",
            "Some ",
            "reasoning ",
            "content",
            "</seed:think>",
            "Final ",
            "answer",
        ] {
            let delta = parser.push(delta_str).unwrap();
            if let Some(r) = delta.reasoning {
                reasoning.push_str(&r);
            }
            if let Some(c) = delta.content {
                content.push_str(&c);
            }
        }
        assert_eq!(reasoning, "Some reasoning content");
        assert_eq!(content, "Final answer");
    }

    #[test]
    fn handles_partial_delimiters_across_pushes() {
        let tokenizer = Arc::new(FakeTokenizer);
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[10]).unwrap();

        // Closing delimiter `</seed:think>` arrives in two halves.
        let first = parser.push("reason</seed:").unwrap();
        assert_eq!(first.reasoning.as_deref(), Some("reason"));
        assert_eq!(first.content, None);

        let second = parser.push("think>answer").unwrap();
        assert_eq!(second.reasoning, None);
        assert_eq!(second.content.as_deref(), Some("answer"));
    }
}
