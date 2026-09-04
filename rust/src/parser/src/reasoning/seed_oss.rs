// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

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

    fn push(&mut self, delta: DecodedText) -> Result<ReasoningDelta> {
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
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{
        SEED_THINK_END_ID, SEED_THINK_START_ID, content_str, fake_tokenizer, push_str,
        reasoning_str,
    };

    #[test]
    fn without_prompt_markers_expects_start_token() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "implicit reasoning</seed:think>answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(
            content_str(&delta),
            Some("implicit reasoning</seed:think>answer")
        );
    }

    #[test]
    fn picks_up_prompt_start_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
        // Prompt prefills `<seed:think>`, opening reasoning before the stream.
        parser.initialize(&[SEED_THINK_START_ID]).unwrap();

        let delta = push_str(&mut parser, "reason</seed:think>answer");
        assert_eq!(reasoning_str(&delta), Some("reason"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn respects_prompt_end_boundary() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
        // Prompt already closed reasoning with `</seed:think>`.
        parser.initialize(&[SEED_THINK_END_ID]).unwrap();

        let delta = push_str(&mut parser, "answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn handles_explicit_start_token() {
        // An explicit start delimiter must not leak into reasoning text.
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "<seed:think>reason</seed:think>answer");
        assert_eq!(reasoning_str(&delta), Some("reason"));
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn streams_explicit_start_token_across_pushes() {
        // Start token, reasoning body, end token, and content arrive in separate
        // streaming deltas.
        let tokenizer = Arc::new(fake_tokenizer());
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
            let delta = push_str(&mut parser, delta_str);
            if let Some(r) = delta.reasoning {
                reasoning.push_str(&r.text);
            }
            if let Some(c) = delta.content {
                content.push_str(&c.text);
            }
        }
        assert_eq!(reasoning, "Some reasoning content");
        assert_eq!(content, "Final answer");
    }

    #[test]
    fn handles_partial_delimiters_across_pushes() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[SEED_THINK_START_ID]).unwrap();

        // Closing delimiter `</seed:think>` arrives in two halves.
        let first = push_str(&mut parser, "reason</seed:");
        assert_eq!(reasoning_str(&first), Some("reason"));
        assert_eq!(first.content, None);

        let second = push_str(&mut parser, "think>answer");
        assert_eq!(second.reasoning, None);
        assert_eq!(content_str(&second), Some("answer"));
    }
}
