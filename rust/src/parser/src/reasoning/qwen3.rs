// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};

/// Reasoning parser for the Qwen3/Qwen3.5 family.
///
/// This parser uses standard `<think>...</think>` delimiters.
pub struct Qwen3ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl Qwen3ReasoningParser {
    /// Create a Qwen3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
                .with_after_start("\n")
                .with_before_end("\n")
                .with_after_end("\n\n")
                .build()?,
        })
    }
}

impl ReasoningParser for Qwen3ReasoningParser {
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
        Ok(self.inner.push(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Qwen3ReasoningParser;
    use crate::reasoning::ReasoningParser;
    use crate::reasoning::tests::{
        SPECIAL_BOUNDARY_ID, THINK_END_ID, THINK_START_ID, content_str, fake_tokenizer, push_str,
        reasoning_str,
    };

    #[test]
    fn qwen3_without_prompt_markers_expects_start_token() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "reason</think>answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("reason</think>answer"));
    }

    #[test]
    fn qwen3_prompt_end_marker_starts_in_content() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
        parser.initialize(&[THINK_END_ID]).unwrap();

        let delta = push_str(&mut parser, "answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }

    #[test]
    fn qwen3_tolerates_old_and_new_formats() {
        let tokenizer = Arc::new(fake_tokenizer());

        let mut old_parser = Qwen3ReasoningParser::new(tokenizer.clone()).unwrap();
        let old = push_str(&mut old_parser, "<think>reason</think>answer");
        assert_eq!(reasoning_str(&old), Some("reason"));
        assert_eq!(content_str(&old), Some("answer"));

        let mut new_parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
        new_parser.initialize(&[THINK_START_ID]).unwrap();
        let new = push_str(&mut new_parser, "reason</think>answer");
        assert_eq!(reasoning_str(&new), Some("reason"));
        assert_eq!(content_str(&new), Some("answer"));
    }

    #[test]
    fn qwen3_stops_scanning_at_last_special_token() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

        parser.initialize(&[THINK_START_ID, SPECIAL_BOUNDARY_ID]).unwrap();

        let delta = push_str(&mut parser, "answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("answer"));
    }
}
