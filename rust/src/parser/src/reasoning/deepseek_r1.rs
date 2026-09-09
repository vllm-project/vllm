// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for DeepSeek R1 style outputs.
pub struct DeepSeekR1ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl DeepSeekR1ReasoningParser {
    /// Create a DeepSeek R1 parser backed by the shared delimited state
    /// machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>")?,
        })
    }
}

impl ReasoningParser for DeepSeekR1ReasoningParser {
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

    use super::DeepSeekR1ReasoningParser;
    use crate::reasoning::tests::{content_str, fake_tokenizer, push_str};

    #[test]
    fn deepseek_r1_without_prompt_markers_expects_start_token() {
        let tokenizer = Arc::new(fake_tokenizer());
        let mut parser = DeepSeekR1ReasoningParser::new(tokenizer).unwrap();

        let delta = push_str(&mut parser, "reason</think>answer");
        assert_eq!(delta.reasoning, None);
        assert_eq!(content_str(&delta), Some("reason</think>answer"));
    }
}
