use vllm_text::tokenizers::Tokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for DeepSeek R1 style outputs.
///
/// DeepSeek R1 may begin generating directly inside a reasoning span and only
/// emit the closing `</think>` delimiter, so the no-boundary fallback defaults
/// to `in_reasoning = true`.
pub struct DeepSeekR1ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl DeepSeekR1ReasoningParser {
    /// Create a DeepSeek R1 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: &dyn Tokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", true)?,
        })
    }
}

impl ReasoningParser for DeepSeekR1ReasoningParser {
    fn create(tokenizer: &dyn Tokenizer) -> Result<Box<dyn ReasoningParser>>
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
