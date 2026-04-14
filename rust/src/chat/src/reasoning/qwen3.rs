use vllm_text::tokenizers::Tokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for the Qwen3/Qwen3.5 family.
///
/// This parser uses standard `<think>...</think>` delimiters and defaults to
/// waiting for an explicit start token when prompt initialization finds no
/// reasoning boundary.
pub struct Qwen3ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl Qwen3ReasoningParser {
    /// Create a Qwen3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: &dyn Tokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false)?,
        })
    }
}

impl ReasoningParser for Qwen3ReasoningParser {
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
