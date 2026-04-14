use vllm_text::tokenizers::Tokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for legacy Kimi models that use Unicode thinking tags.
pub struct KimiReasoningParser {
    inner: DelimitedReasoningParser,
}

impl KimiReasoningParser {
    /// Create a Kimi parser backed by the shared delimited state machine.
    pub fn new(tokenizer: &dyn Tokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "◁think▷", "◁/think▷", false)?,
        })
    }
}

impl ReasoningParser for KimiReasoningParser {
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
