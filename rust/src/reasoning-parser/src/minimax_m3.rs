use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for MiniMax M3 style outputs.
///
/// MiniMax M3 uses `<mm:think>...</mm:think>` delimiters. Its chat template may
/// prefill either delimiter depending on the requested thinking mode, so the
/// shared delimited parser derives the starting state from the rendered prompt.
pub struct MiniMaxM3ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl MiniMaxM3ReasoningParser {
    /// Create a MiniMax M3 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<mm:think>", "</mm:think>", false)?,
        })
    }
}

impl ReasoningParser for MiniMaxM3ReasoningParser {
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
