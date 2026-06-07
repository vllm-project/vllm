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
