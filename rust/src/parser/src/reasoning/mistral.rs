use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningParser, Result};

/// Reasoning parser for Mistral models that wrap reasoning in `[THINK]` /
/// `[/THINK]`, such as Magistral.
///
/// `[THINK]` is optional, so the no-boundary fallback starts in content
/// (`in_reasoning = false`). The delimiters are Mistral special tokens, so
/// `preserve_special_tokens` returns `true` to keep them in the decoded text.
///
/// Python reference:
/// <https://github.com/vllm-project/vllm/blob/6f573f486/vllm/reasoning/mistral_reasoning_parser.py>
pub struct MistralReasoningParser {
    inner: DelimitedReasoningParser,
}

impl MistralReasoningParser {
    /// Create a Mistral parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "[THINK]", "[/THINK]", false)?,
        })
    }
}

impl ReasoningParser for MistralReasoningParser {
    fn create(tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tokenizer)?))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
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
