// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningError, ReasoningParser, Result};

/// Internal HY3 reasoning stage used by the unified HY3 parser.
pub(crate) struct HyV3ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl HyV3ReasoningParser {
    /// Create a HY3 reasoning parser for the tokenizer-specific marker suffix.
    pub(crate) fn new(tokenizer: DynTokenizer, suffix: &str) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParser::new(
                tokenizer,
                format!("<think{suffix}>"),
                format!("</think{suffix}>"),
                false,
            )?,
        })
    }
}

impl ReasoningParser for HyV3ReasoningParser {
    // Suffix discovery belongs to `HyV3UnifiedParser` so its reasoning and
    // tool delimiters always use the same tokenizer-derived value.
    fn create(_tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Err(ReasoningError::DummyUnifiedParser {
            name: "hy_v3".to_string(),
        })
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
