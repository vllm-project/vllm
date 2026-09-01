// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningError, ReasoningParser, Result};

/// Internal HY reasoning stage used by the unified HY parsers.
pub(crate) struct HyReasoningParser {
    inner: DelimitedReasoningParser,
}

impl HyReasoningParser {
    /// Create a HY reasoning parser for the tokenizer-specific marker suffix.
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

impl ReasoningParser for HyReasoningParser {
    // Suffix discovery belongs to the unified HY parser so its reasoning and
    // tool delimiters always use the same tokenizer-derived value.
    fn create(_tokenizer: DynTokenizer) -> Result<Box<dyn ReasoningParser>>
    where
        Self: Sized + 'static,
    {
        Err(ReasoningError::DummyUnifiedParser {
            name: "hy".to_string(),
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
