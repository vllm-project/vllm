// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{
    DelimitedReasoningParser, DelimitedReasoningParserBuilder, ReasoningDelta, ReasoningParser,
    Result,
};

/// Reasoning parser for Poolside V1 (Laguna) style outputs.
///
/// Uses the standard `<think>...</think>` delimiters with no fixed framing.
/// The Laguna chat template prefills the current turn with an explicit
/// boundary — `<assistant><think>` when thinking is enabled and
/// `<assistant></think>` when it is not — so the prompt-derived boundary
/// determines whether the completion starts inside a reasoning span. With no
/// boundary the shared state machine defaults to content.
pub struct PoolsideV1ReasoningParser {
    inner: DelimitedReasoningParser,
}

impl PoolsideV1ReasoningParser {
    /// Create a Poolside V1 parser backed by the shared delimited state machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        Ok(Self {
            inner: DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
                .build()?,
        })
    }
}

impl ReasoningParser for PoolsideV1ReasoningParser {
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
