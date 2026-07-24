// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::DynTokenizer;

use super::{DelimitedReasoningParser, ReasoningDelta, ReasoningError, ReasoningParser, Result};

/// Reasoning parser for Poolside V1 style outputs.
///
/// Uses the standard `<think>...</think>` delimiters, but scopes the prompt
/// boundary scan to the current assistant turn: the scan starts after the last
/// `<assistant>` start-of-message token. A `</think>` in a prior user turn,
/// few-shot example or tool description is no longer visible, so it cannot be
/// mistaken for a template-injected "thinking already ended" marker that would
/// disable reasoning parsing for the whole response.
///
/// Like DeepSeek R1, the model may begin generating directly inside a
/// reasoning span and only emit the closing `</think>` delimiter, so the
/// no-boundary fallback defaults to `in_reasoning = true`.
pub struct PoolsideV1ReasoningParser {
    inner: DelimitedReasoningParser,
    assistant_token_id: u32,
}

impl PoolsideV1ReasoningParser {
    const START_OF_ASSISTANT_MESSAGE: &'static str = "<assistant>";

    /// Create a Poolside V1 parser backed by the shared delimited state
    /// machine.
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        let assistant_token_id = tokenizer
            .token_to_id(Self::START_OF_ASSISTANT_MESSAGE)
            .ok_or_else(|| ReasoningError::MissingToken {
                token: Self::START_OF_ASSISTANT_MESSAGE.to_string(),
            })?;

        Ok(Self {
            inner: DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", true)?,
            assistant_token_id,
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
        // Scope the boundary scan to the current assistant turn: tokens before
        // the last `<assistant>` belong to prior conversation and are ignored.
        let scan_start = prompt_token_ids
            .iter()
            .rposition(|&id| id == self.assistant_token_id)
            .map_or(0, |idx| idx + 1);
        self.inner.initialize(&prompt_token_ids[scan_start..]);
        Ok(())
    }

    fn push(&mut self, delta: &str) -> Result<ReasoningDelta> {
        Ok(self.inner.push(delta))
    }

    fn finish(&mut self) -> Result<ReasoningDelta> {
        Ok(self.inner.finish())
    }
}
