// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::Usage;

/// Tracks cumulative token counts for OpenAI streaming chunks.
///
/// This helper is intentionally only a counter. Callers decide whether to
/// attach `counts()` to each streamed data chunk, while final usage-only chunks
/// should still be built from the authoritative terminal `TokenUsage`.
#[derive(Debug, Clone, Default)]
pub(crate) struct ContinuousUsage {
    prompt_tokens: usize,
    output_tokens: usize,
}

impl ContinuousUsage {
    /// Record the prompt-token count reported when a stream starts.
    pub(crate) fn set_prompt_tokens(&mut self, prompt_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
    }

    /// Add newly decoded output tokens to the running completion count.
    pub(crate) fn add_output_tokens(&mut self, output_tokens: usize) {
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
    }

    /// Replace the running counts with the final counts reported by generation.
    pub(crate) fn set_final_counts(&mut self, prompt_tokens: usize, output_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
        self.output_tokens = output_tokens;
    }

    /// Build a streaming usage snapshot without prompt cache details.
    pub(crate) fn to_usage(&self) -> Usage {
        Usage::from_counts(self.prompt_tokens, self.output_tokens, None)
    }
}
