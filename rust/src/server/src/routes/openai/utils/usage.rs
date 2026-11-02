// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_llm::TokenUsage;

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
    /// Running reasoning-token count, following the decoder clock. `None`
    /// when the route does not count reasoning tokens.
    reasoning_tokens: Option<usize>,
}

impl ContinuousUsage {
    /// Create a tracker; with `count_reasoning_tokens`, reasoning tokens are
    /// counted from zero.
    pub(crate) fn new(count_reasoning_tokens: bool) -> Self {
        Self {
            reasoning_tokens: count_reasoning_tokens.then_some(0),
            ..Self::default()
        }
    }

    /// Record the prompt-token count reported when a stream starts.
    pub(crate) fn set_prompt_tokens(&mut self, prompt_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
    }

    /// Add newly decoded output tokens to the running completion count.
    pub(crate) fn add_output_tokens(&mut self, output_tokens: usize) {
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
    }

    /// Add the tokens attributed to one reasoning delta to the running count.
    pub(crate) fn add_reasoning_tokens(&mut self, token_count: usize) {
        if let Some(reasoning_tokens) = &mut self.reasoning_tokens {
            *reasoning_tokens = reasoning_tokens.saturating_add(token_count);
        }
    }

    /// Converge the running reasoning count onto the authoritative terminal
    /// value reported at stream end.
    pub(crate) fn set_reasoning_tokens(&mut self, reasoning_tokens: usize) {
        if let Some(count) = &mut self.reasoning_tokens {
            *count = reasoning_tokens;
        }
    }

    /// Replace the running counts with the final counts reported by generation.
    pub(crate) fn set_final_counts(&mut self, prompt_tokens: usize, output_tokens: usize) {
        self.prompt_tokens = prompt_tokens;
        self.output_tokens = output_tokens;
    }

    /// Build a streaming usage snapshot without prompt cache details.
    pub(crate) fn to_usage(&self) -> Usage {
        let usage = TokenUsage {
            prompt_token_count: self.prompt_tokens,
            output_token_count: self.output_tokens,
            prompt_cache_usage: Default::default(),
        };
        Usage::from_token_usage(usage, false, self.reasoning_tokens)
    }
}
