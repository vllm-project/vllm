// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared text-generation support used by chat and future raw completions.
//!
//! This crate intentionally stays below chat semantics:
//! prompt text handling, tokenizer/model loading, incremental detokenization,
//! and the thin generate-facing backend interface live here.

use std::mem::take;

pub use backend::{DynTextBackend, SamplingHints, SamplingLimits, TextBackend};
pub use error::{Error, LogprobsError, Result, SamplingParamsError, TokenIdsError};
use futures::Stream;
pub use lower::{
    PreparedTextRequest, lower_sampling_params, lower_text_request, resolve_max_tokens,
};
pub use output::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
    DecodedTextEvent, DecodedTokenLogprob, Finished, TextDecodeOptions, TextOutputStreamExt,
};
pub use request::{Prompt, SamplingParams, TextRequest, TruncationSide};
use trait_set::trait_set;
use vllm_engine_core_client::EngineCoreClient;
pub use vllm_llm::FinishReason;
use vllm_llm::{GenerateOutputStream, Llm};
use vllm_tokenizer::DynTokenizer;

pub mod backend;
mod error;
mod lower;
pub mod output;
mod request;
pub use vllm_tokenizer as tokenizer;

trait_set! {
    /// Shared streamed text output type used by raw completions and other text-only northbound paths.
    pub trait TextOutputStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
}

/// Text request preparation shared by inference and render-only frontends.
pub struct TextRequestProcessor {
    /// Tokenizer/model metadata backend responsible for prompt encode/decode
    /// and sampling hints.
    backend: DynTextBackend,
    /// Runtime context window size reported by the engine startup handshake.
    /// Render-only frontends supply the downstream engine's effective value.
    max_model_len: u32,
    /// Maximum number of top log probabilities accepted by this text facade.
    max_logprobs: i32,
}

impl TextRequestProcessor {
    /// Create a processor with the effective model context length.
    pub fn new(backend: DynTextBackend, max_model_len: u32) -> Self {
        Self {
            backend,
            max_model_len,
            max_logprobs: SamplingLimits::DEFAULT_MAX_LOGPROBS,
        }
    }

    /// Override the maximum accepted logprobs count.
    pub fn with_max_logprobs(mut self, max_logprobs: Option<i32>) -> Self {
        if let Some(max_logprobs) = max_logprobs {
            self.max_logprobs = max_logprobs;
        }
        self
    }

    /// Return the tokenizer used by this processor.
    pub fn tokenizer(&self) -> DynTokenizer {
        self.backend.tokenizer()
    }

    /// Return the effective model context length.
    pub fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn tokenize_prompt(
        tokenizer: &DynTokenizer,
        prompt: Prompt,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>> {
        match prompt {
            Prompt::Text(text) => tokenizer.encode(&text, add_special_tokens).map_err(Into::into),
            // Pre-tokenized prompts are the main completions-side escape hatch that lets benchmark
            // and infra workloads bypass chat rendering and tokenizer overhead entirely.
            Prompt::TokenIds(token_ids) => Ok(token_ids),
        }
    }

    /// Tokenize one request without generation-specific lowering.
    pub fn tokenize(&self, request: TextRequest) -> Result<Vec<u32>> {
        request.validate()?;
        Self::tokenize_prompt(
            &self.backend.tokenizer(),
            request.prompt,
            request.add_special_tokens,
        )
    }

    /// Tokenize and lower one request without submitting it to an engine.
    pub fn prepare(&self, mut request: TextRequest) -> Result<PreparedTextRequest> {
        request.validate()?;

        if request.arrival_time.is_none() {
            request.arrival_time = Some(vllm_llm::current_unix_timestamp_secs());
        }

        let tokenizer = self.backend.tokenizer();
        let mut prompt_token_ids = Self::tokenize_prompt(
            &tokenizer,
            take(&mut request.prompt),
            request.add_special_tokens,
        )?;

        apply_truncate_prompt_tokens(
            &mut prompt_token_ids,
            request.truncate_prompt_tokens,
            request.truncation_side,
            request.sampling_params.max_tokens.unwrap_or(0),
            self.max_model_len,
        )?;
        let sampling_hints = self.backend.sampling_hints()?;
        let sampling_limits = SamplingLimits {
            max_model_len: self.max_model_len,
            max_logprobs: self.max_logprobs,
            model_vocab_size: self.backend.model_vocab_size(),
            tokenizer_vocab_size: self.backend.tokenizer_vocab_size(),
        };

        lower_text_request(
            request,
            prompt_token_ids,
            sampling_hints,
            sampling_limits,
            tokenizer.as_ref(),
        )
    }
}

pub(crate) fn apply_truncate_prompt_tokens(
    prompt_token_ids: &mut Vec<u32>,
    truncate_prompt_tokens: Option<i64>,
    truncation_side: Option<request::TruncationSide>,
    max_output_tokens: u32,
    max_model_len: u32,
) -> Result<()> {
    let Some(truncate_prompt_tokens) = truncate_prompt_tokens else {
        return Ok(());
    };

    // Defensive guard: request.validate() should catch this first, but
    // apply_truncate_prompt_tokens is pub(crate) and may be called independently.
    if truncate_prompt_tokens < -1 {
        return Ok(()); // validated upstream; silently skip if somehow missed
    }

    let budget = max_model_len.saturating_sub(max_output_tokens);
    let max_input_tokens = if truncate_prompt_tokens == -1 {
        budget as usize
    } else {
        let max_input_tokens = truncate_prompt_tokens as usize;
        if max_input_tokens > budget as usize {
            return Err(Error::TruncatePromptTokensExceedsBudget {
                value: truncate_prompt_tokens,
                budget,
            });
        }
        max_input_tokens
    };

    if prompt_token_ids.len() > max_input_tokens {
        let side = truncation_side.unwrap_or(request::TruncationSide::Left);
        if side == request::TruncationSide::Left {
            let start = prompt_token_ids.len() - max_input_tokens;
            prompt_token_ids.drain(0..start);
        } else {
            prompt_token_ids.truncate(max_input_tokens);
        }
    }

    Ok(())
}

/// Raw text facade above [`Llm`].
///
/// This layer stays below chat semantics: prompt text or prompt token IDs flow
/// in, decoded text deltas and terminal metadata flow out.
pub struct TextLlm {
    /// Generate-only client owned by this text facade.
    llm: Llm,
    /// Shared engine-free request preparation.
    processor: TextRequestProcessor,
}

impl TextLlm {
    /// Create a new text-generation facade from a shared LLM client plus a text
    /// backend.
    pub fn new(llm: Llm, backend: DynTextBackend) -> Self {
        // The engine-reported value reflects the post-profiling, auto-fitted
        // KV cache limit used at runtime.
        let max_model_len = llm.engine_core_client().max_model_len();

        Self {
            llm,
            processor: TextRequestProcessor::new(backend, max_model_len),
        }
    }

    /// Override the maximum accepted logprobs count.
    pub fn with_max_logprobs(mut self, max_logprobs: Option<i32>) -> Self {
        self.processor = self.processor.with_max_logprobs(max_logprobs);
        self
    }

    /// Return the backend model ID.
    pub fn model_id(&self) -> &str {
        self.processor.backend.model_id()
    }

    /// Expose the underlying engine-core client for low-level utility/admin
    /// calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        self.llm.engine_core_client()
    }

    /// Return the text request processor.
    pub fn request_processor(&self) -> &TextRequestProcessor {
        &self.processor
    }

    /// Return the tokenizer used by this text backend.
    pub fn tokenizer(&self) -> DynTokenizer {
        self.processor.tokenizer()
    }

    /// Tokenizer vocabulary size (the number of tokens the tokenizer knows),
    /// used to bound `allowed_token_ids` like the Python frontend `len(tokenizer)`.
    pub fn tokenizer_vocab_size(&self) -> usize {
        self.processor.backend.tokenizer_vocab_size()
    }

    /// Model vocabulary size from the model config, used to bound generated
    /// token IDs and logits-domain sampling controls.
    pub fn model_vocab_size(&self) -> usize {
        self.processor.backend.model_vocab_size()
    }

    /// Tokenize if needed, lower to a generate request, and return the raw
    /// token stream.
    pub async fn generate_raw(&self, request: TextRequest) -> Result<GenerateOutputStream> {
        let (_, raw_stream) = self.generate_inner(request).await?;
        Ok(raw_stream)
    }

    /// Tokenize if needed, lower to a generate request, and stream
    /// incrementally decoded text.
    pub async fn generate(&self, request: TextRequest) -> Result<impl TextOutputStream> {
        let (text_request, raw_stream) = self.generate_inner(request).await?;
        let tokenizer = self.processor.tokenizer();
        let decoded_stream = output::decoded_text_event_stream(
            text_request.request_id,
            tokenizer,
            raw_stream,
            text_request.decode_options,
            text_request.intermediate,
        );

        Ok(decoded_stream)
    }

    async fn generate_inner(
        &self,
        request: TextRequest,
    ) -> Result<(TextRequest, GenerateOutputStream)> {
        let PreparedTextRequest {
            text_request,
            generate_request,
        } = self.processor.prepare(request)?;

        let raw_stream = self.llm.generate(generate_request).await?;
        Ok((text_request, raw_stream))
    }

    /// Abort in-flight requests by their external (user-supplied) request ids.
    pub async fn abort(&self, external_ids: &[String]) -> Result<()> {
        self.llm.abort(external_ids).await?;
        Ok(())
    }

    /// Shut down the underlying LLM client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.llm.shutdown().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::TruncationSide;

    fn apply(
        prompt: &mut Vec<u32>,
        truncate: Option<i64>,
        side: Option<TruncationSide>,
        max_output: u32,
        model_len: u32,
    ) -> Result<()> {
        apply_truncate_prompt_tokens(prompt, truncate, side, max_output, model_len)
    }

    #[test]
    fn test_truncate_none() {
        let mut p = vec![1, 2, 3];
        apply(&mut p, None, None, 10, 100).unwrap();
        assert_eq!(p, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_right() {
        let mut p = vec![1, 2, 3, 4, 5];
        apply(&mut p, Some(3), Some(TruncationSide::Right), 10, 100).unwrap();
        assert_eq!(p, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_left() {
        let mut p = vec![1, 2, 3, 4, 5];
        apply(&mut p, Some(3), Some(TruncationSide::Left), 10, 100).unwrap();
        assert_eq!(p, vec![3, 4, 5]);
    }

    #[test]
    fn test_truncate_default_side_is_left() {
        let mut p = vec![1, 2, 3, 4, 5];
        apply(&mut p, Some(3), None, 10, 100).unwrap();
        assert_eq!(p, vec![3, 4, 5]);
    }

    #[test]
    fn test_truncate_zero() {
        let mut p = vec![1, 2, 3];
        apply(&mut p, Some(0), None, 10, 100).unwrap();
        assert!(p.is_empty());
    }

    #[test]
    fn test_truncate_noop_when_value_ge_prompt() {
        let mut p = vec![1, 2, 3];
        apply(&mut p, Some(10), None, 10, 100).unwrap();
        assert_eq!(p, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_sentinel_uses_budget() {
        let mut p = vec![0; 100];
        apply(&mut p, Some(-1), None, 30, 100).unwrap();
        assert_eq!(p.len(), 70); // 100 - 30
    }

    #[test]
    fn test_truncate_exceeds_budget() {
        let mut p = vec![1, 2, 3];
        let err = apply(&mut p, Some(80), None, 30, 100).unwrap_err();
        match err {
            Error::TruncatePromptTokensExceedsBudget { value, budget } => {
                assert_eq!(value, 80);
                assert_eq!(budget, 70);
            }
            _ => panic!("unexpected error"),
        }
    }

    #[test]
    fn test_truncate_less_than_minus_one_is_noop_in_apply() {
        // apply_truncate_prompt_tokens skips < -1 values since request.validate()
        // is the authoritative gate. Verify no mutation occurs.
        let mut p = vec![1, 2, 3];
        apply(&mut p, Some(-2), None, 30, 100).unwrap();
        assert_eq!(
            p,
            vec![1, 2, 3],
            "prompt must not be mutated for invalid sentinel"
        );
    }

    #[test]
    fn test_truncate_sentinel_with_saturating_budget_zero() {
        // If max_output_tokens >= max_model_len, budget saturates to 0.
        // Sentinel -1 then resolves to 0, draining the entire prompt.
        let mut p = vec![1, 2, 3];
        apply(&mut p, Some(-1), None, 200, 100).unwrap();
        assert!(p.is_empty(), "saturating budget should drain all tokens");
    }

    #[test]
    fn test_truncate_exact_boundary_noop() {
        // Value equals prompt length exactly: no truncation needed.
        let mut p = vec![1, 2, 3];
        apply(&mut p, Some(3), None, 10, 100).unwrap();
        assert_eq!(p, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_i64_max_hits_budget_check() {
        // i64::MAX as usize is valid on 64-bit; must not panic, must return budget error.
        let mut p = vec![1, 2, 3];
        let err = apply(&mut p, Some(i64::MAX), None, 30, 100).unwrap_err();
        assert!(matches!(
            err,
            Error::TruncatePromptTokensExceedsBudget { .. }
        ));
    }
}
