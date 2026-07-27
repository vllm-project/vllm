// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared text-generation support used by chat and future raw completions.
//!
//! This crate intentionally stays below chat semantics:
//! prompt text handling, tokenizer/model loading, incremental detokenization,
//! and the thin generate-facing backend interface live here.

use std::mem::take;

pub use backend::{DynTextBackend, SamplingHints, SamplingLimits, TextBackend};
pub use error::{Error, LogprobsError, Result, TokenIdsError};
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

    /// Tokenize and lower one request without submitting it to an engine.
    pub fn prepare(&self, mut request: TextRequest) -> Result<PreparedTextRequest> {
        request.validate()?;

        if request.arrival_time.is_none() {
            request.arrival_time = Some(vllm_llm::current_unix_timestamp_secs());
        }

        let tokenizer = self.backend.tokenizer();
        let prompt_token_ids = match take(&mut request.prompt) {
            Prompt::Text(text) => tokenizer.encode(&text, request.add_special_tokens)?,
            // Pre-tokenized prompts are the main completions-side escape hatch that lets benchmark
            // and infra workloads bypass chat rendering and tokenizer overhead entirely.
            Prompt::TokenIds(token_ids) => token_ids,
        };
        let prompt_token_ids = truncate_prompt_token_ids(
            prompt_token_ids,
            request.truncate_prompt_tokens,
            request.truncation_side,
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

/// Apply `truncate_prompt_tokens` to a tokenized prompt.
///
/// `None` means no truncation, `-1` maps to the model context length, and any
/// other value keeps that many tokens from the side named by `truncation_side`.
/// Truncation runs after tokenization so pre-tokenized prompts are covered too,
/// and before length validation so a truncated prompt is measured at its final
/// length.
fn truncate_prompt_token_ids(
    mut prompt_token_ids: Vec<u32>,
    truncate_prompt_tokens: Option<i64>,
    truncation_side: Option<TruncationSide>,
    max_model_len: u32,
) -> Result<Vec<u32>> {
    let Some(truncate_prompt_tokens) = truncate_prompt_tokens else {
        return Ok(prompt_token_ids);
    };

    let keep = match truncate_prompt_tokens {
        -1 => max_model_len as usize,
        n if (0..=i64::from(max_model_len)).contains(&n) => n as usize,
        n => {
            return Err(Error::TruncatePromptTokensTooLarge {
                max_model_len,
                truncate_prompt_tokens: n,
            });
        }
    };

    if prompt_token_ids.len() > keep {
        match truncation_side.unwrap_or_default() {
            TruncationSide::Right => prompt_token_ids.truncate(keep),
            TruncationSide::Left => {
                prompt_token_ids.drain(..prompt_token_ids.len() - keep);
            }
        }
    }

    Ok(prompt_token_ids)
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

    const MAX_MODEL_LEN: u32 = 8;

    fn truncate(
        ids: &[u32],
        truncate_prompt_tokens: Option<i64>,
        truncation_side: Option<TruncationSide>,
    ) -> Result<Vec<u32>> {
        truncate_prompt_token_ids(
            ids.to_vec(),
            truncate_prompt_tokens,
            truncation_side,
            MAX_MODEL_LEN,
        )
    }

    #[test]
    fn truncate_prompt_token_ids_keeps_prompt_without_truncation() {
        assert_eq!(truncate(&[1, 2, 3], None, None).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn truncate_prompt_token_ids_defaults_to_keeping_the_first_tokens() {
        assert_eq!(truncate(&[1, 2, 3, 4], Some(2), None).unwrap(), vec![1, 2]);
        assert_eq!(
            truncate(&[1, 2, 3, 4], Some(2), Some(TruncationSide::Right)).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn truncate_prompt_token_ids_keeps_the_last_tokens_when_truncating_left() {
        assert_eq!(
            truncate(&[1, 2, 3, 4], Some(2), Some(TruncationSide::Left)).unwrap(),
            vec![3, 4]
        );
    }

    #[test]
    fn truncate_prompt_token_ids_leaves_shorter_prompts_alone() {
        assert_eq!(truncate(&[1, 2], Some(4), None).unwrap(), vec![1, 2]);
    }

    #[test]
    fn truncate_prompt_token_ids_maps_negative_one_to_the_context_length() {
        let ids: Vec<u32> = (0..12).collect();
        let truncated = truncate(&ids, Some(-1), None).unwrap();
        assert_eq!(truncated.len(), MAX_MODEL_LEN as usize);
        assert_eq!(truncated.first(), Some(&0));
    }

    #[test]
    fn truncate_prompt_token_ids_rejects_sizes_above_the_context_length() {
        let err = truncate(&[1, 2, 3], Some(i64::from(MAX_MODEL_LEN) + 1), None).unwrap_err();
        assert!(
            matches!(err, Error::TruncatePromptTokensTooLarge { .. }),
            "unexpected error: {err:?}"
        );
    }
}
