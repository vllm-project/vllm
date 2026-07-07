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

/// Raw text facade above [`Llm`].
///
/// This layer stays below chat semantics: prompt text or prompt token IDs flow
/// in, decoded text deltas and terminal metadata flow out.
pub struct TextLlm {
    /// Generate-only client owned by this text facade.
    llm: Llm,
    /// Tokenizer/model metadata backend responsible for prompt encode/decode
    /// and sampling hints.
    backend: DynTextBackend,
    /// Runtime context window size reported by the engine startup handshake.
    max_model_len: u32,
    /// Maximum number of top log probabilities accepted by this text facade.
    max_logprobs: i32,
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

    /// Return the backend model ID.
    pub fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    /// Expose the underlying engine-core client for low-level utility/admin
    /// calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        self.llm.engine_core_client()
    }

    /// Return the tokenizer used by this text backend.
    pub fn tokenizer(&self) -> DynTokenizer {
        self.backend.tokenizer()
    }

    /// Tokenizer vocabulary size (the number of tokens the tokenizer knows),
    /// used to bound `allowed_token_ids` like the Python frontend `len(tokenizer)`.
    pub fn tokenizer_vocab_size(&self) -> usize {
        self.backend.tokenizer_vocab_size()
    }

    /// Model vocabulary size from the model config, used to bound generated
    /// token IDs and logits-domain sampling controls.
    pub fn model_vocab_size(&self) -> usize {
        self.backend.model_vocab_size()
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
        let tokenizer = self.backend.tokenizer();
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
        mut request: TextRequest,
    ) -> Result<(TextRequest, GenerateOutputStream)> {
        request.validate()?;

        let tokenizer = self.backend.tokenizer();
        let prompt_token_ids = match take(&mut request.prompt) {
            Prompt::Text(text) => tokenizer.encode(&text, request.add_special_tokens)?,
            // Pre-tokenized prompts are the main completions-side escape hatch that lets benchmark
            // and infra workloads bypass chat rendering and tokenizer overhead entirely.
            Prompt::TokenIds(token_ids) => token_ids,
        };

        let prompt_token_ids = apply_truncate_prompt_tokens(
            prompt_token_ids,
            request.truncate_prompt_tokens,
            request.truncation_side,
            self.max_model_len,
            request.sampling_params.max_tokens,
        )?;

        let sampling_hints = self.backend.sampling_hints()?;
        let sampling_limits = SamplingLimits {
            max_model_len: self.max_model_len,
            max_logprobs: self.max_logprobs,
            model_vocab_size: self.backend.model_vocab_size(),
            tokenizer_vocab_size: self.backend.tokenizer_vocab_size(),
        };

        let PreparedTextRequest {
            text_request,
            generate_request,
        } = lower_text_request(
            request,
            prompt_token_ids,
            sampling_hints,
            sampling_limits,
            &*tokenizer,
        )?;

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

/// Truncate `prompt_token_ids` to `truncate_prompt_tokens` according to
/// `truncation_side`, mirroring the Python `TokenizeParams._token_truncation`
/// contract.
///
/// `None` means no truncation. `Some(-1)` is a sentinel that resolves to the
/// input-token budget `max_model_len - max_tokens`; any other negative value
/// is rejected upstream by [`TextRequest::validate`]. A non-negative value
/// that is at least as large as the prompt is a no-op.
///
/// `truncation_side` defaults to [`TruncationSide::Left`] when unset.
/// Python falls back to the tokenizer's own `truncation_side` attribute when
/// the request leaves it unset, and `vllm/tokenizers/registry.py`
/// initializes generate / draft tokenizers with `truncation_side = "left"`,
/// so we hard-default to left here for parity. The Rust `Tokenizer` trait
/// does not yet expose the per-tokenizer attribute; once it does, this can
/// fall back through that path. Callers that need right truncation must
/// request it explicitly via `truncation_side: "right"`.
fn apply_truncate_prompt_tokens(
    prompt_token_ids: Vec<u32>,
    truncate_prompt_tokens: Option<i64>,
    truncation_side: Option<TruncationSide>,
    max_model_len: u32,
    max_tokens: Option<u32>,
) -> Result<Vec<u32>> {
    let Some(value) = truncate_prompt_tokens else {
        return Ok(prompt_token_ids);
    };

    let max_tokens = max_tokens.unwrap_or(0);
    let max_input_tokens = max_model_len.saturating_sub(max_tokens);

    let target = if value == -1 {
        max_input_tokens as usize
    } else if value >= 0 {
        let value_u32: u32 = value
            .try_into()
            // u32::MAX is far larger than any plausible context window; clamp
            // through max_input_tokens below for the actual safety check.
            .unwrap_or(u32::MAX);
        if value_u32 > max_input_tokens {
            return Err(Error::TruncatePromptTokensExceedsBudget {
                value,
                max_input_tokens,
                max_model_len,
                max_tokens,
            });
        }
        value_u32 as usize
    } else {
        // `validate()` rejects values < -1, so this branch is unreachable for
        // a validated request. Treat it as a no-op rather than panicking if a
        // caller skips validation.
        return Ok(prompt_token_ids);
    };

    if target >= prompt_token_ids.len() {
        return Ok(prompt_token_ids);
    }

    let side = truncation_side.unwrap_or(TruncationSide::Left);
    let mut prompt_token_ids = prompt_token_ids;
    match side {
        TruncationSide::Right => prompt_token_ids.truncate(target),
        TruncationSide::Left => {
            let drop = prompt_token_ids.len() - target;
            prompt_token_ids.drain(..drop);
        }
    }
    Ok(prompt_token_ids)
}

#[cfg(test)]
mod truncate_tests {
    use super::*;

    fn ids(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn truncate_none_is_no_op() {
        let out = apply_truncate_prompt_tokens(ids(10), None, None, 100, Some(10)).unwrap();
        assert_eq!(out, ids(10));
    }

    #[test]
    fn truncate_right_keeps_prefix() {
        let out = apply_truncate_prompt_tokens(
            ids(10),
            Some(4),
            Some(TruncationSide::Right),
            100,
            Some(0),
        )
        .unwrap();
        assert_eq!(out, vec![0, 1, 2, 3]);
    }

    #[test]
    fn truncate_left_keeps_suffix() {
        let out = apply_truncate_prompt_tokens(
            ids(10),
            Some(4),
            Some(TruncationSide::Left),
            100,
            Some(0),
        )
        .unwrap();
        assert_eq!(out, vec![6, 7, 8, 9]);
    }

    #[test]
    fn truncate_default_side_is_left() {
        // Matches the generate-tokenizer default in
        // `vllm/tokenizers/registry.py`, which sets `truncation_side = "left"`
        // for the `generate` / `draft` runner types.
        let out = apply_truncate_prompt_tokens(ids(6), Some(3), None, 100, Some(0)).unwrap();
        assert_eq!(out, vec![3, 4, 5]);
    }

    #[test]
    fn truncate_zero_returns_empty_prompt() {
        let out = apply_truncate_prompt_tokens(ids(4), Some(0), None, 100, Some(0)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn truncate_request_exceeding_prompt_is_no_op() {
        let out = apply_truncate_prompt_tokens(ids(4), Some(20), None, 100, Some(0)).unwrap();
        assert_eq!(out, ids(4));
    }

    #[test]
    fn truncate_sentinel_minus_one_uses_input_budget() {
        // max_model_len = 100, max_tokens = 30 -> input budget = 70.
        // Prompt has 80 tokens -> truncate to 70, dropping the prefix because
        // the generate-tokenizer default is left truncation.
        let out = apply_truncate_prompt_tokens(ids(80), Some(-1), None, 100, Some(30)).unwrap();
        assert_eq!(out.len(), 70);
        assert_eq!(out[0], 10);
        assert_eq!(out[69], 79);
    }

    #[test]
    fn truncate_sentinel_minus_one_left_keeps_suffix() {
        let out = apply_truncate_prompt_tokens(
            ids(80),
            Some(-1),
            Some(TruncationSide::Left),
            100,
            Some(30),
        )
        .unwrap();
        assert_eq!(out.len(), 70);
        assert_eq!(out[0], 10);
        assert_eq!(out[69], 79);
    }

    #[test]
    fn truncate_value_above_input_budget_rejected() {
        let err = apply_truncate_prompt_tokens(ids(10), Some(80), None, 100, Some(30)).unwrap_err();
        assert!(
            matches!(
                err,
                Error::TruncatePromptTokensExceedsBudget { value: 80, .. }
            ),
            "expected TruncatePromptTokensExceedsBudget, got {err:?}",
        );
    }

    #[test]
    fn truncate_sentinel_when_max_tokens_unset_uses_full_context() {
        let out = apply_truncate_prompt_tokens(ids(80), Some(-1), None, 50, None).unwrap();
        assert_eq!(out.len(), 50);
    }

    #[test]
    fn truncate_invalid_negative_value_passes_through_validate_layer() {
        // `validate()` is responsible for rejecting -2 and below before the
        // truncation helper runs; if it ever slips through, the helper must
        // not panic and should fall back to a no-op.
        let out = apply_truncate_prompt_tokens(ids(5), Some(-2), None, 100, Some(0)).unwrap();
        assert_eq!(out, ids(5));
    }
}
