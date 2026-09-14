// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared text-generation support used by chat and future raw completions.
//!
//! This crate intentionally stays below chat semantics:
//! prompt text handling, tokenizer/model loading, incremental detokenization,
//! and the thin generate-facing backend interface live here.

use std::mem::take;

pub use backend::{
    DynTextBackend, GenerationConfigMode, SamplingHints, SamplingLimits, TextBackend,
};
pub use error::{Error, LogprobsError, Result, SamplingParamsError, TokenIdsError};
use futures::Stream;
pub use lower::{
    PreparedTextRequest, lower_sampling_params, lower_text_request, resolve_max_tokens,
};
pub use output::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
    DecodedText, DecodedTextEvent, DecodedTokenLogprob, Finished, SampledDelta, TextDecodeOptions,
    TextOutputStreamExt, TokenAnchor, TokenAttribution,
};
pub use request::{Prompt, SamplingParams, TextRequest, normalize_top_k};
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
mod truncation;
pub use truncation::{PromptTruncation, PromptTruncationLimit, TruncationSide};
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

    fn prepare_prompt_tokens(
        &self,
        prompt: Prompt,
        add_special_tokens: bool,
        prompt_truncation: Option<PromptTruncation>,
        max_output_tokens: Option<u32>,
    ) -> Result<Vec<u32>> {
        let tokenizer = self.backend.tokenizer();
        let mut prompt_token_ids = match prompt {
            Prompt::Text(text) => tokenizer.encode(&text, add_special_tokens)?,
            // Pre-tokenized prompts are the main completions-side escape hatch that lets benchmark
            // and infra workloads bypass chat rendering and tokenizer overhead entirely.
            Prompt::TokenIds(token_ids) => token_ids,
        };
        if let Some(prompt_truncation) = prompt_truncation {
            let input_budget = self.max_model_len.saturating_sub(max_output_tokens.unwrap_or(0));
            prompt_truncation.apply(&mut prompt_token_ids, input_budget)?;
        }
        Ok(prompt_token_ids)
    }

    /// Prepare one request's prompt tokens without generation-specific lowering.
    pub fn tokenize(&self, request: TextRequest) -> Result<Vec<u32>> {
        request.validate()?;
        self.prepare_prompt_tokens(
            request.prompt,
            request.add_special_tokens,
            request.prompt_truncation,
            request.sampling_params.max_tokens,
        )
    }

    /// Tokenize and lower one request without submitting it to an engine.
    pub fn prepare(&self, mut request: TextRequest) -> Result<PreparedTextRequest> {
        request.validate()?;

        if request.arrival_time.is_none() {
            request.arrival_time = Some(vllm_llm::current_unix_timestamp_secs());
        }

        let prompt_token_ids = self.prepare_prompt_tokens(
            take(&mut request.prompt),
            request.add_special_tokens,
            request.prompt_truncation,
            request.sampling_params.max_tokens,
        )?;
        let tokenizer = self.backend.tokenizer();
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
    use std::sync::Arc;

    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;

    #[derive(Debug)]
    struct TestTextBackend;

    impl TextBackend for TestTextBackend {
        fn tokenizer(&self) -> DynTokenizer {
            Arc::new(TestTokenizer::new())
        }

        fn model_id(&self) -> &str {
            "test-model"
        }
    }

    #[test]
    fn tokenize_and_prepare_apply_the_same_prompt_truncation() {
        let processor = TextRequestProcessor::new(Arc::new(TestTextBackend), 10);
        let request = TextRequest {
            prompt: Prompt::TokenIds(vec![1, 2, 3, 4, 5]),
            prompt_truncation: Some(PromptTruncation {
                limit: PromptTruncationLimit::Fixed(3),
                side: TruncationSide::Left,
            }),
            ..TextRequest::for_test()
        };

        assert_eq!(processor.tokenize(request.clone()).unwrap(), vec![3, 4, 5]);
        assert_eq!(
            processor.prepare(request).unwrap().generate_request.prompt_token_ids,
            vec![3, 4, 5]
        );
    }
}
