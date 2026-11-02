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
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputBackend;
pub use vllm_llm::FinishReason;
use vllm_llm::{GenerateOutputStream, Llm};
use vllm_tokenizer::{DynTokenizer, EncodedPrompt};

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
    /// Backend that compiles every structured-output constraint.
    structured_outputs_backend: StructuredOutputBackend,
}

impl TextRequestProcessor {
    /// Create a processor with the effective model context length.
    pub fn new(backend: DynTextBackend, max_model_len: u32) -> Self {
        Self {
            backend,
            max_model_len,
            max_logprobs: SamplingLimits::DEFAULT_MAX_LOGPROBS,
            structured_outputs_backend: StructuredOutputBackend::default(),
        }
    }

    /// Override the maximum accepted logprobs count.
    pub fn with_max_logprobs(mut self, max_logprobs: Option<i32>) -> Self {
        if let Some(max_logprobs) = max_logprobs {
            self.max_logprobs = max_logprobs;
        }
        self
    }

    /// Override the structured-output backend, xgrammar by default.
    pub fn with_structured_outputs_backend(mut self, backend: StructuredOutputBackend) -> Self {
        self.structured_outputs_backend = backend;
        self
    }

    /// Return the backend that compiles every structured-output constraint.
    pub fn structured_outputs_backend(&self) -> StructuredOutputBackend {
        self.structured_outputs_backend
    }

    /// Return the tokenizer used by this processor.
    pub fn tokenizer(&self) -> DynTokenizer {
        self.backend.tokenizer()
    }

    /// Return the effective model context length.
    pub fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    /// Tokenize and truncate the prompt in place, recording token offsets when
    /// requested. Running it again on the result is a no-op.
    fn prepare_prompt_tokens(&self, request: &mut TextRequest) -> Result<()> {
        // Pre-tokenized prompts are the main completions-side escape hatch that lets benchmark
        // and infra workloads bypass chat rendering and tokenizer overhead entirely.
        if let Prompt::Text(text) = &request.prompt {
            let tokenizer = self.backend.tokenizer();
            let EncodedPrompt {
                token_ids,
                token_offsets,
            } = if request.return_token_offsets && request.mm_features.is_none() {
                tokenizer.encode_with_offsets(text, request.add_special_tokens)?
            } else {
                EncodedPrompt {
                    token_ids: tokenizer.encode(text, request.add_special_tokens)?,
                    token_offsets: None,
                }
            };
            request.prompt = Prompt::TokenIds(token_ids);
            request.prompt_token_offsets = token_offsets;
        }
        let Prompt::TokenIds(token_ids) = &mut request.prompt else {
            unreachable!("prompt was just replaced with token IDs");
        };
        if let Some(truncation) = request.prompt_truncation.take() {
            truncation.apply(token_ids, self.max_model_len)?;
            if let Some(offsets) = &mut request.prompt_token_offsets {
                truncation.apply(offsets, self.max_model_len)?;
            }
        }
        Ok(())
    }

    /// Tokenize and truncate the prompt in place, so that later preparation
    /// reuses the final token IDs instead of tokenizing again.
    ///
    /// Frontends that need the final prompt tokens before submission (for
    /// example, to initialize a prompt-aware output parser) call this first
    /// and then hand the same request to [`TextLlm::generate`].
    pub fn tokenize_in_place<'r>(&self, request: &'r mut TextRequest) -> Result<&'r [u32]> {
        request.validate()?;
        self.prepare_prompt_tokens(request)?;
        match &request.prompt {
            Prompt::TokenIds(prompt_token_ids) => Ok(prompt_token_ids),
            Prompt::Text(_) => unreachable!("prompt was just replaced with token IDs"),
        }
    }

    /// Prepare one request's prompt tokens without generation-specific lowering.
    pub fn tokenize(&self, mut request: TextRequest) -> Result<Vec<u32>> {
        request.validate()?;
        self.prepare_prompt_tokens(&mut request)?;
        match request.prompt {
            Prompt::TokenIds(prompt_token_ids) => Ok(prompt_token_ids),
            Prompt::Text(_) => unreachable!("prompt was just replaced with token IDs"),
        }
    }

    /// Tokenize and lower one request without submitting it to an engine.
    pub fn prepare(&self, mut request: TextRequest) -> Result<PreparedTextRequest> {
        request.validate()?;

        if request.arrival_time.is_none() {
            request.arrival_time = Some(vllm_llm::current_unix_timestamp_secs());
        }

        self.prepare_prompt_tokens(&mut request)?;
        let Prompt::TokenIds(prompt_token_ids) = take(&mut request.prompt) else {
            unreachable!("prompt was just replaced with token IDs");
        };
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
            self.structured_outputs_backend,
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

    /// Override the structured-output backend, xgrammar by default.
    pub fn with_structured_outputs_backend(mut self, backend: StructuredOutputBackend) -> Self {
        self.processor = self.processor.with_structured_outputs_backend(backend);
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

    /// `truncate_prompt_tokens=-1` keeps `max_model_len - 1` prompt tokens
    /// whether or not `max_tokens` is set, and generation gets the one token
    /// left. Without `max_tokens`, truncating to the full context used to fail
    /// the request as too long.
    #[test]
    fn auto_truncation_leaves_one_output_token() {
        let processor = TextRequestProcessor::new(Arc::new(TestTextBackend), 10);
        for max_tokens in [None, Some(5)] {
            let request = TextRequest {
                prompt: Prompt::TokenIds((0..15).collect()),
                prompt_truncation: Some(PromptTruncation {
                    limit: PromptTruncationLimit::InputBudget,
                    side: TruncationSide::Left,
                }),
                sampling_params: SamplingParams {
                    max_tokens,
                    ..SamplingParams::default()
                },
                ..TextRequest::for_test()
            };

            let generate_request = processor.prepare(request).unwrap().generate_request;
            assert_eq!(
                generate_request.prompt_token_ids,
                (6..15).collect::<Vec<_>>()
            );
            assert_eq!(generate_request.sampling_params.max_tokens, 1);
        }
    }

    #[test]
    fn token_offsets_survive_truncation_and_repeated_preparation() {
        let processor = TextRequestProcessor::new(Arc::new(TestTextBackend), 4);
        for (side, expected_ids, expected_offsets) in [
            (
                TruncationSide::Left,
                vec![3, 4, 5],
                vec![(2, 3), (3, 4), (4, 5)],
            ),
            (
                TruncationSide::Right,
                vec![1, 2, 3],
                vec![(0, 1), (1, 2), (2, 3)],
            ),
        ] {
            let mut request = TextRequest {
                prompt: Prompt::TokenIds(vec![1, 2, 3, 4, 5]),
                prompt_token_offsets: Some((0..5).map(|i| (i, i + 1)).collect()),
                prompt_truncation: Some(PromptTruncation {
                    limit: PromptTruncationLimit::InputBudget,
                    side,
                }),
                sampling_params: SamplingParams {
                    max_tokens: Some(2),
                    ..Default::default()
                },
                ..TextRequest::for_test()
            };
            processor.tokenize_in_place(&mut request).unwrap();
            let prepared = processor.prepare(request).unwrap();
            assert_eq!(prepared.generate_request.prompt_token_ids, expected_ids);
            assert_eq!(
                prepared.text_request.prompt_token_offsets,
                Some(expected_offsets.clone())
            );
        }
    }
}
