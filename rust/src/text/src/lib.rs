#![feature(coroutines)]
#![feature(trait_alias)]
#![feature(iterator_try_collect)]

//! Shared text-generation support used by chat and future raw completions.
//!
//! This crate intentionally stays below chat semantics:
//! prompt text handling, tokenizer/model loading, incremental detokenization,
//! and the thin generate-facing backend interface live here.

use std::mem::take;

pub use backend::{DynTextBackend, SamplingHints, TextBackend};
pub use error::{Error, Result};
use futures::Stream;
pub use lower::{
    PreparedTextRequest, lower_sampling_params, lower_text_request, resolve_max_tokens,
};
pub use output::{
    CollectedTextOutput, DecodedLogprobs, DecodedPositionLogprobs, DecodedPromptLogprobs,
    DecodedTextEvent, DecodedTokenLogprob, Finished, TextDecodeOptions, TextOutputStreamExt,
};
pub use request::{Prompt, SamplingParams, TextRequest};
use vllm_engine_core_client::EngineCoreClient;
pub use vllm_llm::FinishReason;
use vllm_llm::Llm;

mod backend;
pub mod backends;
mod error;
mod incremental;
mod lower;
pub mod output;
mod request;

/// Shared streamed text output type used by raw completions and other text-only northbound paths.
pub trait TextOutputStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;

/// Raw text facade above [`Llm`].
///
/// This layer stays below chat semantics: prompt text or prompt token IDs flow in, decoded text
/// deltas and terminal metadata flow out.
pub struct TextLlm {
    /// Generate-only client owned by this text facade.
    llm: Llm,
    /// Tokenizer/model metadata backend responsible for prompt encode/decode and sampling hints.
    backend: DynTextBackend,
    /// Optional override for the backend-derived context window size.
    max_model_len: Option<u32>,
}

impl TextLlm {
    /// Create a new text-generation facade from a shared LLM client plus a text backend.
    pub fn new(llm: Llm, backend: DynTextBackend) -> Self {
        Self {
            llm,
            backend,
            max_model_len: None,
        }
    }

    /// Override the maximum model context length, taking priority over tokenizer/model metadata.
    pub fn with_max_model_len(mut self, max_model_len: u32) -> Self {
        self.max_model_len = Some(max_model_len);
        self
    }

    /// Return the backend model ID when available.
    pub fn model_id(&self) -> Option<&str> {
        self.backend.model_id()
    }

    /// Expose the underlying engine-core client for low-level utility/admin calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        self.llm.engine_core_client()
    }

    /// Tokenize if needed, lower to a generate request, and stream incrementally decoded text.
    pub async fn generate(&self, mut request: TextRequest) -> Result<impl TextOutputStream> {
        request.validate()?;

        let prompt_token_ids = match take(&mut request.prompt) {
            Prompt::Text(text) => self.backend.encode(&text)?,
            // Pre-tokenized prompts are the main completions-side escape hatch that lets benchmark
            // and infra workloads bypass chat rendering and tokenizer overhead entirely.
            Prompt::TokenIds(token_ids) => token_ids,
        };

        let mut sampling_hints = self.backend.sampling_hints()?;
        if let Some(max_model_len) = self.max_model_len {
            sampling_hints.max_model_len = Some(max_model_len);
        }
        let prepared = lower_text_request(request, prompt_token_ids, sampling_hints)?;
        let raw_stream = self.llm.generate(prepared.generate_request).await?;
        let decoded_stream = output::decoded_text_event_stream(
            prepared.text_request.request_id,
            self.backend.clone(),
            raw_stream,
            prepared.text_request.decode_options,
            prepared.text_request.intermediate,
        );

        Ok(decoded_stream)
    }

    /// Abort one in-flight request by request ID.
    pub async fn abort(&self, request_id: &str) -> Result<()> {
        self.llm.abort(request_id).await?;
        Ok(())
    }

    /// Shut down the underlying LLM client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.llm.shutdown().await?;
        Ok(())
    }
}
