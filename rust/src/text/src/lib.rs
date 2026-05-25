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
    /// Context window size derived by the backend or from engine startup
    /// handshake, with optional override from config.
    max_model_len: Option<u32>,
}

impl TextLlm {
    /// Create a new text-generation facade from a shared LLM client plus a text
    /// backend.
    pub fn new(llm: Llm, backend: DynTextBackend) -> Self {
        // Prefer the engine-reported max_model_len because it reflects the
        // post-profiling, auto-fitted KV cache limit rather than static
        // frontend metadata.
        let max_model_len = llm.engine_core_client().max_model_len();

        Self {
            llm,
            backend,
            max_model_len,
        }
    }

    /// Override the maximum model context length explicitly.
    ///
    /// This takes priority over both the engine-reported default and any
    /// tokenizer/model metadata exposed by the backend.
    pub fn with_max_model_len(mut self, max_model_len: u32) -> Self {
        self.max_model_len = Some(max_model_len);
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

        let mut sampling_hints = self.backend.sampling_hints()?;
        if let Some(max_model_len) = self.max_model_len {
            sampling_hints.max_model_len = Some(max_model_len);
        }
        let PreparedTextRequest {
            text_request,
            generate_request,
        } = lower_text_request(request, prompt_token_ids, sampling_hints, &*tokenizer)?;

        let raw_stream = self.llm.generate(generate_request).await?;
        Ok((text_request, raw_stream))
    }

    /// Shut down the underlying LLM client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.llm.shutdown().await?;
        Ok(())
    }
}
