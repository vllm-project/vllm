#![feature(coroutines)]
#![feature(trait_alias)]

//! Minimal chat facade above [`vllm_text`].
//!
//! This crate keeps the northbound boundary intentionally small:
//! `messages -> rendered prompt -> tokenized prompt -> engine request -> streamed structured
//! assistant events`. The request side remains text-first, while the response side can emit
//! structured reasoning and final-answer blocks. It is closer to vLLM's internal chat-rendering
//! flow than to a full OpenAI-compatible surface.

pub use backend::{
    ChatBackend, ChatTextBackend, DynChatBackend, DynChatTextBackend, SamplingHints,
};
pub use backends::{LoadedModelBackends, load_model_backends};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
use futures::{StreamExt, TryStreamExt as _};
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, SamplingParams,
};
pub use stream::{ChatEventStream, ChatEventStreamTrait, CollectedAssistantMessage};
pub use vllm_llm::FinishReason;

mod backend;
pub mod backends;
mod error;
mod event;
mod output;
mod request;
mod stream;
mod template;

use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use vllm_engine_core_client::EngineCoreClient;
use vllm_llm::Llm;
use vllm_text::{Prompt, TextLlm, TextRequest};

/// Structured chat facade above [`TextLlm`].
///
/// This layer stays above raw text semantics: it takes care of chat-template rendering, exposes
/// structured assistant events, and adds chat-specific request semantics such as tool calls.
pub struct ChatLlm {
    text: TextLlm,
    backend: DynChatBackend,
    reasoning_parser_factory: ReasoningParserFactory,
    tool_parser_factory: ToolParserFactory,
    /// Explicit tool call parser name override (bypasses model-based auto-detection).
    tool_call_parser: Option<String>,
    /// Explicit reasoning parser name override (bypasses model-based auto-detection).
    reasoning_parser: Option<String>,
}

impl ChatLlm {
    /// Create a new chat facade from a text-generation facade plus a chat backend.
    pub fn new(text: TextLlm, backend: DynChatBackend) -> Self {
        Self {
            text,
            backend,
            reasoning_parser_factory: ReasoningParserFactory::new(),
            tool_parser_factory: ToolParserFactory::new(),
            tool_call_parser: None,
            reasoning_parser: None,
        }
    }

    /// Convenience constructor for one shared backend object that implements both text and chat
    /// responsibilities.
    pub fn from_shared_backend(llm: Llm, backend: DynChatTextBackend) -> Self {
        let text = TextLlm::new(llm, backend.clone());
        Self::new(text, backend)
    }

    /// Set an explicit tool call parser name, bypassing model-based auto-detection.
    pub fn with_tool_call_parser(mut self, name: impl Into<String>) -> Self {
        self.tool_call_parser = Some(name.into());
        self
    }

    /// Set an explicit reasoning parser name, bypassing model-based auto-detection.
    pub fn with_reasoning_parser(mut self, name: impl Into<String>) -> Self {
        self.reasoning_parser = Some(name.into());
        self
    }

    /// Expose the underlying text facade for raw text-generation routes such as `/v1/completions`.
    pub fn text(&self) -> &TextLlm {
        &self.text
    }

    /// Return the model ID reported by the underlying text backend when available.
    pub fn model_id(&self) -> Option<&str> {
        self.text.model_id()
    }

    /// Expose the underlying engine-core client for low-level utility/admin calls.
    pub fn engine_core_client(&self) -> &EngineCoreClient {
        self.text.engine_core_client()
    }

    /// Render, tokenize, and submit one chat request.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;

        let prompt = self.backend.apply_chat_template(&request)?;
        let text_request = TextRequest {
            request_id: request.request_id.clone(),
            prompt: Prompt::Text(prompt),
            sampling_params: request.sampling_params,
            decode_options: request.decode_options,
            intermediate: request.intermediate,
            priority: request.priority,
            cache_salt: request.cache_salt,
            add_special_tokens: request.add_special_tokens,
            data_parallel_rank: request.data_parallel_rank,
        };
        let decoded_stream = self.text.generate(text_request).await?.map_err(Error::from);
        let structured_stream = output::output_stream(
            request.intermediate,
            request.tools,
            request.tool_choice,
            decoded_stream,
            self.text.model_id(),
            &self.reasoning_parser_factory,
            &self.tool_parser_factory,
            self.reasoning_parser.as_deref(),
            self.tool_call_parser.as_deref(),
        )?;

        Ok(ChatEventStream::new(
            request.request_id,
            structured_stream.boxed(),
        ))
    }

    /// Shut down the underlying LLM client and its background tasks.
    pub async fn shutdown(self) -> Result<()> {
        self.text.shutdown().await?;
        Ok(())
    }
}
