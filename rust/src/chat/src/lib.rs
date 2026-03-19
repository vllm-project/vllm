#![feature(coroutines)]
#![feature(trait_alias)]

//! Minimal chat facade above [`vllm_llm`].
//!
//! This crate keeps the northbound boundary intentionally small:
//! `messages -> rendered prompt -> tokenized prompt -> engine request -> streamed structured
//! assistant events`. The request side remains text-first, while the response side can emit
//! structured reasoning and final-answer blocks. It is closer to vLLM's internal chat-rendering
//! flow than to a full OpenAI-compatible surface.

pub use backend::{ChatBackend, DynChatBackend, SamplingHints};
pub use error::{Error, Result};
pub use event::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall, ChatEvent,
};
use futures::StreamExt;
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTool,
    ChatToolChoice, UserSamplingParams,
};
pub use stream::ChatEventStream;

mod backend;
pub mod backends;
mod error;
mod event;
mod lower;
mod output;
mod request;
mod stream;
mod template;

use lower::lower_chat_request;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use tool_parser::ParserFactory as ToolParserFactory;
use vllm_llm::Llm;

/// Chat facade with a text-first request model layered above [`vllm_llm::Llm`].
///
/// This mirrors the useful shape of vLLM's frontend pipeline:
/// `messages -> rendered prompt -> tokenized prompt -> engine request -> streamed structured
/// assistant events`.
pub struct ChatLlm {
    llm: Llm,
    backend: DynChatBackend,
    reasoning_parser_factory: ReasoningParserFactory,
    tool_parser_factory: ToolParserFactory,
    /// Explicit tool call parser name override (bypasses model-based auto-detection).
    tool_call_parser: Option<String>,
    /// Explicit reasoning parser name override (bypasses model-based auto-detection).
    reasoning_parser: Option<String>,
    /// Explicit override for the maximum model context length.
    max_model_len: Option<u32>,
}

impl ChatLlm {
    /// Create a new chat facade from an LLM client plus a chat backend.
    pub fn new(llm: Llm, backend: DynChatBackend) -> Self {
        Self {
            llm,
            backend,
            reasoning_parser_factory: ReasoningParserFactory::new(),
            tool_parser_factory: ToolParserFactory::new(),
            tool_call_parser: None,
            reasoning_parser: None,
            max_model_len: None,
        }
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

    /// Override the maximum model context length, taking priority over the backend's
    /// `max_position_embeddings` from `config.json`.
    pub fn with_max_model_len(mut self, max_model_len: u32) -> Self {
        self.max_model_len = Some(max_model_len);
        self
    }

    /// Render, tokenize, and submit one chat request.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;

        let model_id = self.backend.model_id();

        let prompt = self.backend.apply_chat_template(&request)?;
        let prompt_token_ids = self.backend.encode(&prompt)?;
        let mut sampling_hints = self.backend.sampling_hints()?;
        if let Some(max_model_len) = self.max_model_len {
            sampling_hints.max_model_len = Some(max_model_len);
        }
        let prepared = lower_chat_request(request, prompt_token_ids, sampling_hints)?;

        let raw_stream = self.llm.generate(prepared.generate_request).await?;
        let structured_stream = output::output_stream(
            prepared.chat_request.clone(),
            self.backend.clone(),
            raw_stream,
            model_id,
            &self.reasoning_parser_factory,
            &self.tool_parser_factory,
            self.reasoning_parser.as_deref(),
            self.tool_call_parser.as_deref(),
        )?;

        Ok(ChatEventStream::new(
            prepared.chat_request.request_id.clone(),
            structured_stream.boxed(),
        ))
    }

    /// Abort one in-flight chat request by request ID.
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
