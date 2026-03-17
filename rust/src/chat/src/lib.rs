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
pub use event::{AssistantBlockKind, AssistantContentBlock, AssistantMessage, ChatEvent};
use futures::StreamExt;
pub use request::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole,
    UserSamplingParams,
};
pub use stream::ChatEventStream;

mod backend;
pub mod backends;
mod decoded;
mod error;
mod event;
mod incremental;
mod lower;
mod request;
mod stream;
mod structured;
mod template;

use lower::lower_chat_request;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
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
}

impl ChatLlm {
    /// Create a new chat facade from an LLM client plus a chat backend.
    pub fn new(llm: Llm, backend: DynChatBackend) -> Self {
        Self {
            llm,
            backend,
            reasoning_parser_factory: ReasoningParserFactory::new(),
        }
    }

    /// Render, tokenize, and submit one chat request.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;

        let prompt = self.backend.apply_chat_template(&request)?;
        let prompt_token_ids = self.backend.encode(&prompt)?;
        let sampling_hints = self.backend.sampling_hints()?;
        let prepared = lower_chat_request(request, prompt_token_ids, sampling_hints)?;

        let raw_stream = self.llm.generate(prepared.generate_request).await?;
        let reasoning_parser = self.backend.model_id().and_then(|hint| {
            self.reasoning_parser_factory
                .registry()
                .create_for_model(hint)
        });

        let decoded_stream = decoded::decoded_text_event_stream(
            prepared.chat_request.clone(),
            self.backend.clone(),
            raw_stream,
        );
        let structured_stream =
            structured::structured_chat_event_stream(decoded_stream, reasoning_parser);

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
