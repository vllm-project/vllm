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
mod decoded;
mod error;
mod event;
mod incremental;
mod lower;
mod pipeline;
mod reasoning;
mod request;
mod stream;
mod structured;
mod template;
mod tool;

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
}

impl ChatLlm {
    /// Create a new chat facade from an LLM client plus a chat backend.
    pub fn new(llm: Llm, backend: DynChatBackend) -> Self {
        Self {
            llm,
            backend,
            reasoning_parser_factory: ReasoningParserFactory::new(),
            tool_parser_factory: ToolParserFactory::new(),
        }
    }

    /// Render, tokenize, and submit one chat request.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;

        let model_id = self.backend.model_id();

        let prompt = self.backend.apply_chat_template(&request)?;
        let prompt_token_ids = self.backend.encode(&prompt)?;
        let sampling_hints = self.backend.sampling_hints()?;
        let prepared = lower_chat_request(request, prompt_token_ids, sampling_hints)?;

        let raw_stream = self.llm.generate(prepared.generate_request).await?;
        let tool_parser = if prepared.chat_request.tool_parsing_enabled() {
            if let Some(model_id) = model_id {
                self.tool_parser_factory
                    .registry()
                    .create_for_model(model_id)
                    .ok_or_else(|| Error::ToolParserUnavailableForModel {
                        model_id: model_id.to_string(),
                    })?
                    .into()
            } else {
                return Err(Error::ToolParserRequiresModelId);
            }
        } else {
            None
        };
        let reasoning_parser = if let Some(model_id) = model_id {
            self.reasoning_parser_factory
                .registry()
                .create_for_model(model_id)
        } else {
            None
        };

        let decoded_stream = decoded::decoded_text_event_stream(
            prepared.chat_request.clone(),
            self.backend.clone(),
            raw_stream,
        );
        let reasoning_stream = reasoning::reasoning_event_stream(decoded_stream, reasoning_parser);
        let content_stream = tool::tool_event_stream(
            reasoning_stream,
            prepared.chat_request.clone(),
            tool_parser,
            self.backend.model_id().map(str::to_owned),
        );
        let structured_stream = structured::structured_chat_event_stream(content_stream);

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
