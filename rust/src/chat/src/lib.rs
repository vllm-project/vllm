#![feature(coroutines)]
//! Minimal text-only chat facade above [`vllm_llm`].
//!
//! This crate keeps the northbound boundary intentionally small:
//! `messages -> rendered prompt -> tokenized prompt -> engine request -> streamed text events`.
//! It is closer to vLLM's internal chat-rendering flow than to a full OpenAI-compatible surface.

pub use error::{Error, Result};
pub use event::ChatEvent;
pub use renderer::{ChatRenderer, DynChatRenderer, RenderedPrompt};
pub use request::{ChatMessage, ChatOptions, ChatRequest, ChatRole};
pub use smg::{SmgTokenizer, SmgTokenizerChatRenderer};
pub use stream::ChatEventStream;
pub use tokenizer::{DynTokenizer, Tokenizer};

mod error;
mod event;
mod lower;
mod renderer;
mod request;
pub mod smg;
mod stream;
mod tokenizer;

use lower::lower_chat_request;
use vllm_llm::Llm;

/// Text-only chat facade layered above [`vllm_llm::Llm`].
///
/// This mirrors the useful shape of vLLM's frontend pipeline:
/// `messages -> rendered prompt -> tokenized prompt -> engine request -> streamed text events`.
pub struct ChatLlm {
    llm: Llm,
    renderer: DynChatRenderer,
    tokenizer: DynTokenizer,
}

impl ChatLlm {
    /// Create a new chat facade from an LLM client plus a renderer/tokenizer pair.
    pub fn new(llm: Llm, renderer: DynChatRenderer, tokenizer: DynTokenizer) -> Self {
        Self {
            llm,
            renderer,
            tokenizer,
        }
    }

    /// Render, tokenize, and submit one chat request.
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatEventStream> {
        request.validate()?;
        let rendered = self.renderer.render(&request)?;
        let prepared = lower_chat_request(request, rendered, self.tokenizer.as_ref())?;
        let raw_stream = self.llm.generate(prepared.generate_request).await?;
        Ok(ChatEventStream::new(
            prepared.request_id,
            self.tokenizer.clone(),
            raw_stream,
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
