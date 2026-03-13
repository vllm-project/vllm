#![feature(coroutines)]

pub use error::{Error, Result};
pub use event::ChatEvent;
pub use renderer::{ChatRenderer, DynChatRenderer, LlmTokenizerChatRenderer, RenderedPrompt};
pub use request::{ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatTemplateContentFormat};
pub use stream::ChatEventStream;
pub use tokenizer::{DynTokenizer, LlmTokenizer, Tokenizer};

mod error;
mod event;
mod lower;
mod renderer;
mod request;
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
    pub fn new(llm: Llm, renderer: DynChatRenderer, tokenizer: DynTokenizer) -> Self {
        Self {
            llm,
            renderer,
            tokenizer,
        }
    }

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

    pub async fn abort(&self, request_id: &str) -> Result<()> {
        self.llm.abort(request_id).await?;
        Ok(())
    }

    pub async fn shutdown(self) -> Result<()> {
        self.llm.shutdown().await?;
        Ok(())
    }
}
