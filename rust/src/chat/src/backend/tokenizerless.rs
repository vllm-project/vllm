// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use crate::backend::{ChatBackend, NewChatOutputProcessorOptions};
use crate::error::Result;
use crate::output::DynChatOutputProcessor;
use crate::renderer::{ChatRenderer, DynChatRenderer, RenderedPrompt};
use crate::request::ChatRequest;

/// Chat backend under `--skip-tokenizer-init`: rendering and parsing a chat
/// need the tokenizer, so every chat request is rejected, like Python.
pub(super) struct TokenizerlessChatBackend;

/// Error for every chat request without a tokenizer.
fn tokenizer_unavailable() -> crate::Error {
    vllm_text::Error::TokenizerUnavailable.into()
}

impl ChatBackend for TokenizerlessChatBackend {
    fn chat_renderer(&self) -> DynChatRenderer {
        Arc::new(Self)
    }

    fn new_chat_output_processor(
        &self,
        _request: &mut ChatRequest,
        _options: NewChatOutputProcessorOptions<'_>,
    ) -> Result<DynChatOutputProcessor> {
        Err(tokenizer_unavailable())
    }
}

impl ChatRenderer for TokenizerlessChatBackend {
    fn render(&self, _request: &ChatRequest) -> Result<RenderedPrompt> {
        Err(tokenizer_unavailable())
    }
}
