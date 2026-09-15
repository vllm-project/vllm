// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_text::Prompt;

use super::deepseek::{self, DsDialect};
use super::{ChatRenderer, RenderedPrompt, request_template_kwargs};
use crate::Result;
use crate::request::ChatRequest;

/// Dedicated DeepSeek V4.1 renderer.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekV41ChatRenderer;

impl DeepSeekV41ChatRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl ChatRenderer for DeepSeekV41ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        let (prompt, media_order) =
            deepseek::render_request_with_media_order(request, DsDialect::V41)?;

        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
            media_order: Some(media_order),
            effective_template_kwargs: request_template_kwargs(request),
        })
    }
}

#[cfg(test)]
mod tests;
