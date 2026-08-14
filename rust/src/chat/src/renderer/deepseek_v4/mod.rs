// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod encoding;

use vllm_text::Prompt;

use super::{ChatRenderer, RenderedPrompt, request_template_kwargs};
use crate::Result;
use crate::request::ChatRequest;

/// Dedicated DeepSeek V4 renderer.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekV4ChatRenderer;

impl DeepSeekV4ChatRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl ChatRenderer for DeepSeekV4ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;

        Ok(RenderedPrompt {
            prompt: Prompt::Text(encoding::render_request(request)?),
            effective_template_kwargs: request_template_kwargs(request),
        })
    }
}

#[cfg(test)]
mod tests;
