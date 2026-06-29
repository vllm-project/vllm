mod encoding;

use vllm_text::Prompt;

use super::{ChatRenderer, RenderedPrompt, request_template_kwargs};
use crate::Result;
use crate::request::ChatRequest;

/// Dedicated DeepSeek V3.2 renderer.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeepSeekV32ChatRenderer;

impl DeepSeekV32ChatRenderer {
    /// Create the dedicated DeepSeek V3.2 renderer.
    pub fn new() -> Self {
        Self
    }
}

impl ChatRenderer for DeepSeekV32ChatRenderer {
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
