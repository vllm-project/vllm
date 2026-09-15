// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod encoding;

use std::collections::HashMap;

use serde_json::Value;
use vllm_text::Prompt;

use super::reasoning::ReasoningControl;
use super::{ChatRenderer, RenderedPrompt};
use crate::Result;
use crate::request::ChatRequest;

/// Dedicated DeepSeek V3.2 renderer.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekV32ChatRenderer {
    default_template_kwargs: HashMap<String, Value>,
}

impl DeepSeekV32ChatRenderer {
    /// Create the dedicated DeepSeek V3.2 renderer.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set deployment defaults used below explicit request reasoning controls.
    pub fn with_default_template_kwargs(mut self, kwargs: HashMap<String, Value>) -> Self {
        self.default_template_kwargs = kwargs;
        self
    }
}

impl ChatRenderer for DeepSeekV32ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        let reasoning = ReasoningControl::resolve(request, &self.default_template_kwargs)?
            .fallback(ReasoningControl::Disabled);
        let reasoning = encoding::resolve_reasoning(reasoning)?;

        Ok(RenderedPrompt {
            prompt: Prompt::Text(encoding::render_request(request, &reasoning)?),
            media_order: None,
            effective_template_kwargs: reasoning.template_kwargs(request),
        })
    }
}

#[cfg(test)]
mod tests;
