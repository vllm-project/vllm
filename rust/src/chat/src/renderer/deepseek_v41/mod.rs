// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde_json::Value;
use vllm_text::Prompt;

use super::deepseek::{self, DsDialect};
use super::{ChatRenderer, RenderedPrompt};
use crate::Result;
use crate::reasoning::ReasoningControl;
use crate::request::ChatRequest;

/// Dedicated DeepSeek V4.1 renderer.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekV41ChatRenderer {
    default_template_kwargs: HashMap<String, Value>,
}

impl DeepSeekV41ChatRenderer {
    pub fn new(default_template_kwargs: HashMap<String, Value>) -> Self {
        Self {
            default_template_kwargs,
        }
    }
}

impl ChatRenderer for DeepSeekV41ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        let reasoning = deepseek::resolve_reasoning(
            ReasoningControl::resolve(request, &self.default_template_kwargs)?,
            DsDialect::V41,
        )?;
        let (prompt, media_order) =
            deepseek::render_request_with_media_order(request, DsDialect::V41, &reasoning)?;

        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
            media_order: Some(media_order),
            effective_template_kwargs: reasoning.template_kwargs(request),
        })
    }
}

#[cfg(test)]
mod tests;
