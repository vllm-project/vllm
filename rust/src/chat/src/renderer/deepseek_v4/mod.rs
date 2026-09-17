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

/// Dedicated DeepSeek V4 renderer.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekV4ChatRenderer {
    default_template_kwargs: HashMap<String, Value>,
}

impl DeepSeekV4ChatRenderer {
    pub fn new(default_template_kwargs: HashMap<String, Value>) -> Self {
        Self {
            default_template_kwargs,
        }
    }
}

impl ChatRenderer for DeepSeekV4ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        let reasoning = deepseek::resolve_reasoning(
            ReasoningControl::resolve(request, &self.default_template_kwargs)?,
            DsDialect::V4,
        )?;

        Ok(RenderedPrompt {
            prompt: Prompt::Text(deepseek::render_request(
                request,
                DsDialect::V4,
                &reasoning,
            )?),
            media_order: None,
            effective_template_kwargs: reasoning.template_kwargs(request),
        })
    }
}

#[cfg(test)]
mod tests;
