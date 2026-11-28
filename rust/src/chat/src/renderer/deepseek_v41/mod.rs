// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use vllm_text::tokenizer::DynTokenizer;

use super::deepseek::{self, DsDialect, SpecialTokenizer};
use super::{ChatRenderer, RenderedPrompt};
use crate::Result;
use crate::reasoning::ReasoningControl;
use crate::request::ChatRequest;

/// Dedicated DeepSeek V4.1 renderer.
#[derive(Debug, Clone, Default)]
pub struct DeepSeekV41ChatRenderer {
    default_template_kwargs: HashMap<String, Value>,
    tokenizer: Option<Arc<SpecialTokenizer>>,
}

impl DeepSeekV41ChatRenderer {
    pub fn new(default_template_kwargs: HashMap<String, Value>) -> Self {
        Self {
            default_template_kwargs,
            tokenizer: None,
        }
    }

    /// Render token IDs in which special-token spellings in client text are
    /// plain text, as in deepseek-recipe.
    pub fn with_tokenizer(mut self, tokenizer: DynTokenizer) -> Result<Self> {
        self.tokenizer = Some(Arc::new(SpecialTokenizer::new(tokenizer)?));
        Ok(self)
    }
}

impl ChatRenderer for DeepSeekV41ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        let reasoning = deepseek::resolve_reasoning(
            ReasoningControl::resolve(request, &self.default_template_kwargs)?,
            DsDialect::V41,
        )?;
        let (prompt, media_order) = deepseek::render_request_with_media_order(
            request,
            DsDialect::V41,
            &reasoning,
            self.tokenizer.as_deref(),
        )?;

        Ok(RenderedPrompt {
            prompt,
            media_order: Some(media_order),
            effective_template_kwargs: reasoning.template_kwargs(request),
        })
    }

    fn render_completion_suffix(&self, prompt: &str, suffix: &str) -> Option<String> {
        Some(deepseek::render_completion_suffix(prompt, suffix))
    }
}

#[cfg(test)]
mod tests;
