// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Native Kimi K3 XTML chat renderer.

mod encoding;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use serde_json::Value;
use vllm_text::Prompt;
use vllm_text::tokenizer::DynTokenizer;

use super::{ChatRenderer, RenderedPrompt};
use crate::Result;
use crate::request::ChatRequest;

/// Dedicated Kimi K3 XTML renderer.
#[derive(Clone)]
pub struct KimiK3ChatRenderer {
    tokenizer: DynTokenizer,
    default_template_kwargs: HashMap<String, Value>,
}

impl KimiK3ChatRenderer {
    /// Create a Kimi K3 renderer.
    pub fn new(tokenizer: DynTokenizer, default_template_kwargs: HashMap<String, Value>) -> Self {
        Self {
            tokenizer,
            default_template_kwargs,
        }
    }
}

impl ChatRenderer for KimiK3ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        let reasoning = encoding::resolve_reasoning(request, &self.default_template_kwargs)?;
        let mut effective_template_kwargs = reasoning.template_kwargs(request);
        // Export K3's native effort key from the resolved standard controls.
        // Input `thinking_effort` is ignored; clear it when reasoning is disabled.
        if let Some(effort) = reasoning.effort() {
            effective_template_kwargs
                .insert("thinking_effort".to_string(), serde_json::json!(effort));
        } else {
            effective_template_kwargs.remove("thinking_effort");
        }
        let (token_ids, media_order) = encoding::render_request_with_media_order(
            request,
            self.tokenizer.as_ref(),
            &reasoning,
        )?;

        Ok(RenderedPrompt {
            prompt: Prompt::TokenIds(token_ids),
            media_order: Some(media_order),
            effective_template_kwargs,
        })
    }
}
