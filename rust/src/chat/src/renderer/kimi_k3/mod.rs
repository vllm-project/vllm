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

use super::{ChatRenderer, RenderedPrompt, RendererSelection};
use crate::Result;
use crate::request::ChatRequest;

/// Kimi K3 checkpoint generation rendered by [`KimiK3ChatRenderer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiK3Variant {
    /// Kimi K3.
    K3,
    /// Kimi K3.1, whose encoder also accepts `thinking_effort="medium"`.
    K31,
}

impl KimiK3Variant {
    /// Resolve the variant from the checkpoint's `model_type`.
    pub fn from_model_type(model_type: &str) -> Self {
        if model_type == RendererSelection::KIMI_K31_MODEL_TYPE {
            Self::K31
        } else {
            Self::K3
        }
    }

    /// `thinking_effort` values accepted by this variant's `encoding_k3.py`.
    fn thinking_efforts(self) -> &'static [&'static str] {
        match self {
            Self::K3 => &["low", "high", "max"],
            Self::K31 => &["low", "medium", "high", "max"],
        }
    }
}

/// Dedicated Kimi K3 XTML renderer.
#[derive(Clone)]
pub struct KimiK3ChatRenderer {
    tokenizer: DynTokenizer,
    default_template_kwargs: HashMap<String, Value>,
    variant: KimiK3Variant,
}

impl KimiK3ChatRenderer {
    /// Create a Kimi K3 renderer for the given checkpoint variant.
    pub fn new(
        tokenizer: DynTokenizer,
        default_template_kwargs: HashMap<String, Value>,
        variant: KimiK3Variant,
    ) -> Self {
        Self {
            tokenizer,
            default_template_kwargs,
            variant,
        }
    }
}

impl ChatRenderer for KimiK3ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        let reasoning =
            encoding::resolve_reasoning(request, &self.default_template_kwargs, self.variant)?;
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
