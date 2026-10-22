// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod encoding;
mod writer;

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use vllm_text::tokenizer::DynTokenizer;

use super::{ChatRenderer, RenderedPrompt};
use crate::error::{Result, bail_chat_template_thrown};
use crate::request::ChatRequest;

#[derive(Clone, Default)]
pub struct Glm5NextChatRenderer {
    default_template_kwargs: HashMap<String, Value>,
    multimodal: bool,
    tokenizer: Option<Arc<writer::ControlTokenizer>>,
}

/// Chat templates shipped with GLM-5.3 and GLM-5.3-Flash, which this renderer
/// reproduces.
const SHIPPED_TEMPLATES: [&str; 2] = [
    include_str!("fixtures/glm_5_3.jinja"),
    include_str!("fixtures/glm_5_3_flash.jinja"),
];

impl Glm5NextChatRenderer {
    /// Model types `auto` may select this renderer for. GLM-5 through GLM-5.3
    /// share `glm_moe_dsa` with different templates, so the checkpoint's
    /// template must also be [shipped](Self::is_shipped_template).
    pub const AUTO_MODEL_TYPES: [&str; 2] = ["glm_moe_dsa", "glm5_next"];

    /// Whether `template` is one this renderer reproduces.
    pub fn is_shipped_template(template: Option<&str>) -> bool {
        // Normalize as `load_chat_template` does for `.jinja` files.
        let normalize = |template: &str| template.trim().replace("\\n", "\n");
        template.is_some_and(|template| {
            let template = normalize(template);
            SHIPPED_TEMPLATES.iter().any(|shipped| normalize(shipped) == template)
        })
    }

    pub fn new(default_template_kwargs: HashMap<String, Value>) -> Self {
        Self {
            default_template_kwargs,
            multimodal: false,
            tokenizer: None,
        }
    }

    /// Render token IDs: control tokens use added-vocabulary IDs and literal
    /// text uses the base vocabulary.
    pub fn with_tokenizer(mut self, tokenizer: DynTokenizer) -> Result<Self> {
        self.tokenizer = Some(Arc::new(writer::ControlTokenizer::new(tokenizer)?));
        Ok(self)
    }

    pub fn with_multimodal(mut self, enabled: bool) -> Self {
        self.multimodal = enabled;
        self
    }
}

impl ChatRenderer for Glm5NextChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        if request.chat_options.chat_template.is_some() {
            bail_chat_template_thrown!(
                "glm5_next does not accept a chat_template override; use tokenizer-mode hf for Jinja"
            );
        }
        let mut kwargs = self.default_template_kwargs.clone();
        kwargs.extend(request.chat_options.template_kwargs.clone());
        if let Some(effort) = &request.chat_options.reasoning_effort {
            kwargs.insert("reasoning_effort".into(), serde_json::json!(effort));
        }
        let messages = encoding::messages(&request.messages);
        let tools = if request.tool_parsing_enabled() {
            encoding::tools(request.tools())
        } else {
            Vec::new()
        };
        let mut out = writer::Glm5PromptWriter::new(self.tokenizer.as_deref());
        let media_order = encoding::render_into(
            &mut out,
            &messages,
            &tools,
            request.chat_options.generation_prompt_mode,
            &kwargs,
            self.multimodal,
        )?;
        Ok(RenderedPrompt {
            prompt: out.finish()?,
            media_order: Some(media_order),
            effective_template_kwargs: kwargs,
        })
    }
}

#[cfg(test)]
mod tests;
