// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Native Kimi K3 XTML chat renderer.

mod encoding;
#[cfg(test)]
mod tests;

use vllm_text::Prompt;
use vllm_text::tokenizer::DynTokenizer;

use super::{ChatRenderer, MediaPartSource, RenderedPrompt, request_template_kwargs};
use crate::Result;
use crate::request::ChatRequest;

/// Dedicated Kimi K3 XTML renderer.
#[derive(Clone)]
pub struct KimiK3ChatRenderer {
    tokenizer: DynTokenizer,
}

impl KimiK3ChatRenderer {
    /// Create a Kimi K3 renderer.
    pub fn new(tokenizer: DynTokenizer) -> Self {
        Self { tokenizer }
    }
}

impl ChatRenderer for KimiK3ChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        self.render_with_media_order(request).map(|(rendered, _)| rendered)
    }

    fn render_with_media_order(
        &self,
        request: &ChatRequest,
    ) -> Result<(RenderedPrompt, Vec<MediaPartSource>)> {
        request.validate()?;
        let (token_ids, media_order) =
            encoding::render_request_with_media_order(request, self.tokenizer.as_ref())?;

        Ok((
            RenderedPrompt {
                prompt: Prompt::TokenIds(token_ids),
                effective_template_kwargs: request_template_kwargs(request),
            },
            media_order,
        ))
    }
}
