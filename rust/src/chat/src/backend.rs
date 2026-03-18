use std::collections::BTreeSet;
use std::sync::Arc;

use crate::error::Result;
use crate::request::ChatRequest;

/// Tokenizer/model-derived hints used to fill Python-aligned internal sampling metadata before
/// requests are lowered into engine-core.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingHints {
    pub primary_eos_token_id: Option<u32>,
    pub extra_eos_token_ids: BTreeSet<u32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<i32>,
    pub default_min_p: Option<f32>,
    pub default_repetition_penalty: Option<f32>,
    pub default_max_tokens: Option<u32>,
}

/// Minimal prompt-processing backend needed by `vllm-chat`.
pub trait ChatBackend: Send + Sync {
    /// Apply the chat template and return the rendered text prompt.
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String>;

    /// Encode one rendered chat prompt into token IDs.
    // TODO: currently, `add_special_tokens` is always false because chat templates are expected to
    // render the model-specific markers directly.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode one cumulative token sequence into text.
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Return the backend model ID when available.
    fn model_id(&self) -> Option<&str> {
        None
    }

    /// Return tokenizer/model-derived hints used to enrich southbound sampling parameters.
    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints::default())
    }
}

/// Shared trait-object form of [`ChatBackend`].
pub type DynChatBackend = Arc<dyn ChatBackend>;
