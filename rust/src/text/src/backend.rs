use std::sync::Arc;

use crate::error::Result;

/// Tokenizer/model-derived hints used to enrich text-generation requests before they are lowered
/// into engine-core.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingHints {
    pub primary_eos_token_id: Option<u32>,
    pub extra_eos_token_ids: std::collections::BTreeSet<u32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<i32>,
    pub default_min_p: Option<f32>,
    pub default_repetition_penalty: Option<f32>,
    pub default_max_tokens: Option<u32>,
    /// Model context window size (`max_position_embeddings` from `config.json`).
    pub max_model_len: Option<u32>,
}

/// Minimal text-processing backend needed by `vllm-text`.
pub trait TextBackend: Send + Sync {
    /// Encode one prompt string into token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Decode one token sequence into text.
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

/// Shared trait-object form of [`TextBackend`].
pub type DynTextBackend = Arc<dyn TextBackend>;
