pub mod hf;

use std::sync::Arc;

use vllm_tokenizer::DynTokenizer;

use crate::error::Result;

/// Tokenizer/model-derived hints used to enrich text-generation requests before
/// they are lowered into engine-core.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingHints {
    pub primary_eos_token_id: Option<u32>,
    pub extra_eos_token_ids: std::collections::BTreeSet<u32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<u32>,
    pub default_min_p: Option<f32>,
    pub default_repetition_penalty: Option<f32>,
    pub default_max_tokens: Option<u32>,
    /// Model context window size (`max_position_embeddings` from
    /// `config.json`).
    pub max_model_len: Option<u32>,
}

/// Minimal text-processing backend needed by `vllm-text`.
pub trait TextBackend: Send + Sync {
    /// Return the tokenizer used by this backend.
    fn tokenizer(&self) -> DynTokenizer;

    /// Return whether the loaded model is a mixture-of-experts model.
    fn is_moe(&self) -> bool {
        false
    }

    /// Return the backend model ID.
    fn model_id(&self) -> &str;

    /// Return tokenizer/model-derived hints used to enrich southbound sampling
    /// parameters.
    fn sampling_hints(&self) -> Result<SamplingHints> {
        Ok(SamplingHints::default())
    }

    /// Return the model vocabulary size from the model config, if known. Used to
    /// range-check request token ids against the engine embedding table.
    fn model_vocab_size(&self) -> Option<usize> {
        None
    }

    /// Return the full tokenizer vocabulary size (Python `len(tokenizer)`).
    /// Used to range-check `allowed_token_ids` and token-id prompts.
    fn tokenizer_vocab_size(&self) -> usize {
        self.tokenizer().vocab_size()
    }
}

/// Shared trait-object form of [`TextBackend`].
pub type DynTextBackend = Arc<dyn TextBackend>;
