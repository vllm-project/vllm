pub mod hf;

use std::sync::Arc;

use vllm_tokenizer::DynTokenizer;

use crate::error::Result;

/// Tokenizer/model-derived defaults used to enrich text-generation requests
/// before they are lowered into engine-core.
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
}

/// Effective bounds used to validate and lower sampling requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingLimits {
    /// Runtime context window size reported by the engine startup handshake.
    pub max_model_len: u32,
    /// Maximum number of top log probabilities accepted by this frontend.
    ///
    /// `-1` means allowing requests up to the model vocabulary size.
    pub max_logprobs: i32,

    /// Model vocabulary size from the model config, used to bound generated
    /// token IDs and logits-domain sampling controls.
    pub model_vocab_size: usize,
    /// Tokenizer vocabulary size, used to bound `allowed_token_ids` and
    /// token-ID prompts.
    pub tokenizer_vocab_size: usize,
}

impl SamplingLimits {
    /// Original Python definition:
    /// <https://github.com/vllm-project/vllm/blob/b5adb027ad03c29b46181752ba3b1cb84eff1dd4/vllm/config/model.py#L216-L220>
    pub const DEFAULT_MAX_LOGPROBS: i32 = 20;
    /// Original Python definition:
    /// <https://github.com/vllm-project/vllm/blob/b5adb027ad03c29b46181752ba3b1cb84eff1dd4/vllm/sampling_params.py#L30-L32>
    pub const MAX_LOGPROB_TOKEN_IDS: usize = 128;

    /// Return the union bound used to validate token-ID prompts.
    pub fn prompt_token_vocab_size(&self) -> usize {
        self.tokenizer_vocab_size.max(self.model_vocab_size)
    }
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

    /// Return the model vocabulary size from the model config.
    ///
    /// The permissive default exists for lightweight test backends. Production
    /// backends should override it with the resolved model config value.
    fn model_vocab_size(&self) -> usize {
        usize::MAX
    }

    /// Return the full tokenizer vocabulary size (Python `len(tokenizer)`).
    /// Used to range-check `allowed_token_ids` and token-id prompts.
    fn tokenizer_vocab_size(&self) -> usize {
        self.tokenizer().vocab_size()
    }
}

/// Shared trait-object form of [`TextBackend`].
pub type DynTextBackend = Arc<dyn TextBackend>;
