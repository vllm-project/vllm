use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::output::TextDecodeOptions;

/// One raw text-generation prompt.
///
/// This supports either ordinary text that still needs tokenization or already-tokenized prompt
/// IDs that should bypass tokenizer work entirely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    /// Untokenized prompt text that still needs tokenizer work before generation.
    Text(String),
    /// Pre-tokenized prompt IDs that should be forwarded southbound without re-encoding.
    TokenIds(Vec<u32>),
}

/// User-facing sampling parameters accepted by `vllm-chat`.
///
/// This intentionally keeps only the subset that the current Rust chat layer
/// supports as northbound request semantics. Engine-core-specific normalized
/// fields are derived later during lowering.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L155-L291>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserSamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling. `None` means no explicit user override.
    pub temperature: Option<f32>,
    /// Cumulative probability threshold for nucleus sampling.
    pub top_p: Option<f32>,
    /// Maximum number of top tokens to consider. `Some(0)` means all tokens.
    pub top_k: Option<i32>,
    /// Random seed used by the sampler when present.
    pub seed: Option<u64>,
    /// Maximum number of tokens to generate. `None` means no explicit user override.
    pub max_tokens: Option<u32>,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    pub min_tokens: Option<u32>,
    /// Minimum probability threshold for token sampling. `None` means no explicit user override.
    pub min_p: Option<f32>,
    /// Frequency penalty applied by the sampler. `None` means no explicit user override.
    pub frequency_penalty: Option<f32>,
    /// Presence penalty applied by the sampler. `None` means no explicit user override.
    pub presence_penalty: Option<f32>,
    /// Repetition penalty applied by the sampler. `None` means no explicit user override.
    pub repetition_penalty: Option<f32>,
    /// Explicit stop token IDs provided by the caller. `None` means no explicit user override.
    pub stop_token_ids: Option<Vec<u32>>,
    /// If true, do not stop on the model's primary EOS token.
    pub ignore_eos: bool,
}

#[allow(clippy::derivable_impls)] // more explicit
impl Default for UserSamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            max_tokens: None,
            min_tokens: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_token_ids: None,
            ignore_eos: false,
        }
    }
}

/// One raw text-generation request ready to be tokenized or sent directly to the engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextRequest {
    /// Stable caller-supplied request ID.
    pub request_id: String,
    /// Prompt text or prompt token IDs for this request.
    pub prompt: Prompt,
    /// User-facing sampling parameters accepted by `vllm-text`.
    pub sampling_params: UserSamplingParams,
    /// Incremental detokenization options for the response path.
    pub decode_options: TextDecodeOptions,
}

impl TextRequest {
    /// Validate the minimum invariants before tokenization or request lowering.
    pub fn validate(&self) -> Result<()> {
        if matches!(&self.prompt, Prompt::TokenIds(ids) if ids.is_empty()) {
            return Err(Error::EmptyPromptTokenIds {
                request_id: self.request_id.clone(),
            });
        }
        Ok(())
    }
}
