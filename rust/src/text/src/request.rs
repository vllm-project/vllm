use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use vllm_engine_core_client::protocol::StructuredOutputsParams;

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

impl Default for Prompt {
    fn default() -> Self {
        Self::Text(String::new()) // placeholder
    }
}

/// User-facing sampling parameters accepted by `vllm-text`.
///
/// This intentionally keeps only the subset that the current Rust text layer
/// supports as northbound request semantics. Engine-core-specific normalized
/// fields are derived later during lowering.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/f22d6e026798a74e6542a52ef776c054f2de572a/vllm/sampling_params.py#L155-L291>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Controls randomness. Lower values are more deterministic; zero means
    /// greedy sampling. `None` means no explicit user override.
    pub temperature: Option<f32>,
    /// Cumulative probability threshold for nucleus sampling.
    pub top_p: Option<f32>,
    /// Maximum number of top tokens to consider. `Some(0)` means all tokens.
    pub top_k: Option<u32>,
    /// Random seed used by the sampler when present.
    pub seed: Option<i64>,
    /// Maximum number of tokens to generate. `None` means no explicit user override.
    pub max_tokens: Option<u32>,
    /// Minimum number of tokens to generate before EOS or stop-token handling.
    pub min_tokens: Option<u32>,
    /// Number of log probabilities to return per generated token.
    ///
    /// `None` disables sample logprobs. `-1` requests the full vocabulary.
    pub logprobs: Option<i32>,
    /// Number of log probabilities to return per prompt token.
    ///
    /// `None` disables prompt logprobs. `-1` requests the full vocabulary.
    pub prompt_logprobs: Option<i32>,
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
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Keys are token IDs.
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Restrict output to these token IDs only.
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Words to avoid during generation (tokenized to IDs during lowering).
    pub bad_words: Option<Vec<String>>,
    /// Parameters for configuring structured outputs (guided decoding).
    pub structured_outputs: Option<StructuredOutputsParams>,
    /// Additional request parameters for custom extensions.
    pub vllm_xargs: Option<HashMap<String, Value>>,
}

#[allow(clippy::derivable_impls)] // more explicit
impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            max_tokens: None,
            min_tokens: None,
            logprobs: None,
            prompt_logprobs: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop_token_ids: None,
            ignore_eos: false,
            logit_bias: None,
            allowed_token_ids: None,
            bad_words: None,
            structured_outputs: None,
            vllm_xargs: None,
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
    pub sampling_params: SamplingParams,
    /// Incremental detokenization options for the response path.
    pub decode_options: TextDecodeOptions,
    /// Whether to emit intermediate northbound deltas before the terminal result.
    ///
    /// If `false`, callers only observe the terminal accumulated output. If `true`, callers may
    /// receive zero or more incremental decoded updates before the final terminal event.
    pub intermediate: bool,
    /// Request scheduling priority (lower means earlier handling; default 0).
    #[serde(default)]
    pub priority: i32,
    /// Salt for prefix cache isolation in multi-user environments.
    pub cache_salt: Option<String>,
    /// Whether to add special tokens (e.g. BOS) during prompt tokenization.
    pub add_special_tokens: bool,
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
