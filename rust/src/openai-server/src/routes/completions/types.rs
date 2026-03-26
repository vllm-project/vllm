use std::collections::HashMap;

use openai_protocol::common::{LogProbs, StreamOptions, StringOrArray, Usage, default_true};
use openai_protocol::validated::Normalizable;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;
use vllm_text::Prompt;

/// vLLM-compatible request type for the Completions API.
///
/// This mirrors [`openai_protocol::completion::CompletionRequest`], but keeps the
/// request type local to the route module so we can extend it directly with
/// vLLM-compatible fields instead of layering wrapper deserializers on top.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct CompletionRequest {
    /// ID of the model to use
    pub model: String,

    /// The prompt(s) to generate completions for.
    ///
    /// We [`vllm_text::Prompt`] here to support token-id input.
    pub prompt: Prompt,

    /// The suffix that comes after a completion of inserted text
    pub suffix: Option<String>,

    /// The maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// What sampling temperature to use, between 0 and 2
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature (nucleus sampling)
    pub top_p: Option<f32>,

    /// How many completions to generate for each prompt
    pub n: Option<u32>,

    /// Whether to stream back partial progress
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming response
    pub stream_options: Option<StreamOptions>,

    /// Include the log probabilities on the logprobs most likely tokens
    pub logprobs: Option<u32>,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: bool,

    /// Up to 4 sequences where the API will stop generating further tokens
    pub stop: Option<StringOrArray>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far
    pub presence_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    /// frequency in the text so far
    pub frequency_penalty: Option<f32>,

    /// Generates best_of completions server-side and returns the "best"
    pub best_of: Option<u32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    pub logit_bias: Option<HashMap<String, f32>>,

    /// A unique identifier representing your end-user
    pub user: Option<String>,

    /// If specified, our system will make a best effort to sample deterministically
    pub seed: Option<i64>,

    // -------- Engine Specific Sampling Parameters --------
    /// Top-k sampling parameter (-1 to disable)
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    pub ebnf: Option<String>,

    /// JSON schema constraint for structured output
    pub json_schema: Option<String>,

    /// Specific token IDs to use as stop conditions
    pub stop_token_ids: Option<Vec<u32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Path to LoRA adapter(s) for model customization
    pub lora_path: Option<String>,

    /// Session parameters for continual prompting
    pub session_params: Option<HashMap<String, Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Sampling seed for deterministic outputs
    pub sampling_seed: Option<u64>,

    /// vLLM-compatible prompt logprobs request field missing from `openai-protocol`.
    pub prompt_logprobs: Option<i32>,

    /// Additional fields including bootstrap info for PD routing
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for CompletionRequest {}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<Usage>,
    pub system_fingerprint: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
    pub matched_stop: Option<Value>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<String, f32>>>>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<CompletionStreamChoice>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionStreamChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(super) enum CompletionSseChunk {
    /// Ordinary OpenAI completions delta/final chunk.
    Chunk(CompletionStreamResponse),
    /// Final usage chunk emitted before `[DONE]` when `include_usage=true`.
    Usage(CompletionUsageChunk),
}

/// Minimal JSON shape for the extra streamed usage chunk used by `vllm-bench`.
///
/// The streamed completion chunk DTO does not include a `usage` field, so the Rust server emits
/// this thin local wrapper for the terminal accounting event.
#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionUsageChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<CompletionStreamChoice>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub usage: Usage,
}
