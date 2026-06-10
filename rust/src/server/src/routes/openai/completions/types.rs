use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;
use vllm_text::Prompt;

use crate::routes::openai::utils::types::{
    LogProbs, Normalizable, StreamOptions, StringOrArray, Usage, default_true, validate_stop,
};

/// Serde default for `CompletionRequest::max_tokens`, matching the Python vLLM
/// / OpenAI default.
fn default_completion_max_tokens() -> Option<u32> {
    Some(16)
}

/// vLLM-compatible request type for the Completions API.
///
/// Mirrors the Python vLLM `CompletionRequest` class. The local copy keeps the
/// request type route-owned so we can accept token-id prompts via
/// [`vllm_text::Prompt`] and add vLLM-only fields directly instead of layering
/// wrapper deserializers on top.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct CompletionRequest {
    // -------- Standard OpenAI API Parameters --------
    /// ID of the model to use
    pub model: String,

    /// The prompt(s) to generate completions for.
    ///
    /// We use [`vllm_text::Prompt`] here to support token-id input.
    pub prompt: Prompt,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: bool,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based
    /// on their existing frequency in the text so far
    pub frequency_penalty: Option<f32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Include the log probabilities on the logprobs most likely tokens
    pub logprobs: Option<u32>,

    /// The maximum number of tokens to generate (defaults to 16 when absent,
    /// matching the Python vLLM / OpenAI API convention)
    #[serde(default = "default_completion_max_tokens")]
    pub max_tokens: Option<u32>,

    /// How many completions to generate for each prompt
    pub n: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based
    /// on whether they appear in the text so far
    pub presence_penalty: Option<f32>,

    /// If specified, our system will make a best effort to sample
    /// deterministically
    pub seed: Option<i64>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// Whether to stream back partial progress
    #[serde(default)]
    pub stream: bool,

    /// The suffix that comes after a completion of inserted text
    pub suffix: Option<String>,

    /// What sampling temperature to use, between 0 and 2
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature (nucleus sampling)
    pub top_p: Option<f32>,

    /// A unique identifier representing your end-user
    pub user: Option<String>,

    // -------- vLLM Sampling Parameters --------
    /// Options for streaming response
    pub stream_options: Option<StreamOptions>,

    /// Use beam search instead of sampling
    #[serde(default)]
    pub use_beam_search: bool,

    /// Top-k sampling parameter
    pub top_k: Option<u32>,

    /// Min-p nucleus sampling parameter
    pub min_p: Option<f32>,

    /// Repetition penalty for reducing repetitive text
    pub repetition_penalty: Option<f32>,

    /// Length penalty for beam search
    pub length_penalty: Option<f32>,

    /// Specific token IDs to use as stop conditions
    pub stop_token_ids: Option<Vec<u32>>,

    /// Include stop string in output
    #[serde(default)]
    pub include_stop_str_in_output: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Minimum number of tokens to generate
    pub min_tokens: Option<u32>,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Add spaces between special tokens during detokenization
    #[serde(default = "default_true")]
    pub spaces_between_special_tokens: bool,

    /// Truncate prompt tokens to this length
    pub truncate_prompt_tokens: Option<i64>,

    /// Restrict output to these token IDs only
    pub allowed_token_ids: Option<Vec<u32>>,

    /// Number of prompt logprobs to return
    pub prompt_logprobs: Option<i32>,

    // -------- Extra vLLM Parameters --------
    /// Whether to add special tokens (e.g. BOS) to the prompt
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,

    /// Format specification for structured output (JSON mode, JSON schema,
    /// etc.)
    pub response_format: Option<Value>,

    /// Additional kwargs for structured outputs
    pub structured_outputs: Option<Value>,

    /// Request scheduling priority (lower means earlier; default 0)
    pub priority: Option<i32>,

    /// External request ID used for response correlation.
    pub request_id: Option<String>,

    /// Tokens represented as strings of the form 'token_id:{token_id}' in
    /// logprobs
    pub return_tokens_as_token_ids: Option<bool>,

    /// Include token IDs alongside generated text
    pub return_token_ids: Option<bool>,

    /// Salt for prefix cache isolation in multi-user environments
    pub cache_salt: Option<String>,

    /// KV transfer parameters for disaggregated serving
    pub kv_transfer_params: Option<HashMap<String, Value>>,

    /// Additional request parameters with string or numeric values for custom
    /// extensions
    pub vllm_xargs: Option<HashMap<String, Value>>,

    /// Additional fields
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for CompletionRequest {}

/// Mirrors the Python vLLM `CompletionResponse` class.
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
    pub kv_transfer_params: Option<Value>,
}

/// Mirrors the Python vLLM `CompletionResponseChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<String, f32>>>>,
    pub token_ids: Option<Vec<u32>>,
    pub prompt_token_ids: Option<Vec<u32>>,
}

/// Mirrors the Python vLLM `CompletionStreamResponse` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct CompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
    pub usage: Option<Usage>,
}

impl CompletionStreamResponse {
    /// Create a stream response with the standard envelope fields pre-filled.
    pub fn new(id: &str, model: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            object: "text_completion".to_string(),
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: None,
        }
    }
}

/// Mirrors the Python vLLM `CompletionResponseStreamChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
pub(super) struct CompletionStreamChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
    pub token_ids: Option<Vec<u32>>,
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(super) enum CompletionSseChunk {
    /// Ordinary OpenAI completions delta/final chunk.
    Chunk(CompletionStreamResponse),
    /// Final usage chunk emitted before `[DONE]` when `include_usage=true`.
    Usage(CompletionStreamResponse),
}
