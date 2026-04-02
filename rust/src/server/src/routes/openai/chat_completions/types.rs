use std::collections::HashMap;

use openai_protocol::chat::{ChatMessage, MessageContent};
use openai_protocol::common::{
    StringOrArray, Tool, ToolCall, ToolCallDelta, ToolChoice, ToolChoiceValue, ToolReference,
    default_true, validate_stop,
};
use openai_protocol::sampling_params::validate_top_p_value;
use openai_protocol::validated::Normalizable;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use crate::routes::openai::utils::structured_outputs::ResponseFormat;
use crate::routes::openai::utils::types::{ChatLogProbs, StreamOptions, Usage};

/// vLLM-compatible request type for the Chat Completions API.
///
/// Mirrors the Python vLLM `ChatCompletionRequest` class. The local copy keeps the request type
/// route-owned so we can add vLLM-only fields directly instead of layering wrapper deserializers
/// on top.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
#[validate(schema(function = "validate_chat_cross_parameters"))]
pub struct ChatCompletionRequest {
    // -------- Standard OpenAI API Parameters --------
    /// A list of messages comprising the conversation so far
    #[validate(custom(function = "validate_messages"))]
    pub messages: Vec<ChatMessage>,

    /// ID of the model to use
    #[serde(default = "default_model")]
    pub model: String,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    /// frequency in the text so far
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Whether to return log probabilities of the output tokens
    #[serde(default)]
    pub logprobs: bool,

    /// An integer specifying the number of most likely tokens to return
    /// -1 means return all
    #[validate(range(min = -1))]
    pub top_logprobs: Option<i32>,

    /// Deprecated: Replaced by max_completion_tokens
    #[deprecated(note = "Use max_completion_tokens instead")]
    #[validate(range(min = 1))]
    pub max_tokens: Option<u32>,

    /// An upper bound for the number of tokens that can be generated for a completion
    #[validate(range(min = 1))]
    pub max_completion_tokens: Option<u32>,

    /// How many chat completion choices to generate for each input message
    #[validate(range(min = 1, max = 10))]
    pub n: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// An object specifying the format that the model must output
    pub response_format: Option<ResponseFormat>,

    /// If specified, our system will make a best effort to sample deterministically
    pub seed: Option<i64>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// If set, partial message deltas will be sent
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming response
    pub stream_options: Option<StreamOptions>,

    /// What sampling temperature to use, between 0 and 2
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// A list of tools the model may call
    pub tools: Option<Vec<Tool>>,

    /// Controls which (if any) tool is called by the model
    pub tool_choice: Option<ToolChoice>,

    /// Effort level for reasoning models (none, low, medium, high)
    pub reasoning_effort: Option<String>,

    /// Whether to enable parallel function calling during tool use
    pub parallel_tool_calls: Option<bool>,

    /// A unique identifier representing your end-user
    pub user: Option<String>,

    // -------- vLLM Sampling Parameters --------
    /// Use beam search instead of sampling
    #[serde(default)]
    pub use_beam_search: bool,

    /// Top-k sampling parameter
    pub top_k: Option<u32>,

    /// Min-p nucleus sampling parameter
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,

    /// Repetition penalty for reducing repetitive text
    #[validate(range(min = 0.0, max = 2.0))]
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
    #[validate(range(min = 1))]
    pub min_tokens: Option<u32>,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Add spaces between special tokens during detokenization
    #[serde(default = "default_true")]
    pub spaces_between_special_tokens: bool,

    /// Truncate prompt tokens to this length
    pub truncate_prompt_tokens: Option<i64>,

    /// Number of prompt logprobs to return
    pub prompt_logprobs: Option<i32>,

    /// Restrict output to these token IDs only
    pub allowed_token_ids: Option<Vec<u32>>,

    /// List of bad words to avoid during generation
    pub bad_words: Option<Vec<String>>,

    // -------- Extra vLLM Parameters --------
    /// Token budget for reasoning/thinking
    pub thinking_token_budget: Option<u32>,

    /// Whether to include reasoning content in the response
    #[serde(default = "default_true")]
    pub include_reasoning: bool,

    /// If true, the new message will be prepended with the last message if they belong to the same
    /// role.
    #[serde(default)]
    pub echo: bool,

    /// Whether to add the generation prompt to the chat template
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,

    /// Continue generating from final assistant message
    #[serde(default)]
    pub continue_final_message: bool,

    /// Whether to add special tokens (e.g. BOS) to the prompt
    #[serde(default)]
    pub add_special_tokens: bool,

    /// Documents for RAG (retrieval-augmented generation)
    pub documents: Option<Vec<Value>>,

    /// Jinja chat template override
    pub chat_template: Option<String>,

    /// Additional keyword args passed to the chat template renderer
    pub chat_template_kwargs: Option<HashMap<String, Value>>,

    /// Additional kwargs for media IO connectors, keyed by modality
    pub media_io_kwargs: Option<HashMap<String, Value>>,

    /// Additional kwargs for the HF processor
    pub mm_processor_kwargs: Option<HashMap<String, Value>>,

    /// Additional kwargs for structured outputs
    pub structured_outputs: Option<Value>,

    /// Request scheduling priority (lower means earlier; default 0)
    pub priority: Option<i32>,

    /// Tokens represented as strings of the form 'token_id:{token_id}' in logprobs
    pub return_tokens_as_token_ids: Option<bool>,

    /// Include token IDs alongside generated text
    pub return_token_ids: Option<bool>,

    /// Salt for prefix cache isolation in multi-user environments
    pub cache_salt: Option<String>,

    /// KV transfer parameters for disaggregated serving
    pub kv_transfer_params: Option<HashMap<String, Value>>,

    /// Additional request parameters with string or numeric values for custom extensions
    pub vllm_xargs: Option<HashMap<String, Value>>,

    /// Parameters for detecting repetitive N-gram patterns in output tokens
    pub repetition_detection: Option<Value>,
}

impl Default for ChatCompletionRequest {
    #[expect(deprecated)]
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            model: default_model(),
            frequency_penalty: None,
            logit_bias: None,
            logprobs: false,
            top_logprobs: None,
            max_tokens: None,
            max_completion_tokens: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            stop: None,
            stream: false,
            stream_options: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            thinking_token_budget: None,
            include_reasoning: true,
            parallel_tool_calls: None,
            user: None,
            use_beam_search: false,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            length_penalty: None,
            stop_token_ids: None,
            include_stop_str_in_output: false,
            ignore_eos: false,
            min_tokens: None,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            truncate_prompt_tokens: None,
            prompt_logprobs: None,
            allowed_token_ids: None,
            bad_words: None,
            echo: false,
            add_generation_prompt: true,
            continue_final_message: false,
            add_special_tokens: false,
            documents: None,
            chat_template: None,
            chat_template_kwargs: None,
            media_io_kwargs: None,
            mm_processor_kwargs: None,
            structured_outputs: None,
            priority: None,
            return_tokens_as_token_ids: None,
            return_token_ids: None,
            cache_salt: None,
            kv_transfer_params: None,
            vllm_xargs: None,
            repetition_detection: None,
        }
    }
}

impl Normalizable for ChatCompletionRequest {
    /// Normalize the request by applying migrations and defaults.
    fn normalize(&mut self) {
        // Migrate deprecated max_tokens → max_completion_tokens
        #[expect(deprecated)]
        if self.max_completion_tokens.is_none() && self.max_tokens.is_some() {
            self.max_completion_tokens = self.max_tokens;
            self.max_tokens = None;
        }

        // Apply tool_choice defaults
        // If tools is None, leave tool_choice as None (don't set it)
        if self.tool_choice.is_none()
            && let Some(tools) = &self.tools
        {
            let choice_value = if tools.is_empty() {
                ToolChoiceValue::None
            } else {
                ToolChoiceValue::Auto
            };
            self.tool_choice = Some(ToolChoice::Value(choice_value));
        }
    }
}

/// Mirrors the Python vLLM `ChatCompletionResponse` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Option<Usage>,
    pub system_fingerprint: Option<String>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<String, f32>>>>,
    pub kv_transfer_params: Option<Value>,
}

/// Mirrors the Python vLLM `ChatCompletionResponseChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
}

/// Mirrors the Python vLLM response `ChatMessage` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct ChatCompletionMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning: Option<String>,
}

/// Mirrors the Python vLLM `ChatCompletionStreamResponse` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub(super) struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionStreamChoice>,
    pub usage: Option<Usage>,
}

impl ChatCompletionStreamResponse {
    /// Create a stream response with the standard envelope fields pre-filled.
    pub fn new(id: &str, model: &str, created: u64) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: None,
        }
    }
}

/// Mirrors the Python vLLM `ChatCompletionResponseStreamChoice` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
pub(super) struct ChatCompletionStreamChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    pub stop_reason: Option<Value>,
}

/// Mirrors the Python vLLM `DeltaMessage` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Serialize)]
pub(super) struct ChatMessageDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    pub reasoning: Option<String>,
}

fn default_model() -> String {
    openai_protocol::UNKNOWN_MODEL_ID.to_string()
}

/// Validates messages array is not empty and has valid content
fn validate_messages(messages: &[ChatMessage]) -> Result<(), validator::ValidationError> {
    if messages.is_empty() {
        return Err(validator::ValidationError::new("messages cannot be empty"));
    }

    for msg in messages {
        if let ChatMessage::User { content, .. } = msg {
            match content {
                MessageContent::Text(text) if text.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content cannot be empty",
                    ));
                }
                MessageContent::Parts(parts) if parts.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content parts cannot be empty",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
fn validate_chat_cross_parameters(
    req: &ChatCompletionRequest,
) -> Result<(), validator::ValidationError> {
    // 1. Validate logprobs dependency
    if req.top_logprobs.is_some() && !req.logprobs {
        let mut e = validator::ValidationError::new("top_logprobs_requires_logprobs");
        e.message = Some("top_logprobs is only allowed when logprobs is enabled".into());
        return Err(e);
    }

    // 2. Validate stream_options dependency
    if req.stream_options.is_some() && !req.stream {
        let mut e = validator::ValidationError::new("stream_options_requires_stream");
        e.message = Some("stream_options can only be used when stream is true".into());
        return Err(e);
    }

    // 3. Validate token limits - min <= max
    if let (Some(min_tokens), Some(max_completion_tokens)) =
        (req.min_tokens, req.max_completion_tokens)
        && min_tokens > max_completion_tokens
    {
        let mut e = validator::ValidationError::new("min_tokens_exceeds_max_completion_tokens");
        e.message = Some("min_tokens cannot be greater than max_completion_tokens".into());
        return Err(e);
    }

    #[expect(deprecated, reason = "Local type still mirrors legacy upstream field")]
    if let (Some(min_tokens), Some(max_tokens)) = (req.min_tokens, req.max_tokens)
        && min_tokens > max_tokens
    {
        let mut e = validator::ValidationError::new("min_tokens_exceeds_max_tokens");
        e.message = Some("min_tokens cannot be greater than max_tokens".into());
        return Err(e);
    }

    // 4. Validate response format JSON schema name
    if let Some(ResponseFormat::JsonSchema { json_schema }) = &req.response_format
        && json_schema.name.is_empty()
    {
        let mut e = validator::ValidationError::new("json_schema_name_empty");
        e.message = Some("JSON schema name cannot be empty".into());
        return Err(e);
    }

    // 5. Validate tool_choice requires tools (except for "none")
    if let Some(ref tool_choice) = req.tool_choice {
        let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());

        // Check if tool_choice is anything other than "none"
        let is_some_choice = !matches!(tool_choice, ToolChoice::Value(ToolChoiceValue::None));

        if is_some_choice && !has_tools {
            let mut e = validator::ValidationError::new("tool_choice_requires_tools");
            e.message = Some("Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.".into());
            return Err(e);
        }

        // Additional validation when tools are present
        if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) {
            match tool_choice {
                ToolChoice::Function { function, .. } => {
                    // Validate that the specified function name exists in tools
                    let function_exists = tools.iter().any(|tool| {
                        tool.tool_type == "function" && tool.function.name == function.name
                    });

                    if !function_exists {
                        let mut e =
                            validator::ValidationError::new("tool_choice_function_not_found");
                        e.message = Some(
                            format!(
                            "Invalid value for 'tool_choice': function '{}' not found in 'tools'.",
                            function.name
                        )
                            .into(),
                        );
                        return Err(e);
                    }
                }
                ToolChoice::AllowedTools {
                    mode,
                    tools: allowed_tools,
                    ..
                } => {
                    // Validate mode is "auto" or "required"
                    if mode != "auto" && mode != "required" {
                        let mut e = validator::ValidationError::new("tool_choice_invalid_mode");
                        e.message = Some(format!(
                            "Invalid value for 'tool_choice.mode': must be 'auto' or 'required', got '{mode}'."
                        ).into());
                        return Err(e);
                    }

                    // Validate that all ToolReferences are Function type (Chat API only supports
                    // function tools)
                    for tool_ref in allowed_tools {
                        match tool_ref {
                            ToolReference::Function { name } => {
                                // Validate that the function exists in tools array
                                let tool_exists = tools.iter().any(|tool| {
                                    tool.tool_type == "function" && tool.function.name == *name
                                });

                                if !tool_exists {
                                    let mut e = validator::ValidationError::new(
                                        "tool_choice_tool_not_found",
                                    );
                                    e.message = Some(
                                        format!(
                                            "Invalid value for 'tool_choice.tools': tool '{name}' not found in 'tools'."
                                        )
                                        .into(),
                                    );
                                    return Err(e);
                                }
                            }
                            _ => {
                                // Chat Completion API only supports function tools in tool_choice
                                let mut e = validator::ValidationError::new(
                                    "tool_choice_invalid_tool_type",
                                );
                                e.message = Some(
                                    format!(
                                        "Invalid value for 'tool_choice.tools': Chat Completion API only supports function tools, got '{}'.",
                                        tool_ref.identifier()
                                    )
                                    .into(),
                                );
                                return Err(e);
                            }
                        }
                    }
                }
                ToolChoice::Value(_) => {}
            }
        }
    }

    Ok(())
}
