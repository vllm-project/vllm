use std::collections::HashMap;

use openai_protocol::chat::{ChatMessage, MessageContent};
use openai_protocol::common::{
    Function, FunctionCall, FunctionChoice, ResponseFormat, StreamOptions, StringOrArray, Tool,
    ToolChoice, ToolChoiceValue, ToolReference, default_true, validate_stop,
};
use openai_protocol::sampling_params::{validate_top_k_value, validate_top_p_value};
use openai_protocol::validated::Normalizable;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

/// vLLM-compatible request type for the Chat Completions API.
///
/// This mirrors [`openai_protocol::chat::ChatCompletionRequest`], but keeps the
/// request type local to the route module so we can extend it directly with
/// vLLM-compatible fields instead of layering wrapper deserializers on top.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Default, Validate)]
#[validate(schema(function = "validate_chat_cross_parameters"))]
pub struct ChatCompletionRequest {
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

    /// Deprecated: Replaced by tool_choice
    #[deprecated(note = "Use tool_choice instead")]
    pub function_call: Option<FunctionCall>,

    /// Deprecated: Replaced by tools
    #[deprecated(note = "Use tools instead")]
    pub functions: Option<Vec<Function>>,

    /// Modify the likelihood of specified tokens appearing in the completion
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Whether to return log probabilities of the output tokens
    #[serde(default)]
    pub logprobs: bool,

    /// Deprecated: Replaced by max_completion_tokens
    #[deprecated(note = "Use max_completion_tokens instead")]
    #[validate(range(min = 1))]
    pub max_tokens: Option<u32>,

    /// An upper bound for the number of tokens that can be generated for a completion
    #[validate(range(min = 1))]
    pub max_completion_tokens: Option<u32>,

    /// Developer-defined tags and values used for filtering completions in the dashboard
    pub metadata: Option<HashMap<String, String>>,

    /// Output types that you would like the model to generate for this request
    pub modalities: Option<Vec<String>>,

    /// How many chat completion choices to generate for each input message
    #[validate(range(min = 1, max = 10))]
    pub n: Option<u32>,

    /// Whether to enable parallel function calling during tool use
    pub parallel_tool_calls: Option<bool>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// Cache key for prompts (beta feature)
    pub prompt_cache_key: Option<String>,

    /// Effort level for reasoning models (low, medium, high)
    pub reasoning_effort: Option<String>,

    /// An object specifying the format that the model must output
    pub response_format: Option<ResponseFormat>,

    /// Safety identifier for content moderation
    pub safety_identifier: Option<String>,

    /// Deprecated: This feature is in Legacy mode
    #[deprecated(note = "This feature is in Legacy mode")]
    pub seed: Option<i64>,

    /// The service tier to use for this request
    pub service_tier: Option<String>,

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

    /// Controls which (if any) tool is called by the model
    pub tool_choice: Option<ToolChoice>,

    /// A list of tools the model may call
    pub tools: Option<Vec<Tool>>,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return
    #[validate(range(min = 0, max = 20))]
    pub top_logprobs: Option<u32>,

    /// An alternative to sampling with temperature
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// Verbosity level for debugging
    pub verbosity: Option<i32>,

    // =============================================================================
    // Engine-Specific Sampling Parameters
    // =============================================================================
    // These parameters are extensions beyond the OpenAI API specification and
    // control model generation behavior in engine-specific ways.
    // =============================================================================
    /// Top-k sampling parameter (-1 to disable)
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    #[validate(range(min = 1))]
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    pub ebnf: Option<String>,

    /// Specific token IDs to use as stop conditions
    pub stop_token_ids: Option<Vec<u32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Continue generating from final assistant message
    #[serde(default)]
    pub continue_final_message: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Path to LoRA adapter(s) for model customization
    pub lora_path: Option<String>,

    /// Session parameters for continual prompting
    pub session_params: Option<HashMap<String, Value>>,

    /// Separate reasoning content from final answer (O1-style models)
    #[serde(default = "default_true")]
    pub separate_reasoning: bool,

    /// Stream reasoning tokens during generation
    #[serde(default = "default_true")]
    pub stream_reasoning: bool,

    /// Chat template kwargs
    pub chat_template_kwargs: Option<HashMap<String, Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Random seed for sampling for deterministic outputs
    pub sampling_seed: Option<u64>,

    /// If true, the new message will be prepended with the last message if they belong to the same
    /// role.
    #[serde(default)]
    pub echo: bool,

    /// vLLM-compatible prompt logprobs request field missing from `openai-protocol`.
    pub prompt_logprobs: Option<i32>,
}

impl Normalizable for ChatCompletionRequest {
    /// Normalize the request by applying migrations and defaults:
    /// 1. Migrate deprecated fields to their replacements
    /// 2. Clear deprecated fields and log warnings
    /// 3. Apply OpenAI defaults for tool_choice
    fn normalize(&mut self) {
        // Migrate deprecated max_tokens → max_completion_tokens
        #[expect(deprecated)]
        if self.max_completion_tokens.is_none() && self.max_tokens.is_some() {
            self.max_completion_tokens = self.max_tokens;
            self.max_tokens = None; // Clear deprecated field
        }

        // Migrate deprecated functions → tools
        #[expect(deprecated)]
        if self.tools.is_none() && self.functions.is_some() {
            tracing::warn!("functions is deprecated, use tools instead");
            self.tools = self.functions.as_ref().map(|functions| {
                functions
                    .iter()
                    .map(|func| Tool {
                        tool_type: "function".to_string(),
                        function: func.clone(),
                    })
                    .collect()
            });
            self.functions = None; // Clear deprecated field
        }

        // Migrate deprecated function_call → tool_choice
        #[expect(deprecated)]
        if self.tool_choice.is_none() && self.function_call.is_some() {
            tracing::warn!("function_call is deprecated, use tool_choice instead");
            self.tool_choice = self.function_call.as_ref().map(|fc| match fc {
                FunctionCall::None => ToolChoice::Value(ToolChoiceValue::None),
                FunctionCall::Auto => ToolChoice::Value(ToolChoiceValue::Auto),
                FunctionCall::Function { name } => ToolChoice::Function {
                    tool_type: "function".to_string(),
                    function: FunctionChoice { name: name.clone() },
                },
            });
            self.function_call = None; // Clear deprecated field
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

    // 4. Validate structured output conflicts
    let has_json_format = matches!(
        req.response_format,
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    );

    if has_json_format && (req.regex.is_some() || req.ebnf.is_some()) {
        let mut e = validator::ValidationError::new("response_format_conflicts_with_constraints");
        e.message = Some("response_format cannot be used together with regex or ebnf".into());
        return Err(e);
    }

    // 5. Validate mutually exclusive structured output constraints
    if req.regex.is_some() && req.ebnf.is_some() {
        let mut e = validator::ValidationError::new("regex_conflicts_with_ebnf");
        e.message = Some("regex and ebnf cannot both be specified".into());
        return Err(e);
    }

    // 6. Validate response format JSON schema name
    if let Some(ResponseFormat::JsonSchema { json_schema }) = &req.response_format
        && json_schema.name.is_empty()
    {
        let mut e = validator::ValidationError::new("json_schema_name_empty");
        e.message = Some("JSON schema name cannot be empty".into());
        return Err(e);
    }

    // 7. Validate tool_choice requires tools (except for "none")
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
