use std::collections::HashMap;
use std::slice;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// Constants
// ============================================================================

/// Default model identifier used when no model is specified.
pub const UNKNOWN_MODEL_ID: &str = "unknown";

// ============================================================================
// Default value helpers
// ============================================================================

/// Helper function for serde default value (returns true).
pub fn default_true() -> bool {
    true
}

// ============================================================================
// String/Array Utilities
// ============================================================================

/// A type that can be either a single string or an array of strings.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn as_slice(&self) -> &[String] {
        match self {
            StringOrArray::String(s) => slice::from_ref(s),
            StringOrArray::Array(arr) => arr,
        }
    }

    #[allow(unused)]
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s],
            StringOrArray::Array(arr) => arr,
        }
    }
}

/// Validates stop sequences (non-empty strings)
pub fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    if stop.as_slice().iter().any(|s| s.is_empty()) {
        return Err(validator::ValidationError::new(
            "stop strings cannot be empty",
        ));
    }
    Ok(())
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validates top_p: 0.0 < top_p <= 1.0.
pub fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

// ============================================================================
// Content Parts (for multimodal messages)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "video_url")]
    VideoUrl { video_url: VideoUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VideoUrl {
    pub url: String,
}

// ============================================================================
// Streaming
// ============================================================================

/// Mirrors the Python vLLM `StreamOptions` class.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
    pub continuous_usage_stats: Option<bool>,
}

// ============================================================================
// Tools and Function Calling
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Function,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    /// Whether to enable strict schema adherence (OpenAI structured outputs).
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Tool choice value for simple string options.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    Required,
    None,
}

/// Tool choice for the Chat Completion API.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Value(ToolChoiceValue),
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        function: FunctionChoice,
    },
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: String,
        mode: String,
        tools: Vec<ToolReference>,
    },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Value(ToolChoiceValue::Auto)
    }
}

/// Function choice specification for `ToolChoice::Function`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// Tool reference for `ToolChoice::AllowedTools`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ToolReference {
    #[serde(rename = "function")]
    Function { name: String },
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "file_search")]
    FileSearch,
    #[serde(rename = "web_search_preview")]
    WebSearchPreview,
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "image_generation")]
    ImageGeneration,
}

impl ToolReference {
    /// Get a unique identifier for this tool reference.
    pub fn identifier(&self) -> String {
        match self {
            ToolReference::Function { name } => format!("function:{name}"),
            ToolReference::Mcp {
                server_label,
                name: Some(n),
            } => format!("mcp:{server_label}:{n}"),
            ToolReference::Mcp {
                server_label,
                name: _,
            } => format!("mcp:{server_label}"),
            ToolReference::FileSearch => "file_search".to_string(),
            ToolReference::WebSearchPreview => "web_search_preview".to_string(),
            ToolReference::ComputerUsePreview => "computer_use_preview".to_string(),
            ToolReference::CodeInterpreter => "code_interpreter".to_string(),
            ToolReference::ImageGeneration => "image_generation".to_string(),
        }
    }
}

// ============================================================================
// Chat Messages
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        name: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<MessageContent>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        /// Reasoning content for O1-style models.
        reasoning_content: Option<String>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: MessageContent,
        tool_call_id: String,
    },
    #[serde(rename = "function")]
    Function { content: String, name: String },
    #[serde(rename = "developer")]
    Developer {
        content: MessageContent,
        tools: Option<Vec<Tool>>,
        name: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

// ============================================================================
// Usage and Logging
// ============================================================================

/// Mirrors the Python vLLM `UsageInfo` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
    pub completion_tokens: Option<u32>,
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

impl Usage {
    /// Create a Usage from prompt and completion token counts.
    pub fn from_counts(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens: Some(completion_tokens),
            prompt_tokens_details: None,
        }
    }
}

/// Mirrors the Python vLLM `PromptTokenUsageInfo` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: Option<u32>,
}

/// OpenAI completions-style logprobs.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

/// Mirrors the Python vLLM `ChatCompletionLogProbs` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbs {
    pub content: Option<Vec<ChatLogProbsContent>>,
}

/// Mirrors the Python vLLM `ChatCompletionLogProbsContent` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

/// Mirrors the Python vLLM `ChatCompletionLogProb` class.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

// ============================================================================
// Model types
// ============================================================================

/// A single model entry in the `/v1/models` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Response body for `GET /v1/models`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

// ============================================================================
// Normalizable trait
// ============================================================================

/// Trait for request types that need post-deserialization normalization.
pub trait Normalizable {
    /// Normalize the request by applying defaults and transformations.
    fn normalize(&mut self) {
        // Default: no-op
    }
}
