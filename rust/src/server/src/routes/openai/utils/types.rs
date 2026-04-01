use serde::{Deserialize, Serialize};

/// Mirrors the Python vLLM `StreamOptions` class.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
    pub continuous_usage_stats: Option<bool>,
}

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
