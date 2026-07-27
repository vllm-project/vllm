// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod openai_chat;
pub mod openai_completions;
pub mod pooling;
pub mod streaming;

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

// --- Typed SSE chunk structs for zero-alloc deserialization ---
// Using typed deserialization avoids building a full serde_json::Value tree.
// Only the fields we need are extracted; everything else is skipped by serde.

/// Completions API streaming chunk (minimal fields).
#[derive(Deserialize)]
pub struct CompletionChunk {
    #[serde(default)]
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<ChunkUsage>,
}

#[derive(Deserialize)]
pub struct CompletionChoice {
    pub text: Option<String>,
}

/// Chat API streaming chunk (minimal fields).
#[derive(Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChunkUsage>,
}

#[derive(Deserialize)]
pub struct ChatChoice {
    pub delta: Option<ChatDelta>,
}

#[derive(Deserialize)]
pub struct ChatDelta {
    pub content: Option<String>,
}

#[derive(Deserialize)]
pub struct ChunkUsage {
    pub completion_tokens: Option<u64>,
}

use crate::cli::BackendKind;
use crate::error::Result;

/// Input for a single benchmark request.
#[derive(Debug, Clone)]
pub struct RequestFuncInput {
    pub prompt: Arc<str>,
    pub api_url: String,
    pub prompt_len: usize,
    pub output_len: usize,
    pub model: String,
    pub model_name: Option<String>,
    pub logprobs: Option<usize>,
    pub extra_headers: Option<HashMap<String, String>>,
    pub extra_body: Option<serde_json::Value>,
    pub ignore_eos: bool,
    pub request_id: Option<String>,
    /// Pre-built messages array for multi-turn conversations.
    /// When set, the chat backend uses this instead of building from `prompt`.
    pub messages: Option<serde_json::Value>,
    /// Pre-computed token IDs for this prompt.
    /// When set, the completions backend sends these directly via `prompt_token_ids`
    /// instead of the text `prompt`, skipping server-side tokenization.
    pub prompt_token_ids: Option<Arc<[u32]>>,
    /// Multimodal content as pre-serialized JSON fragments.
    /// When set, the chat backend concatenates these directly into the payload bytes,
    /// avoiding any parsing or deep-cloning of base64 image data.
    pub multi_modal_content: Option<Arc<[Arc<str>]>>,
    /// Complete pre-serialized chat `messages` array (--enable-multimodal-chat).
    /// When set, the chat backend splices it verbatim into the payload bytes,
    /// taking precedence over `messages`, `prompt`, and `multi_modal_content`.
    pub chat_messages_json: Option<Arc<str>>,
    /// Multiple text inputs for one request (pooling backends only):
    /// embeddings batch (`"input": [...]`) or rerank query+documents.
    pub prompt_list: Option<Arc<[Arc<str>]>>,
}

/// Output from a single benchmark request including timing metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestFuncOutput {
    pub generated_text: String,
    pub success: bool,
    pub latency: f64,
    pub output_tokens: usize,
    pub ttft: f64,
    pub itl: Vec<f64>,
    pub tpot: f64,
    pub prompt_len: usize,
    pub error: String,
    pub start_time: f64,
}

impl Default for RequestFuncOutput {
    fn default() -> Self {
        Self {
            generated_text: String::new(),
            success: false,
            latency: 0.0,
            output_tokens: 0,
            ttft: 0.0,
            itl: Vec::new(),
            tpot: 0.0,
            prompt_len: 0,
            error: String::new(),
            start_time: 0.0,
        }
    }
}

impl Default for RequestFuncInput {
    fn default() -> Self {
        Self {
            prompt: Arc::from(""),
            api_url: String::new(),
            prompt_len: 0,
            output_len: 0,
            model: String::new(),
            model_name: None,
            logprobs: None,
            extra_headers: None,
            extra_body: None,
            ignore_eos: false,
            request_id: None,
            messages: None,
            prompt_token_ids: None,
            multi_modal_content: None,
            chat_messages_json: None,
            prompt_list: None,
        }
    }
}

/// Enum dispatch for backend implementations (avoids async trait object issues).
#[derive(Clone)]
pub enum Backend {
    OpenAICompletions(openai_completions::OpenAICompletionsBackend),
    OpenAIChat(openai_chat::OpenAIChatBackend),
    Pooling(pooling::PoolingBackend),
}

impl Backend {
    /// Send a single request and collect timing metrics.
    pub async fn send_request(
        &self,
        input: &RequestFuncInput,
        client: &reqwest::Client,
    ) -> Result<RequestFuncOutput> {
        match self {
            Backend::OpenAICompletions(b) => b.send_request(input, client).await,
            Backend::OpenAIChat(b) => b.send_request(input, client).await,
            Backend::Pooling(b) => b.send_request(input, client).await,
        }
    }
}

/// Get a backend by kind.
pub fn get_backend(kind: BackendKind) -> Result<Backend> {
    match kind {
        BackendKind::Vllm | BackendKind::Openai => Ok(Backend::OpenAICompletions(
            openai_completions::OpenAICompletionsBackend,
        )),
        BackendKind::OpenaiChat => Ok(Backend::OpenAIChat(openai_chat::OpenAIChatBackend)),
        kind if kind.is_pooling() => Ok(Backend::Pooling(pooling::PoolingBackend { kind })),
        _ => unreachable!(),
    }
}

/// Cached API key to avoid per-request env var syscall.
static API_KEY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

fn cached_api_key() -> &'static Option<String> {
    API_KEY.get_or_init(|| std::env::var("OPENAI_API_KEY").ok())
}

/// Build common headers including auth and extras.
pub fn build_headers(
    content_type: Option<&str>,
    extra_headers: &Option<HashMap<String, String>>,
    request_id: &Option<String>,
) -> HashMap<String, String> {
    let mut headers = HashMap::new();

    if let Some(ct) = content_type {
        headers.insert("Content-Type".to_string(), ct.to_string());
    }

    if let Some(api_key) = cached_api_key() {
        headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
    }

    if let Some(extra) = extra_headers {
        headers.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    if let Some(rid) = request_id {
        headers.insert("x-request-id".to_string(), rid.clone());
    }

    headers
}
