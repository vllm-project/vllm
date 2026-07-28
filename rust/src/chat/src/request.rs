// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use serde_json::Value;
use vllm_chat_renderer::RenderRequest;
pub use vllm_chat_types::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRole, ChatToolChoice,
    GenerationPromptMode, ImageDetail, ReasoningEffort, Tool as ChatTool,
};
use vllm_engine_core_client::protocol::lora::LoraRequest;
pub use vllm_text::SamplingParams;
use vllm_text::TextDecodeOptions;

use crate::error::Result;

/// One chat request ready to be rendered into a prompt and lowered into a
/// generate request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Stable caller-supplied request ID.
    pub request_id: String,
    /// Ordered chat history to render.
    pub messages: Vec<ChatMessage>,
    /// User-facing sampling parameters accepted by `vllm-chat`.
    pub sampling_params: SamplingParams,
    /// Chat-specific rendering options.
    pub chat_options: ChatOptions,
    /// Function tools made available to the model for this request.
    pub tools: Vec<ChatTool>,
    /// Tool-choice behavior for this request.
    pub tool_choice: ChatToolChoice,
    /// Whether the model may return more than one tool call per response.
    ///
    /// When `false`, only the first parsed tool call is surfaced northbound.
    pub parallel_tool_calls: bool,
    /// Text decode options for incremental detokenization.
    pub decode_options: TextDecodeOptions,
    /// Whether to emit intermediate northbound content deltas before the
    /// terminal result.
    ///
    /// If `false`, callers only observe the terminal accumulated assistant
    /// output. If `true`, callers may receive zero or more incremental
    /// content events before the final terminal one.
    pub intermediate: bool,
    /// Request scheduling priority (lower means earlier handling; default 0).
    pub priority: i32,
    /// Documents for RAG (retrieval-augmented generation), passed to the chat
    /// template.
    pub documents: Option<Vec<Value>>,
    /// Salt for prefix cache isolation in multi-user environments.
    pub cache_salt: Option<String>,
    /// Whether to add special tokens (e.g. BOS) during prompt tokenization.
    pub add_special_tokens: bool,
    /// Override data parallel rank.
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
    /// LoRA adapter selected for this request.
    #[serde(default)]
    pub lora_request: Option<LoraRequest>,
}

impl ChatRequest {
    /// Borrow the renderer-owned subset of this serving request.
    pub fn as_render_request(&self) -> RenderRequest<'_> {
        RenderRequest {
            messages: &self.messages,
            chat_options: &self.chat_options,
            tools: &self.tools,
            tool_choice: &self.tool_choice,
            documents: self.documents.as_deref(),
        }
    }

    /// Return one minimal valid request fixture for tests.
    pub fn for_test() -> Self {
        Self {
            request_id: "test-request".to_string(),
            messages: vec![ChatMessage::text(ChatRole::User, "test")],
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions::default(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            parallel_tool_calls: true,
            decode_options: TextDecodeOptions::default(),
            intermediate: true,
            priority: 0,
            documents: None,
            cache_salt: None,
            add_special_tokens: false,
            data_parallel_rank: None,
            lora_request: None,
        }
    }

    /// Validate basic request invariants before rendering.
    pub fn validate(&self) -> Result<()> {
        self.as_render_request().validate().map_err(Into::into)
    }

    /// Return true if this request contains any multimodal content in its
    /// messages.
    pub fn has_multimodal(&self) -> bool {
        self.as_render_request().has_multimodal()
    }

    /// Return true if this request should enable tool parsing based on the tool
    /// choice and tool list.
    pub(crate) fn tool_parsing_enabled(&self) -> bool {
        self.as_render_request().tool_parsing_enabled()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ChatRequest;

    #[test]
    fn enable_thinking_is_none_when_no_kwargs_are_present() {
        let request = ChatRequest::for_test();
        assert_eq!(request.as_render_request().enable_thinking().unwrap(), None);
    }

    #[test]
    fn enable_thinking_accepts_matching_duplicate_kwargs() {
        let mut request = ChatRequest::for_test();
        request.chat_options.template_kwargs.insert("thinking".to_string(), json!(true));
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), json!(true));

        assert_eq!(
            request.as_render_request().enable_thinking().unwrap(),
            Some(true)
        );
    }

    #[test]
    fn enable_thinking_rejects_non_boolean_kwargs() {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("thinking".to_string(), json!("yes"));

        assert!(matches!(
            request.as_render_request().enable_thinking(),
            Err(vllm_chat_renderer::Error::ChatTemplate(message))
                if message.contains("`thinking` must be a boolean")
        ));
    }

    #[test]
    fn enable_thinking_rejects_conflicting_duplicate_kwargs() {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("thinking".to_string(), json!(false));
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), json!(true));

        assert!(matches!(
            request.as_render_request().enable_thinking(),
            Err(vllm_chat_renderer::Error::ChatTemplate(message))
                if message.contains("`thinking` and `enable_thinking` must match")
        ));
    }
}
