// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use serde_json::Value;
pub use vllm_chat_types::{
    ChatContent, ChatContentPart, ChatMessage, ChatOptions, ChatRole, ChatToolChoice,
    GenerationPromptMode, ImageDetail, ReasoningEffort, Tool as ChatTool,
};
use vllm_engine_core_client::protocol::lora::LoraRequest;
pub use vllm_text::SamplingParams;
use vllm_text::TextDecodeOptions;

use crate::error::{Error, Result};

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
        if self.messages.is_empty() {
            return Err(Error::EmptyMessages);
        }
        match (
            self.chat_options.generation_prompt_mode,
            self.messages.last().map(ChatMessage::role),
        ) {
            (GenerationPromptMode::ContinueFinalAssistant, Some(ChatRole::Assistant)) => {}
            (GenerationPromptMode::ContinueFinalAssistant, _) => {
                return Err(Error::ContinueFinalAssistantWithoutFinalAssistant);
            }
            (GenerationPromptMode::NoGenerationPrompt, _)
            | (GenerationPromptMode::StartNewAssistant, _) => {}
        }
        Ok(())
    }

    /// Return true if this request contains any multimodal content in its
    /// messages.
    pub fn has_multimodal(&self) -> bool {
        self.messages.iter().any(ChatMessage::has_multimodal)
    }

    /// Return true if this request should enable tool parsing based on the tool
    /// choice and tool list.
    pub(crate) fn tool_parsing_enabled(&self) -> bool {
        !matches!(self.tool_choice, ChatToolChoice::None) && !self.tools.is_empty()
    }

    /// Return the request-level thinking toggle when explicitly requested.
    ///
    /// We currently accept the two request kwargs `thinking` and
    /// `enable_thinking`. Both must be booleans when present. If both are
    /// present, they must have the same value. If neither key is provided,
    /// return `None`.
    pub(crate) fn enable_thinking(&self) -> Result<Option<bool>> {
        let thinking = self.parse_template_bool("thinking")?;
        let enable_thinking = self.parse_template_bool("enable_thinking")?;

        match (thinking, enable_thinking) {
            (None, None) => Ok(None),
            (Some(thinking), Some(enable_thinking)) if thinking != enable_thinking => {
                Err(Error::ChatTemplate(
                    "template kwargs `thinking` and `enable_thinking` must match when both are set"
                        .to_string(),
                ))
            }
            (Some(thinking), _) => Ok(Some(thinking)),
            (None, Some(enable_thinking)) => Ok(Some(enable_thinking)),
        }
    }

    pub(crate) fn parse_template_bool(&self, key: &str) -> Result<Option<bool>> {
        match self.chat_options.template_kwargs.get(key) {
            None => Ok(None),
            Some(Value::Bool(value)) => Ok(Some(*value)),
            Some(other) => Err(Error::ChatTemplate(format!(
                "template kwarg `{key}` must be a boolean, got {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ChatRequest;
    use crate::Error;

    #[test]
    fn enable_thinking_is_none_when_no_kwargs_are_present() {
        let request = ChatRequest::for_test();
        assert_eq!(request.enable_thinking().unwrap(), None);
    }

    #[test]
    fn enable_thinking_accepts_matching_duplicate_kwargs() {
        let mut request = ChatRequest::for_test();
        request.chat_options.template_kwargs.insert("thinking".to_string(), json!(true));
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), json!(true));

        assert_eq!(request.enable_thinking().unwrap(), Some(true));
    }

    #[test]
    fn enable_thinking_rejects_non_boolean_kwargs() {
        let mut request = ChatRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("thinking".to_string(), json!("yes"));

        assert!(matches!(
            request.enable_thinking(),
            Err(Error::ChatTemplate(message))
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
            request.enable_thinking(),
            Err(Error::ChatTemplate(message))
                if message.contains("`thinking` and `enable_thinking` must match")
        ));
    }
}
