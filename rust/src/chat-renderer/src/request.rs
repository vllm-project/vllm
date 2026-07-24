// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::Value;
use vllm_chat_types::{
    ChatMessage, ChatOptions, ChatRole, ChatToolChoice, GenerationPromptMode, Tool,
};

use crate::{Error, Result};

/// Borrowed input consumed by one chat renderer.
///
/// This view contains only chat-domain values that affect prompt construction.
/// Serving, sampling, decoding, scheduling, and engine metadata stay with the
/// caller.
#[derive(Debug, Clone, Copy)]
pub struct RenderRequest<'a> {
    /// Ordered chat history to render.
    pub messages: &'a [ChatMessage],
    /// Chat-template and generation-prompt controls.
    pub chat_options: &'a ChatOptions,
    /// Request-level tools available to the model.
    pub tools: &'a [Tool],
    /// Tool-choice behavior used to decide whether tools are exposed.
    pub tool_choice: &'a ChatToolChoice,
    /// Optional retrieval documents exposed to HF chat templates.
    pub documents: Option<&'a [Value]>,
}

impl RenderRequest<'_> {
    /// Validate renderer-owned request invariants.
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

    /// Return whether any message contains multimodal content.
    pub fn has_multimodal(&self) -> bool {
        self.messages.iter().any(ChatMessage::has_multimodal)
    }

    /// Return whether request-level tools should be exposed to the renderer.
    pub fn tool_parsing_enabled(&self) -> bool {
        !matches!(self.tool_choice, ChatToolChoice::None) && !self.tools.is_empty()
    }

    /// Return the request-level thinking toggle when explicitly requested.
    ///
    /// The `thinking` and `enable_thinking` kwargs must be booleans when
    /// present and must carry the same value when both are set.
    pub fn enable_thinking(&self) -> Result<Option<bool>> {
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

    /// Parse one optional boolean chat-template kwarg.
    pub fn parse_template_bool(&self, key: &str) -> Result<Option<bool>> {
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

    use super::RenderRequest;
    use crate::{ChatMessage, ChatToolChoice, Error, GenerationPromptMode, TestRenderRequest};

    #[test]
    fn rejects_empty_message_history() {
        let mut request = TestRenderRequest::for_test();
        request.messages.clear();

        assert!(matches!(
            request.as_request().validate(),
            Err(Error::EmptyMessages)
        ));
    }

    #[test]
    fn continuation_requires_a_final_assistant_message() {
        let mut request = TestRenderRequest::for_test();
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;

        assert!(matches!(
            request.as_request().validate(),
            Err(Error::ContinueFinalAssistantWithoutFinalAssistant)
        ));

        request.messages.push(ChatMessage::assistant_text("partial"));
        request.as_request().validate().unwrap();
    }

    #[test]
    fn thinking_aliases_must_match() {
        let mut request = TestRenderRequest::for_test();
        request
            .chat_options
            .template_kwargs
            .insert("thinking".to_string(), json!(false));
        request
            .chat_options
            .template_kwargs
            .insert("enable_thinking".to_string(), json!(true));

        assert!(matches!(
            request.as_request().enable_thinking(),
            Err(Error::ChatTemplate(message))
                if message.contains("`thinking` and `enable_thinking` must match")
        ));
    }

    #[test]
    fn tool_exposure_requires_tools_and_an_enabled_choice() {
        let mut request = TestRenderRequest::for_test();
        assert!(!request.as_request().tool_parsing_enabled());

        request.tools.push(crate::Tool {
            name: "lookup".to_string(),
            description: None,
            parameters: json!({"type": "object"}),
            strict: None,
        });
        assert!(!request.as_request().tool_parsing_enabled());

        request.tool_choice = ChatToolChoice::Auto;
        assert!(request.as_request().tool_parsing_enabled());
    }

    #[test]
    fn borrowed_request_carries_optional_documents() {
        let mut request = TestRenderRequest::for_test();
        request.documents = Some(vec![json!({"title": "doc"})]);

        let borrowed: RenderRequest<'_> = request.as_request();
        assert_eq!(
            borrowed.documents.unwrap(),
            request.documents.as_deref().unwrap()
        );
    }
}
