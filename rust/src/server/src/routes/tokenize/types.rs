// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use itertools::Itertools as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::{Validate, ValidationErrors};
use vllm_chat::{ChatOptions, ChatRequest, ChatToolChoice, SamplingParams};
use vllm_text::output::TextDecodeOptions;

use crate::error::ApiError;
use crate::routes::openai::chat_completions::convert::{
    convert_message, convert_tools, normalize_generation_prompt_mode,
};
use crate::routes::openai::utils::types::{
    ChatMessage, Normalizable, Tool, default_true, validate_messages,
};

/// `POST /tokenize` body. Untagged: a JSON object with `messages` parses as the
/// chat variant; one with `prompt` parses as the completion variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum TokenizeRequest {
    Chat(TokenizeChatRequest),
    Completion(TokenizeCompletionRequest),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeCompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    #[serde(default)]
    pub return_token_strs: bool,
}

#[derive(Debug, Clone, Deserialize, Validate)]
pub struct TokenizeChatRequest {
    pub model: Option<String>,
    #[validate(custom(function = "validate_messages"))]
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default)]
    pub continue_final_message: bool,
    #[serde(default)] // chat default is FALSE (template adds specials)
    pub add_special_tokens: bool,
    #[serde(default)]
    pub return_token_strs: bool,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
}

impl TokenizeChatRequest {
    /// Lower this tokenize body into a [`ChatRequest`] for template rendering.
    ///
    /// Reuses [`convert_message`] and [`normalize_generation_prompt_mode`] from
    /// `chat_completions/convert` so message lowering and generation-prompt
    /// rules match chat completions. Only fields that affect rendering are set;
    /// `sampling_params`, `decode_options`, etc. stay at default because
    /// tokenize never generates.
    pub fn into_chat_request(self, request_id: String) -> Result<ChatRequest, ApiError> {
        let messages: Vec<_> = self.messages.into_iter().map(convert_message).try_collect()?;
        let generation_prompt_mode = normalize_generation_prompt_mode(
            Some(self.add_generation_prompt),
            self.continue_final_message,
            &messages,
        )?;

        Ok(ChatRequest {
            request_id,
            messages,
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions {
                generation_prompt_mode,
                chat_template: self.chat_template,
                reasoning_effort: None,
                template_kwargs: self.chat_template_kwargs.unwrap_or_default(),
            },
            tools: convert_tools(self.tools)?,
            tool_choice: ChatToolChoice::Auto,
            parallel_tool_calls: true,
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            priority: 0,
            documents: None,
            cache_salt: None,
            add_special_tokens: self.add_special_tokens,
            data_parallel_rank: None,
            lora_request: None,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    pub model: Option<String>,
    pub tokens: Vec<u32>,
}

/// Do not skip serializing `None` fields here: non-streaming response types
/// should serialize `None` as explicit `null`.
#[derive(Debug, Clone, Serialize)]
pub struct TokenizeResponse {
    pub count: usize,
    pub max_model_len: u32,
    pub tokens: Vec<u32>,
    pub token_strs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetokenizeResponse {
    pub prompt: String,
}

// ---- trait impls required by ValidatedJson ----
impl Validate for TokenizeRequest {
    fn validate(&self) -> Result<(), ValidationErrors> {
        if let Self::Chat(req) = self {
            req.validate()?;
        }
        Ok(())
    }
}
impl Validate for DetokenizeRequest {
    fn validate(&self) -> Result<(), ValidationErrors> {
        Ok(())
    }
}
impl Normalizable for TokenizeRequest {} // default no-op normalize()
impl Normalizable for DetokenizeRequest {}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_chat::ChatTool;

    use super::*;
    use crate::routes::openai::utils::types::{ChatMessage, MessageContent};

    #[test]
    fn tokenize_request_converts_openai_tools() {
        // The untagged `TokenizeRequest` must resolve a messages+tools body to
        // the chat variant and accept standard OpenAI tool objects
        // (`{"type":"function",...}`), then convert them to `ChatTool`.
        let request: TokenizeRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }],
        }))
        .expect("OpenAI tool JSON deserializes to the chat variant");

        let TokenizeRequest::Chat(req) = request else {
            panic!("messages+tools body should parse as the chat variant");
        };

        let chat_request =
            req.into_chat_request("tokenize-test".to_string()).expect("request is valid");

        assert_eq!(
            chat_request.tools,
            vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }),
                strict: None,
            }]
        );
    }

    #[test]
    fn into_chat_request_rejects_conflicting_generation_flags() {
        let req = TokenizeChatRequest {
            model: None,
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            add_generation_prompt: true,
            continue_final_message: true,
            add_special_tokens: false,
            return_token_strs: false,
            chat_template: None,
            chat_template_kwargs: None,
            tools: None,
        };

        let error = req
            .into_chat_request("tokenize-test".to_string())
            .expect_err("conflicting flags");
        assert_eq!(
            error.to_error_response().error.message,
            "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
        );
    }
}
