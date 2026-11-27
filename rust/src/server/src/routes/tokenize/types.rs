// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use itertools::Itertools as _;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;
use serde_with::DefaultOnNull;
use vllm_chat::{ChatOptions, ChatRequest, ResolvedToolContext, SamplingParams};
use vllm_text::output::TextDecodeOptions;
use vllm_text::{Prompt, TextRequest};

use crate::error::ApiError;
use crate::routes::openai::chat_completions::convert::{
    convert_message, convert_tools, normalize_generation_prompt_mode,
};
use crate::routes::openai::utils::structured_outputs::ResponseFormat;
use crate::routes::openai::utils::types::{ChatMessage, ReasoningEffort, Tool, default_true};

/// `POST /tokenize` body: a JSON object whose `messages` is a list parses as the
/// chat variant; any other as the completion variant.
#[derive(Debug, Clone)]
pub enum TokenizeRequest {
    Chat(TokenizeChatRequest),
    Completion(TokenizeCompletionRequest),
}

impl<'de> Deserialize<'de> for TokenizeRequest {
    /// Select the variant by `messages` rather than `#[serde(untagged)]`, which
    /// would tokenize `prompt` instead when a body carrying both has a malformed
    /// chat part, and would replace the chat error with a generic one.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        let request = if value.get("messages").is_some_and(Value::is_array) {
            TokenizeChatRequest::deserialize(value).map(Self::Chat)
        } else {
            TokenizeCompletionRequest::deserialize(value).map(Self::Completion)
        };
        request.map_err(de::Error::custom)
    }
}

#[serde_with::serde_as]
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeCompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
    #[serde(default)]
    #[serde_as(as = "DefaultOnNull")]
    pub return_token_strs: bool,
}

impl TokenizeCompletionRequest {
    /// Lower this tokenize body into a [`TextRequest`] for prompt tokenization.
    pub fn into_text_request(self, request_id: String) -> TextRequest {
        TextRequest {
            request_id,
            prompt: Prompt::Text(self.prompt),
            mm_features: None,
            sampling_params: SamplingParams::default(),
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            priority: 0,
            cache_salt: None,
            prompt_truncation: None,
            add_special_tokens: self.add_special_tokens,
            data_parallel_rank: None,
            session_id: None,
            kv_hints: None,
            reasoning_parser_kwargs: Default::default(),
            reasoning_ended: None,
            lora_request: None,
            arrival_time: None,
        }
    }
}

#[serde_with::serde_as]
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default)]
    pub continue_final_message: bool,
    #[serde(default)] // chat default is FALSE (template adds specials)
    pub add_special_tokens: bool,
    #[serde(default)]
    #[serde_as(as = "DefaultOnNull")]
    pub return_token_strs: bool,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    /// Effort level for reasoning models, with the same meaning as in chat
    /// completions.
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Documents for RAG (retrieval-augmented generation).
    #[serde(default)]
    pub documents: Option<Vec<Value>>,
    /// Output format to include in model-specific chat rendering.
    pub response_format: Option<Box<ResponseFormat>>,
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
        let response_format =
            self.response_format.map(serde_json::to_value).transpose().map_err(|error| {
                ApiError::invalid_request(
                    format!("failed to serialize response_format: {error}"),
                    Some("response_format"),
                )
            })?;
        let generation_prompt_mode = normalize_generation_prompt_mode(
            Some(self.add_generation_prompt),
            self.continue_final_message,
            &messages,
        )?;

        let tool_context =
            ResolvedToolContext::new(&messages, convert_tools(self.tools)?, None, true).map_err(
                |error| crate::error::chat_submit_error("failed to resolve request tools", error),
            )?;

        Ok(ChatRequest {
            request_id,
            messages,
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions {
                generation_prompt_mode,
                chat_template: self.chat_template,
                reasoning_effort: self.reasoning_effort.map(|effort| effort.as_str().into()),
                response_format,
                template_kwargs: self.chat_template_kwargs.unwrap_or_default(),
            },
            tool_context,
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            prompt_truncation: None,
            priority: 0,
            documents: self.documents,
            cache_salt: None,
            add_special_tokens: self.add_special_tokens,
            data_parallel_rank: None,
            session_id: None,
            lora_request: None,
            return_prompt_text: false,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use expect_test::expect;
    use serde_json::json;
    use vllm_chat::{ChatRenderer, ChatTool, EffortValue, KimiK3ChatRenderer, KimiK3Variant};
    use vllm_tokenizer::{Tokenizer as _, test_utils::TestTokenizer};

    use super::*;
    use crate::routes::openai::utils::types::{ChatMessage, MessageContent};

    #[test]
    fn tokenize_response_format_reaches_kimi_k3_prompt() {
        let tokenizer = Arc::new(TestTokenizer::new());
        let renderer =
            KimiK3ChatRenderer::new(tokenizer.clone(), Default::default(), KimiK3Variant::K3);
        let prompts: Vec<_> = [
            json!({"type": "json_object"}),
            json!({"type": "json_schema", "json_schema": {
                "name": "answer", "schema": {"type": "boolean"}, "strict": true,
            }}),
            Value::Null,
        ]
        .into_iter()
        .map(|response_format| {
            let request: TokenizeRequest = serde_json::from_value(json!({
                "messages": [{"role": "user", "content": "Answer"}],
                "response_format": response_format,
                "chat_template_kwargs": {"enable_thinking": false},
            }))
            .unwrap();
            let TokenizeRequest::Chat(request) = request else {
                panic!("expected chat tokenization");
            };
            let request = request.into_chat_request("tokenize-test".into()).unwrap();
            let prompt = renderer.render(&request).unwrap().prompt.into_token_ids().unwrap();
            tokenizer.decode(&prompt, false).unwrap()
        })
        .collect();
        expect![[r#"
            [
                "<|open|>message role=\"user\"<|sep|>Answer<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"system\" type=\"response-format\"<|sep|>The system is invoked with `response_format=json_object`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>",
                "<|open|>message role=\"user\"<|sep|>Answer<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"system\" type=\"response-format\"<|sep|>The system is invoked with `response_format=json_schema`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.\nThe JSON data must match the following schema:\n```json\n{\"type\":\"boolean\"}\n```<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>",
                "<|open|>message role=\"user\"<|sep|>Answer<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>",
            ]
        "#]].assert_debug_eq(&prompts);
    }

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
            chat_request.initial_tools(),
            vec![ChatTool {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }),
                strict: None,
                defer_loading: None,
            }]
        );
    }

    /// Python declares `return_token_strs` as `bool | None` on both bodies, so
    /// an explicit `null` means `false` instead of matching neither variant.
    #[test]
    fn null_return_token_strs_means_false() {
        let request: TokenizeRequest = serde_json::from_value(json!({
            "prompt": "hi",
            "return_token_strs": null,
        }))
        .expect("completion body with `return_token_strs: null` is accepted");
        let TokenizeRequest::Completion(req) = request else {
            panic!("prompt body should parse as the completion variant");
        };
        assert!(!req.return_token_strs);

        let request: TokenizeRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "return_token_strs": null,
        }))
        .expect("chat body with `return_token_strs: null` is accepted");
        let TokenizeRequest::Chat(req) = request else {
            panic!("messages body should parse as the chat variant");
        };
        assert!(!req.return_token_strs);
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
            reasoning_effort: None,
            documents: None,
            response_format: None,
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

    /// Chat completions pass `reasoning_effort` and `documents` to the chat
    /// template, so `/tokenize` must too or it renders a different prompt.
    #[test]
    fn tokenize_request_forwards_reasoning_effort_and_documents() {
        let documents = json!([{"title": "Math", "text": "2+2 equals 4."}]);
        let request: TokenizeRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none",
            "documents": documents,
        }))
        .expect("body with reasoning_effort and documents is the chat variant");
        let TokenizeRequest::Chat(req) = request else {
            panic!("messages body should parse as the chat variant");
        };
        let chat_request =
            req.into_chat_request("tokenize-test".to_string()).expect("request is valid");

        assert_eq!(
            chat_request.chat_options.reasoning_effort,
            Some(EffortValue::from("none"))
        );
        assert_eq!(chat_request.documents, documents.as_array().cloned());
    }

    #[test]
    fn tokenize_request_without_reasoning_effort_and_documents_leaves_them_unset() {
        let request: TokenizeRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .expect("plain chat body");
        let TokenizeRequest::Chat(req) = request else {
            panic!("messages body should parse as the chat variant");
        };
        let chat_request =
            req.into_chat_request("tokenize-test".to_string()).expect("request is valid");

        assert_eq!(chat_request.chat_options.reasoning_effort, None);
        assert_eq!(chat_request.documents, None);
    }
}
