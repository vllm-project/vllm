use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde_json::json;
use smg_tokenizer::TokenizerTrait as SmgTokenizerTrait;
use smg_tokenizer::chat_template::{ChatTemplateContentFormat, ChatTemplateParams};
use smg_tokenizer::factory::create_tokenizer_async;
use thiserror_ext::AsReport as _;

use crate::backend::{ChatBackend, SamplingHints};
use crate::error::{Error, Result};
use crate::request::{ChatContent, ChatMessage, ChatRequest};

/// [`ChatBackend`] implementation backed by the crates.io `llm-tokenizer` package, imported here
/// as `smg_tokenizer`.
#[derive(Clone)]
pub struct SmgChatBackend {
    inner: Arc<dyn SmgTokenizerTrait>,
}

impl fmt::Debug for SmgChatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmgChatBackend").finish_non_exhaustive()
    }
}

impl SmgChatBackend {
    /// Load a tokenizer and any adjacent/default chat template for one model or local path.
    pub async fn from_model_or_path(model_name_or_path: &str) -> Result<Self> {
        let inner = create_tokenizer_async(model_name_or_path)
            .await
            .map_err(|error| Error::Tokenizer(error.to_report_string()))?;

        Ok(Self { inner })
    }

    /// Wrap an existing smg-tokenizer trait object.
    pub fn from_inner(inner: Arc<dyn SmgTokenizerTrait>) -> Self {
        Self { inner }
    }

    fn apply_chat_template_inner(
        &self,
        request: &ChatRequest,
        template_kwargs: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<String> {
        let messages = request
            .messages
            .iter()
            .map(|message| self.template_message_to_json(message))
            .collect::<Result<Vec<_>>>()?;
        let mut merged_template_kwargs = template_kwargs.cloned().unwrap_or_default();
        merged_template_kwargs.insert(
            "continue_final_message".to_string(),
            serde_json::Value::Bool(request.chat_options.continue_final_message),
        );

        self.inner
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: request.chat_options.add_generation_prompt,
                    tools: None,
                    documents: None,
                    template_kwargs: Some(&merged_template_kwargs),
                },
            )
            .map_err(|error| Error::Tokenizer(error.to_report_string()))
    }

    fn template_message_to_json(&self, message: &ChatMessage) -> Result<serde_json::Value> {
        Ok(json!({
            "role": message.role.as_str(),
            "content": self.template_content_to_json(&message.content)?,
        }))
    }

    fn template_content_to_json(&self, content: &ChatContent) -> Result<serde_json::Value> {
        Ok(match self.inner.chat_template_content_format() {
            ChatTemplateContentFormat::String => {
                serde_json::Value::String(content.try_flatten_to_text()?)
            }
            ChatTemplateContentFormat::OpenAI => serde_json::to_value(content)
                .expect("text-only chat content should serialize to valid JSON"),
        })
    }
}

impl ChatBackend for SmgChatBackend {
    fn apply_chat_template(&self, request: &ChatRequest) -> Result<String> {
        match self.apply_chat_template_inner(request, Some(&request.chat_options.template_kwargs)) {
            Ok(prompt) => Ok(prompt),
            Err(Error::Tokenizer(message))
                if message.contains("tokenizer.chat_template is not set") =>
            {
                Err(Error::MissingChatTemplate)
            }
            Err(error) => Err(error),
        }
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|error| Error::Tokenizer(error.to_report_string()))?;
        Ok(encoding.token_ids().to_vec())
    }

    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| Error::Tokenizer(error.to_report_string()))
    }

    fn sampling_hints(&self) -> Result<SamplingHints> {
        let primary_eos_token_id = self
            .inner
            .get_special_tokens()
            .eos_token
            .as_deref()
            .and_then(|token| self.inner.token_to_id(token));

        Ok(SamplingHints {
            primary_eos_token_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_tokenizer::chat_template::{ChatTemplateContentFormat, ChatTemplateParams};
    use smg_tokenizer::{
        ChatTemplateState, Decoder, Encoder, Encoding, SpecialTokens, TokenizerTrait,
    };

    use super::SmgChatBackend;
    use crate::backend::ChatBackend;
    use crate::request::{ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole};

    struct FakeSmgTokenizer {
        chat_template: ChatTemplateState,
        special_tokens: SpecialTokens,
    }

    impl FakeSmgTokenizer {
        fn new(template: &str) -> anyhow::Result<Self> {
            Ok(Self {
                chat_template: ChatTemplateState::new(Some(template.to_string()))?,
                special_tokens: SpecialTokens::default(),
            })
        }
    }

    impl Encoder for FakeSmgTokenizer {
        fn encode(&self, _input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
            Ok(Encoding::Plain(Vec::new()))
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            _add_special_tokens: bool,
        ) -> anyhow::Result<Vec<Encoding>> {
            Ok(inputs.iter().map(|_| Encoding::Plain(Vec::new())).collect())
        }
    }

    impl Decoder for FakeSmgTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    impl TokenizerTrait for FakeSmgTokenizer {
        fn vocab_size(&self) -> usize {
            0
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            &self.special_tokens
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn apply_chat_template(
            &self,
            messages: &[serde_json::Value],
            params: ChatTemplateParams,
        ) -> anyhow::Result<String> {
            self.chat_template.apply(messages, params)
        }

        fn chat_template_content_format(
            &self,
        ) -> smg_tokenizer::chat_template::ChatTemplateContentFormat {
            self.chat_template.content_format()
        }
    }

    #[test]
    fn smg_chat_backend_supports_pycompat_templates() {
        let backend = SmgChatBackend::from_inner(Arc::new(
            FakeSmgTokenizer::new(
                "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
            )
            .unwrap(),
        ));
        let request = ChatRequest {
            request_id: "render-1".to_string(),
            messages: vec![ChatMessage::text(ChatRole::User, "<think>hello")],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
        };

        assert_eq!(backend.apply_chat_template(&request).unwrap(), "think");
    }

    #[test]
    fn smg_chat_backend_passes_continue_final_message_to_template() {
        let backend = SmgChatBackend::from_inner(Arc::new(
            FakeSmgTokenizer::new(
                "{% if continue_final_message %}continue{% else %}new{% endif %}",
            )
            .unwrap(),
        ));
        let mut request = ChatRequest {
            request_id: "render-2".to_string(),
            messages: vec![ChatMessage::text(ChatRole::Assistant, "The capital of")],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
        };

        assert_eq!(backend.apply_chat_template(&request).unwrap(), "new");

        request.chat_options.continue_final_message = true;
        request.chat_options.add_generation_prompt = false;

        assert_eq!(backend.apply_chat_template(&request).unwrap(), "continue");
    }

    #[test]
    fn smg_chat_backend_flattens_text_parts_for_string_templates() {
        let backend = SmgChatBackend::from_inner(Arc::new(
            FakeSmgTokenizer::new("{{ messages[0].content }}").unwrap(),
        ));
        let request = ChatRequest {
            request_id: "render-3".to_string(),
            messages: vec![ChatMessage::new(
                ChatRole::User,
                vec![
                    ChatContentPart::text("hello"),
                    ChatContentPart::text(" world"),
                ],
            )],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
        };

        assert_eq!(
            backend.apply_chat_template(&request).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn smg_chat_backend_keeps_string_text_for_openai_detected_templates() {
        let tokenizer = FakeSmgTokenizer::new(
            "{%- for message in messages %}{%- if message.content is string %}{%- set content = message.content %}{{ content }}{%- endif %}{%- endfor %}",
        )
        .unwrap();
        assert_eq!(
            tokenizer.chat_template.content_format(),
            ChatTemplateContentFormat::OpenAI
        );
        let backend = SmgChatBackend::from_inner(Arc::new(tokenizer));
        let request = ChatRequest {
            request_id: "render-4".to_string(),
            messages: vec![ChatMessage::text(ChatRole::User, "hello")],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
        };

        assert_eq!(backend.apply_chat_template(&request).unwrap(), "hello");
    }

    #[test]
    fn smg_chat_backend_emits_openai_text_blocks_for_structured_templates() {
        let tokenizer = FakeSmgTokenizer::new(
            "{%- for message in messages %}{%- for item in message.content %}{{ item.text }}|{%- endfor %}{%- endfor %}",
        )
        .unwrap();
        assert_eq!(
            tokenizer.chat_template.content_format(),
            ChatTemplateContentFormat::OpenAI
        );
        let backend = SmgChatBackend::from_inner(Arc::new(tokenizer));
        let request = ChatRequest {
            request_id: "render-5".to_string(),
            messages: vec![ChatMessage::new(
                ChatRole::User,
                vec![
                    ChatContentPart::text("hello"),
                    ChatContentPart::text("world"),
                ],
            )],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
        };

        assert_eq!(
            backend.apply_chat_template(&request).unwrap(),
            "hello|world|"
        );
    }
}
