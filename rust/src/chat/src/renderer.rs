use std::collections::HashMap;

use llm_tokenizer::chat_template::{
    ChatTemplateContentFormat as LlmChatTemplateContentFormat, ChatTemplateParams,
    ChatTemplateProcessor, detect_chat_template_content_format,
};
use serde_json::json;

use crate::error::{Error, Result};
use crate::request::{ChatRequest, ChatTemplateContentFormat};
use crate::tokenizer::LlmTokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderedPrompt {
    Text {
        prompt: String,
    },
    Tokens {
        prompt_token_ids: Vec<u32>,
        prompt_text: Option<String>,
    },
}

pub trait ChatRenderer: Send + Sync {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt>;
}

#[derive(Debug, Clone)]
pub struct LlmTokenizerChatRenderer {
    tokenizer: LlmTokenizer,
}

impl LlmTokenizerChatRenderer {
    pub fn new(tokenizer: LlmTokenizer) -> Self {
        Self { tokenizer }
    }
}

impl ChatRenderer for LlmTokenizerChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        match request.chat_options.chat_template_content_format {
            ChatTemplateContentFormat::String => {}
            format => return Err(Error::UnsupportedChatTemplateContentFormat(format)),
        }

        let messages = request
            .messages
            .iter()
            .map(|message| json!({ "role": message.role.as_str(), "content": message.content }))
            .collect::<Vec<_>>();
        let template_kwargs = (!request.chat_options.template_kwargs.is_empty()).then(|| {
            request
                .chat_options
                .template_kwargs
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<HashMap<_, _>>()
        });

        let prompt = if let Some(template) = request.chat_options.chat_template.as_deref() {
            let content_format = match detect_chat_template_content_format(template) {
                LlmChatTemplateContentFormat::String => ChatTemplateContentFormat::String,
                LlmChatTemplateContentFormat::OpenAI => ChatTemplateContentFormat::OpenAi,
            };
            if content_format != ChatTemplateContentFormat::String {
                return Err(Error::UnsupportedChatTemplateContentFormat(content_format));
            }

            ChatTemplateProcessor::new(template.to_string())
                .map_err(|error| Error::Tokenizer(error.to_string()))?
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        add_generation_prompt: request.chat_options.add_generation_prompt,
                        tools: None,
                        documents: None,
                        template_kwargs: template_kwargs.as_ref(),
                    },
                )
                .map_err(|error| Error::Tokenizer(error.to_string()))?
        } else {
            let content_format = self.tokenizer.chat_template_content_format();
            if content_format != ChatTemplateContentFormat::String {
                return Err(Error::UnsupportedChatTemplateContentFormat(content_format));
            }

            match self.tokenizer.apply_chat_template(
                &messages,
                request.chat_options.add_generation_prompt,
                template_kwargs.as_ref(),
            ) {
                Ok(prompt) => prompt,
                Err(Error::Tokenizer(message))
                    if message.contains("tokenizer.chat_template is not set") =>
                {
                    return Err(Error::MissingChatTemplate);
                }
                Err(error) => return Err(error),
            }
        };

        Ok(RenderedPrompt::Text { prompt })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_tokenizer::{
        ChatTemplateState, Decoder, Encoder, Encoding, SpecialTokens, TokenizerTrait,
    };

    use super::{ChatRenderer, ChatTemplateParams, LlmTokenizerChatRenderer, RenderedPrompt};
    use crate::request::{ChatMessage, ChatOptions, ChatRequest, ChatRole};
    use crate::tokenizer::LlmTokenizer;

    struct FakeLlmTokenizer {
        chat_template: ChatTemplateState,
        special_tokens: SpecialTokens,
    }

    impl FakeLlmTokenizer {
        fn new(template: &str) -> anyhow::Result<Self> {
            Ok(Self {
                chat_template: ChatTemplateState::new(Some(template.to_string()))?,
                special_tokens: SpecialTokens::default(),
            })
        }
    }

    impl Encoder for FakeLlmTokenizer {
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

    impl Decoder for FakeLlmTokenizer {
        fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
            Ok(String::new())
        }
    }

    impl TokenizerTrait for FakeLlmTokenizer {
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
        ) -> llm_tokenizer::chat_template::ChatTemplateContentFormat {
            self.chat_template.content_format()
        }
    }

    #[test]
    fn llm_tokenizer_renderer_supports_pycompat_templates() {
        let tokenizer = LlmTokenizer::from_arc(Arc::new(
            FakeLlmTokenizer::new(
                "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
            )
            .unwrap(),
        ));
        let renderer = LlmTokenizerChatRenderer::new(tokenizer);
        let request = ChatRequest {
            request_id: "render-1".to_string(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "<think>hello".to_string(),
            }],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
            cache_salt: None,
            trace_headers: None,
            priority: 0,
            data_parallel_rank: None,
        };

        assert_eq!(
            renderer.render(&request).unwrap(),
            RenderedPrompt::Text {
                prompt: "think".to_string(),
            }
        );
    }
}
