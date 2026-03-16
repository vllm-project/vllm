use serde_json::json;

use super::SmgTokenizer;
use crate::error::{Error, Result};
use crate::renderer::{ChatRenderer, RenderedPrompt};
use crate::request::ChatRequest;

/// [`ChatRenderer`] implementation backed by [`SmgTokenizer`]'s chat-template support.
///
/// This currently supports only string-style chat templates, matching `ChatMessage { content:
/// String }`.
#[derive(Debug, Clone)]
pub struct SmgTokenizerChatRenderer {
    tokenizer: SmgTokenizer,
}

impl SmgTokenizerChatRenderer {
    /// Create a renderer from an SMG-backed tokenizer.
    pub fn new(tokenizer: SmgTokenizer) -> Self {
        Self { tokenizer }
    }
}

impl ChatRenderer for SmgTokenizerChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        let messages = request
            .messages
            .iter()
            .map(|message| json!({ "role": message.role.as_str(), "content": message.content }))
            .collect::<Vec<_>>();
        let template_kwargs = (!request.chat_options.template_kwargs.is_empty())
            .then_some(&request.chat_options.template_kwargs);

        if !self.tokenizer.supports_string_chat_template() {
            return Err(Error::UnsupportedChatTemplateFormat);
        }

        let prompt = match self.tokenizer.apply_chat_template(
            &messages,
            request.chat_options.add_generation_prompt,
            request.chat_options.continue_final_message,
            template_kwargs,
        ) {
            Ok(prompt) => prompt,
            Err(Error::Tokenizer(message))
                if message.contains("tokenizer.chat_template is not set") =>
            {
                return Err(Error::MissingChatTemplate);
            }
            Err(error) => return Err(error),
        };

        Ok(RenderedPrompt::Text { prompt })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_tokenizer::chat_template::ChatTemplateParams;
    use smg_tokenizer::{
        ChatTemplateState, Decoder, Encoder, Encoding, SpecialTokens, TokenizerTrait,
    };

    use super::SmgTokenizerChatRenderer;
    use crate::renderer::{ChatRenderer, RenderedPrompt};
    use crate::request::{ChatMessage, ChatOptions, ChatRequest, ChatRole};
    use crate::smg::SmgTokenizer;

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
    fn smg_tokenizer_renderer_supports_pycompat_templates() {
        let tokenizer = SmgTokenizer::from_arc(Arc::new(
            FakeSmgTokenizer::new(
                "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
            )
            .unwrap(),
        ));
        let renderer = SmgTokenizerChatRenderer::new(tokenizer);
        let request = ChatRequest {
            request_id: "render-1".to_string(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "<think>hello".to_string(),
            }],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
        };

        assert_eq!(
            renderer.render(&request).unwrap(),
            RenderedPrompt::Text {
                prompt: "think".to_string(),
            }
        );
    }

    #[test]
    fn smg_tokenizer_renderer_passes_continue_final_message_to_template() {
        let tokenizer = SmgTokenizer::from_arc(Arc::new(
            FakeSmgTokenizer::new(
                "{% if continue_final_message %}continue{% else %}new{% endif %}",
            )
            .unwrap(),
        ));
        let renderer = SmgTokenizerChatRenderer::new(tokenizer);
        let mut request = ChatRequest {
            request_id: "render-2".to_string(),
            messages: vec![ChatMessage {
                role: ChatRole::Assistant,
                content: "The capital of".to_string(),
            }],
            sampling_params: Default::default(),
            chat_options: ChatOptions::default(),
        };

        assert_eq!(
            renderer.render(&request).unwrap(),
            RenderedPrompt::Text {
                prompt: "new".to_string(),
            }
        );

        request.chat_options.continue_final_message = true;
        request.chat_options.add_generation_prompt = false;

        assert_eq!(
            renderer.render(&request).unwrap(),
            RenderedPrompt::Text {
                prompt: "continue".to_string(),
            }
        );
    }
}
