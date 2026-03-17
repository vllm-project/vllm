use std::path::Path;

use serde::Serialize;
use serde_json::Value;
use smg_tokenizer::ChatTemplateState;
use smg_tokenizer::chat_template::{
    ChatTemplateContentFormat, ChatTemplateParams, load_chat_template_from_config,
    load_chat_template_from_file,
};
use thiserror_ext::AsReport as _;
use tracing::trace;

use crate::error::Result;
use crate::request::{ChatContent, ChatMessage, ChatRequest};
use crate::{AssistantMessageExt, Error};

/// Chat template handling for Hugging Face models.
///
/// Currently it's a simple wrapper around smg's [`ChatTemplateState`].
pub struct ChatTemplate {
    inner: ChatTemplateState,
}

impl ChatTemplate {
    /// Load a chat template from the given tokenizer config and/or adjacent template file.
    pub fn load(
        tokenizer_config_path: Option<&Path>,
        chat_template_path: Option<&Path>,
    ) -> Result<Self> {
        // Match the usual HF precedence: tokenizer_config first, then any
        // adjacent dedicated chat template file.
        let mut chat_template = tokenizer_config_path
            .and_then(|path| path.to_str())
            .map(load_chat_template_from_config)
            .transpose()
            .map_err(|error| Error::Tokenizer(error.to_report_string()))?
            .flatten();

        if let Some(chat_template_path) = chat_template_path {
            chat_template = load_chat_template_from_file(
                chat_template_path
                    .to_str()
                    .expect("chat template path should be valid UTF-8"),
            )
            .map_err(|error| Error::Tokenizer(error.to_report_string()))?;
        }

        Self::new(chat_template)
    }

    /// Create a chat template from the given template string.
    pub fn new(template: Option<String>) -> Result<Self> {
        Ok(Self {
            inner: ChatTemplateState::new(template)
                .map_err(|error| Error::Tokenizer(error.to_report_string()))?,
        })
    }

    /// Apply the chat template to one chat request, rendering the prompt string to be tokenized and
    /// submitted to the model.
    pub fn apply_chat_template(&self, request: &ChatRequest) -> Result<String> {
        let messages = template_messages_to_json(&request.messages, self.inner.content_format())?;
        trace!(
            request_id = %request.request_id,
            message_count = messages.len(),
            content_format = ?self.inner.content_format(),
            ?messages,
            "applying chat template"
        );

        let merged_template_kwargs = {
            let mut kwargs = request.chat_options.template_kwargs.clone();
            kwargs.insert(
                "continue_final_message".to_string(),
                Value::Bool(request.chat_options.continue_final_message),
            );
            kwargs
        };

        let prompt = self
            .inner
            .apply(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: request.chat_options.add_generation_prompt,
                    tools: None,
                    documents: None,
                    template_kwargs: Some(&merged_template_kwargs),
                },
            )
            .map_err(|error| {
                let message = error.to_report_string();
                if message.contains("tokenizer.chat_template is not set") {
                    Error::MissingChatTemplate
                } else {
                    Error::Tokenizer(message)
                }
            })?;

        trace!(
            request_id = %request.request_id,
            prompt_len = prompt.len(),
            prompt,
            "rendered chat template prompt"
        );

        Ok(prompt)
    }
}

/// Chat message in the JSON shape expected by Jinja chat templates.
#[derive(Serialize)]
struct AssistantTemplateMessage {
    role: &'static str,
    content: Value,
    // Reasoning-capable HF templates are inconsistent on the exact field name,
    // so expose both variants for compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    // tool_calls: Option<Vec<...>>,
}

/// Convert chat messages into the JSON shape expected by Jinja chat templates.
fn template_messages_to_json(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<Value>> {
    messages
        .iter()
        .map(|message| template_message_to_json(message, content_format))
        .collect()
}

fn template_message_to_json(
    message: &ChatMessage,
    content_format: ChatTemplateContentFormat,
) -> Result<Value> {
    let msg = match message {
        ChatMessage::System { content } => AssistantTemplateMessage {
            role: "system",
            content: template_content_to_json(content, content_format)?,
            reasoning: None,
            reasoning_content: None,
        },
        ChatMessage::User { content } => AssistantTemplateMessage {
            role: "user",
            content: template_content_to_json(content, content_format)?,
            reasoning: None,
            reasoning_content: None,
        },
        ChatMessage::Assistant { content } => {
            let text = content.text();
            let reasoning = content.reasoning();
            let content = template_content_to_json(&ChatContent::Text(text), content_format)?;
            AssistantTemplateMessage {
                role: "assistant",
                content,
                reasoning: reasoning.clone(),
                reasoning_content: reasoning,
            }
        }
    };

    Ok(serde_json::to_value(msg).expect("chat message should serialize to valid JSON"))
}

fn template_content_to_json(
    content: &ChatContent,
    content_format: ChatTemplateContentFormat,
) -> Result<Value> {
    Ok(match content_format {
        ChatTemplateContentFormat::String => Value::String(content.try_flatten_to_text()?),
        ChatTemplateContentFormat::OpenAI => serde_json::to_value(content)
            .expect("text-only chat content should serialize to valid JSON"),
    })
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::ChatTemplate;
    use crate::request::{
        ChatContentPart, ChatMessage, ChatOptions, ChatRequest, ChatRole, UserSamplingParams,
    };
    use crate::{AssistantContentBlock, Error, Result};

    const QWEN3_0_6B_TEMPLATE: &str = include_str!("../tests/templates/qwen3.jinja");

    fn sample_request(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest {
            request_id: "render-test".to_string(),
            messages,
            sampling_params: UserSamplingParams::default(),
            chat_options: ChatOptions::default(),
        }
    }

    fn render(template: Option<&str>, request: &ChatRequest) -> Result<String> {
        ChatTemplate::new(template.map(str::to_owned))?.apply_chat_template(request)
    }

    #[test]
    fn chat_template_supports_pycompat_templates() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "<think>hello")]);

        let rendered = render(
            Some(
                "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "think");
    }

    #[test]
    fn chat_template_passes_continue_final_message_to_template() {
        let mut request = sample_request(vec![ChatMessage::text(
            ChatRole::Assistant,
            "The capital of",
        )]);

        assert_eq!(
            render(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
            )
            .unwrap(),
            "new"
        );

        request.chat_options.continue_final_message = true;
        request.chat_options.add_generation_prompt = false;

        assert_eq!(
            render(
                Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
                &request,
            )
            .unwrap(),
            "continue"
        );
    }

    #[test]
    fn chat_template_flattens_text_parts_for_string_templates() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text(" world"),
        ])]);

        let rendered = render(Some("{{ messages[0].content }}"), &request).unwrap();

        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn chat_template_keeps_string_text_for_openai_detected_templates() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- if message.content is string %}{%- set content = message.content %}{{ content }}{%- endif %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello");
    }

    #[test]
    fn chat_template_emits_openai_text_blocks_for_structured_templates() {
        let request = sample_request(vec![ChatMessage::user(vec![
            ChatContentPart::text("hello"),
            ChatContentPart::text("world"),
        ])]);

        let rendered = render(
            Some(
                "{%- for message in messages %}{%- for item in message.content %}{{ item.text }}|{%- endfor %}{%- endfor %}",
            ),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "hello|world|");
    }

    #[test]
    fn chat_template_requires_a_template() {
        let request = sample_request(vec![ChatMessage::text(ChatRole::User, "hello")]);
        let error = render(None, &request).unwrap_err();

        assert!(matches!(error, Error::MissingChatTemplate));
    }

    #[test]
    fn chat_template_exposes_assistant_reasoning_separately() {
        let request = sample_request(vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: "inner".to_string(),
            },
            AssistantContentBlock::Text {
                text: "outer".to_string(),
            },
        ])]);

        let rendered = render(
            Some("{{ messages[0].reasoning_content }}|{{ messages[0].content }}"),
            &request,
        )
        .unwrap();

        assert_eq!(rendered, "inner|outer");
    }

    #[test]
    fn qwen3_template_omits_reasoning_for_historical_assistant_messages() {
        let request = sample_request(vec![
            ChatMessage::text(
                ChatRole::User,
                "Hi. Tell me about the capital of France in short",
            ),
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "\nOkay, the user is asking... I think that's all.\n".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "Paris is the capital of France.".to_string(),
                },
            ]),
            ChatMessage::text(ChatRole::User, "Tell me about Paris more."),
        ]);

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            Hi. Tell me about the capital of France in short<|im_end|>
            <|im_start|>assistant
            Paris is the capital of France.<|im_end|>
            <|im_start|>user
            Tell me about Paris more.<|im_end|>
            <|im_start|>assistant
        "#]]
        .assert_eq(&rendered);
    }

    #[test]
    fn qwen3_template_keeps_reasoning_after_the_last_user_query() {
        let mut request = sample_request(vec![
            ChatMessage::text(ChatRole::User, "What is 1 + 1?"),
            ChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Reasoning {
                    text: "need simple arithmetic".to_string(),
                },
                AssistantContentBlock::Text {
                    text: "2".to_string(),
                },
            ]),
        ]);
        request.chat_options.add_generation_prompt = false;

        let rendered = render(Some(QWEN3_0_6B_TEMPLATE), &request).unwrap();

        expect![[r#"
            <|im_start|>user
            What is 1 + 1?<|im_end|>
            <|im_start|>assistant
            <think>
            need simple arithmetic
            </think>

            2<|im_end|>
        "#]]
        .assert_eq(&rendered);
    }
}
