use std::collections::HashMap;

use serde_json::{Value, json};
use smg_tokenizer::chat_template::ChatTemplateContentFormat;

use crate::error::Result;
use crate::request::{ChatContent, ChatMessage, ChatRequest};

/// Convert chat messages into the JSON shape expected by Jinja chat templates.
pub(crate) fn template_messages_to_json(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<Value>> {
    messages
        .iter()
        .map(|message| template_message_to_json(message, content_format))
        .collect()
}

/// Merge request-scoped chat template kwargs with chat-layer defaults.
pub(crate) fn merged_template_kwargs(
    request: &ChatRequest,
    template_kwargs: Option<&HashMap<String, Value>>,
) -> HashMap<String, Value> {
    let mut merged_template_kwargs = template_kwargs.cloned().unwrap_or_default();
    merged_template_kwargs.insert(
        "continue_final_message".to_string(),
        Value::Bool(request.chat_options.continue_final_message),
    );
    merged_template_kwargs
}

fn template_message_to_json(
    message: &ChatMessage,
    content_format: ChatTemplateContentFormat,
) -> Result<Value> {
    Ok(json!({
        "role": message.role.as_str(),
        "content": template_content_to_json(&message.content, content_format)?,
    }))
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
