// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde_json::{Map, Value, json};
use thiserror_ext::AsReport as _;
use vllm_text::Prompt;
use vllm_text::tokenizer::{DynTokenizer, Tokenizer};

use super::{ChatRenderer, RenderedPrompt, request_template_kwargs};
use crate::error::{Error, Result};
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatTool};
use crate::{AssistantContentBlock, AssistantToolCall};

const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_IMAGE: &str = "<|content_image|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const CONTENT_AUDIO_INPUT: &str = "<|content_audio_input|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
// Inkling renderer semantics use this slot for structured payloads such as tool
// declarations.
const CONTENT_XML: &str = "<|content_xml|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const END_MESSAGE: &str = "<|end_message|>";
const AUDIO_END: &str = "<|audio_end|>";
const MAX_REASONING_EFFORT: f64 = 0.99;

/// Native Inkling renderer that emits token IDs directly.
#[derive(Clone)]
pub struct InklingChatRenderer {
    tokenizer: DynTokenizer,
    special: InklingSpecialTokenIds,
}

#[derive(Debug, Clone, Copy)]
struct InklingSpecialTokenIds {
    message_user: u32,
    message_model: u32,
    message_system: u32,
    message_tool: u32,
    content_text: u32,
    content_image: u32,
    content_model_end_sampling: u32,
    content_audio_input: u32,
    content_thinking: u32,
    content_xml: u32,
    content_invoke_tool_json: u32,
    end_message: u32,
    audio_end: u32,
}

impl InklingSpecialTokenIds {
    fn resolve(tokenizer: &dyn Tokenizer) -> Result<Self> {
        Ok(Self {
            message_user: resolve_special_token(tokenizer, MESSAGE_USER)?,
            message_model: resolve_special_token(tokenizer, MESSAGE_MODEL)?,
            message_system: resolve_special_token(tokenizer, MESSAGE_SYSTEM)?,
            message_tool: resolve_special_token(tokenizer, MESSAGE_TOOL)?,
            content_text: resolve_special_token(tokenizer, CONTENT_TEXT)?,
            content_image: resolve_special_token(tokenizer, CONTENT_IMAGE)?,
            content_model_end_sampling: resolve_special_token(
                tokenizer,
                CONTENT_MODEL_END_SAMPLING,
            )?,
            content_audio_input: resolve_special_token(tokenizer, CONTENT_AUDIO_INPUT)?,
            content_thinking: resolve_special_token(tokenizer, CONTENT_THINKING)?,
            content_xml: resolve_special_token(tokenizer, CONTENT_XML)?,
            content_invoke_tool_json: resolve_special_token(tokenizer, CONTENT_INVOKE_TOOL_JSON)?,
            end_message: resolve_special_token(tokenizer, END_MESSAGE)?,
            audio_end: resolve_special_token(tokenizer, AUDIO_END)?,
        })
    }
}

impl InklingChatRenderer {
    pub fn new(tokenizer: DynTokenizer) -> Result<Self> {
        let special = InklingSpecialTokenIds::resolve(tokenizer.as_ref())?;
        Ok(Self { tokenizer, special })
    }

    fn write_text_tokens(&self, out: &mut Vec<u32>, text: &str) -> Result<()> {
        out.extend(self.tokenizer.encode(text, false)?);
        Ok(())
    }

    fn write_message_start(
        &self,
        out: &mut Vec<u32>,
        role_token_id: u32,
        author_name: Option<&str>,
    ) -> Result<()> {
        out.push(role_token_id);
        if let Some(author_name) = author_name
            && !author_name.is_empty()
        {
            self.write_text_tokens(out, author_name)?;
        }
        Ok(())
    }

    fn write_text_block(
        &self,
        out: &mut Vec<u32>,
        role_token_id: u32,
        author_name: Option<&str>,
        text: &str,
    ) -> Result<()> {
        self.write_message_start(out, role_token_id, author_name)?;
        out.push(self.special.content_text);
        self.write_text_tokens(out, text)?;
        out.push(self.special.end_message);
        Ok(())
    }

    /// Write one image block holding only the `<|content_image|>` marker.
    /// Multimodal preprocessing later expands the marker into per-patch
    /// image placeholder tokens once the patch count is known, mirroring the
    /// Python `InklingMultiModalProcessor` marker-anchored prompt updates.
    fn write_image_block(&self, out: &mut Vec<u32>, role_token_id: u32) {
        out.push(role_token_id);
        out.push(self.special.content_image);
        out.push(self.special.end_message);
    }

    /// Write one audio block holding the `<|content_audio_input|>` marker
    /// followed by the `<|audio_end|>` terminator. Multimodal preprocessing
    /// later expands the marker into per-frame audio placeholder tokens
    /// (landing before `<|audio_end|>`) once the clip length is known.
    fn write_audio_block(&self, out: &mut Vec<u32>, role_token_id: u32) {
        out.push(role_token_id);
        out.push(self.special.content_audio_input);
        out.push(self.special.audio_end);
        out.push(self.special.end_message);
    }

    fn write_reasoning_block(&self, out: &mut Vec<u32>, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        out.push(self.special.message_model);
        out.push(self.special.content_thinking);
        self.write_text_tokens(out, text)?;
        out.push(self.special.end_message);
        Ok(())
    }

    fn write_reasoning_effort(&self, out: &mut Vec<u32>, effort: f64) -> Result<()> {
        if !(0.0..=MAX_REASONING_EFFORT).contains(&effort) {
            return Err(Error::ChatTemplate(format!(
                "Inkling reasoning_effort must be in [0.0, 0.99], got {effort}"
            )));
        }
        let formatted = format!("{effort:.2}");
        let effort = formatted.trim_end_matches('0').trim_end_matches('.');
        let effort = if matches!(effort, "0" | "-0") {
            "0.0"
        } else {
            effort
        };
        self.write_text_block(
            out,
            self.special.message_system,
            None,
            &format!("Thinking effort level: {effort}"),
        )
    }

    fn write_tool_declarations(&self, out: &mut Vec<u32>, tools: &[&ChatTool]) -> Result<()> {
        if tools.is_empty() {
            return Ok(());
        }

        let mut specs = Vec::with_capacity(tools.len());
        for tool in tools {
            specs.push(json!({
                "description": tool.description.as_deref().unwrap_or(""),
                "name": tool.name,
                "parameters": sort_json(&tool.parameters),
                "type": "function",
            }));
        }
        let payload = compact_json(&sort_json(&Value::Array(specs)))?;

        self.write_message_start(out, self.special.message_system, Some("tool_declare"))?;
        out.push(self.special.content_xml);
        self.write_text_tokens(out, &payload)?;
        out.push(self.special.end_message);
        Ok(())
    }

    fn write_chat_content(
        &self,
        out: &mut Vec<u32>,
        role_token_id: u32,
        content: &ChatContent,
    ) -> Result<()> {
        match content {
            ChatContent::Text(text) => {
                if !text.is_empty() {
                    self.write_text_block(out, role_token_id, None, text)?;
                }
            }
            ChatContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ChatContentPart::Text { text } => {
                            self.write_text_block(out, role_token_id, None, text)?;
                        }
                        ChatContentPart::ImageUrl { .. } => {
                            self.write_image_block(out, role_token_id);
                        }
                        ChatContentPart::InputAudio { .. } | ChatContentPart::AudioUrl { .. } => {
                            self.write_audio_block(out, role_token_id);
                        }
                        // Inkling has no video modality.
                        ChatContentPart::VideoUrl { .. } => {
                            return Err(Error::UnsupportedMultimodalContent("video_url"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn write_assistant_tool_call(
        &self,
        out: &mut Vec<u32>,
        tool_call: &AssistantToolCall,
    ) -> Result<()> {
        let payload = tool_call_json(tool_call)?;

        self.write_message_start(out, self.special.message_model, Some(&tool_call.name))?;
        out.push(self.special.content_invoke_tool_json);
        self.write_text_tokens(out, &payload)?;
        out.push(self.special.end_message);
        Ok(())
    }

    fn write_assistant_content(
        &self,
        out: &mut Vec<u32>,
        content: &[AssistantContentBlock],
        tool_call_id_to_name: &mut HashMap<String, String>,
    ) -> Result<()> {
        for block in content {
            match block {
                AssistantContentBlock::Reasoning { text } => {
                    self.write_reasoning_block(out, text)?;
                }
                AssistantContentBlock::Text { text } => {
                    if !text.is_empty() {
                        self.write_text_block(out, self.special.message_model, None, text)?;
                    }
                }
                AssistantContentBlock::ToolCall(tool_call) => {
                    if !tool_call.id.is_empty() {
                        tool_call_id_to_name.insert(tool_call.id.clone(), tool_call.name.clone());
                    }
                    self.write_assistant_tool_call(out, tool_call)?;
                }
            }
        }
        out.push(self.special.content_model_end_sampling);
        Ok(())
    }

    fn write_tool_response(
        &self,
        out: &mut Vec<u32>,
        content: &ChatContent,
        tool_call_id: &str,
        tool_call_id_to_name: &HashMap<String, String>,
    ) -> Result<()> {
        let text = content.try_flatten_to_text()?;
        let tool_name = tool_call_id_to_name.get(tool_call_id).map(String::as_str).unwrap_or("");
        self.write_text_block(out, self.special.message_tool, Some(tool_name), &text)
    }
}

impl ChatRenderer for InklingChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        if request.chat_options.continue_final_message() {
            return Err(Error::ChatTemplate(
                "Inkling renderer does not support continue_final_message".to_string(),
            ));
        }

        let mut out = Vec::new();
        let mut tool_call_id_to_name = HashMap::new();
        let effective_template_kwargs = request_template_kwargs(request);
        let tools = rendered_tools(request);
        self.write_tool_declarations(&mut out, &tools)?;
        let mut reasoning_effort =
            resolve_reasoning_effort(effective_template_kwargs.get("reasoning_effort"));

        for message in &request.messages {
            if !matches!(
                message,
                ChatMessage::System { .. } | ChatMessage::Developer { .. }
            ) && let Some(effort) = reasoning_effort.take()
            {
                self.write_reasoning_effort(&mut out, effort)?;
            }

            match message {
                ChatMessage::System { content } => {
                    self.write_chat_content(&mut out, self.special.message_system, content)?;
                }
                ChatMessage::Developer { content, .. } => {
                    self.write_chat_content(&mut out, self.special.message_system, content)?;
                }
                ChatMessage::User { content } => {
                    self.write_chat_content(&mut out, self.special.message_user, content)?;
                }
                ChatMessage::Assistant { content } => {
                    self.write_assistant_content(&mut out, content, &mut tool_call_id_to_name)?;
                }
                ChatMessage::ToolResponse {
                    content,
                    tool_call_id,
                } => {
                    self.write_tool_response(
                        &mut out,
                        content,
                        tool_call_id,
                        &tool_call_id_to_name,
                    )?;
                }
            }
        }

        if let Some(effort) = reasoning_effort {
            self.write_reasoning_effort(&mut out, effort)?;
        }

        if request.chat_options.add_generation_prompt() {
            out.push(self.special.message_model);
        }

        Ok(RenderedPrompt {
            prompt: Prompt::TokenIds(out),
            effective_template_kwargs,
        })
    }
}

fn resolve_reasoning_effort(value: Option<&Value>) -> Option<f64> {
    let Some(value) = value else {
        return Some(0.9);
    };
    match value {
        Value::String(name) => match name.as_str() {
            "none" => Some(0.0),
            "minimal" => Some(0.1),
            "low" => Some(0.2),
            "medium" => Some(0.7),
            "high" => Some(0.9),
            "xhigh" | "max" => Some(0.99),
            _ => None,
        },
        Value::Number(number) => number.as_f64(),
        _ => None,
    }
}

fn resolve_special_token(tokenizer: &dyn Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        Error::ChatTemplate(format!(
            "Inkling tokenizer is missing special token `{token}`"
        ))
    })
}

fn rendered_tools(request: &ChatRequest) -> Vec<&ChatTool> {
    if !request.tool_parsing_enabled() {
        return Vec::new();
    }

    let mut tools = Vec::with_capacity(request.tools.len());
    tools.extend(request.tools.iter());
    for message in &request.messages {
        if let ChatMessage::Developer {
            tools: Some(local_tools),
            ..
        } = message
        {
            tools.extend(local_tools.iter());
        }
    }
    tools
}

fn tool_call_json(tool_call: &AssistantToolCall) -> Result<String> {
    let name_json = serde_json::to_string(&tool_call.name)
        .map_err(|error| Error::ChatTemplate(error.as_report().to_string()))?;
    let arguments = if tool_call.arguments.trim().is_empty() {
        Value::Object(Map::new())
    } else {
        serde_json::from_str(&tool_call.arguments).map_err(|error| {
            Error::ChatTemplate(format!(
                "Inkling tool call arguments must decode to a JSON object: {error}"
            ))
        })?
    };
    let Value::Object(_) = arguments else {
        return Err(Error::ChatTemplate(
            "Inkling tool call arguments must decode to a JSON object".to_string(),
        ));
    };
    let args_json = compact_json(&sort_json(&arguments))?;
    Ok(format!("{{\"name\":{name_json},\"args\":{args_json}}}"))
}

fn compact_json(value: &Value) -> Result<String> {
    serde_json::to_string(value).map_err(|error| Error::ChatTemplate(error.as_report().to_string()))
}

fn sort_json(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(sort_json).collect()),
        Value::Object(map) => {
            let mut sorted = Map::new();
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), sort_json(&map[key]));
            }
            Value::Object(sorted)
        }
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests;
