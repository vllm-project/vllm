// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Inkling2 rendering.
//!
//! Inkling2 keeps Inkling's block format, but its reference renderer differs:
//! no generation prefix (the model emits its own `<|message_model|>`), no tool
//! name in tool-call headers, no tool-result names (the request carries none),
//! empty string contents kept as empty blocks, payload text encoded without
//! special-token matching, effort quantized to integer milli-effort, and its
//! own canonical JSON spelling for floats.

use std::collections::HashMap;

use serde_json::{Map, Value};
use vllm_text::Prompt;

use super::{
    InklingChatRenderer, MAX_REASONING_EFFORT, compact_json, resolve_reasoning_effort,
    tool_call_json_value,
};
use crate::error::{Error, Result};
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatRequest};
use crate::{AssistantContentBlock, EffortValue, RenderedPrompt};

impl InklingChatRenderer {
    pub(super) fn render_inkling2(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        let effort = self.inkling2_reasoning_effort(request)?;

        let mut messages = request.messages.as_slice();
        let mut assistant_prefix = None;
        if request.chat_options.continue_final_message()
            && let Some((ChatMessage::Assistant { content }, rest)) = messages.split_last()
            && let [AssistantContentBlock::Text { text }] = content.as_slice()
        {
            assistant_prefix = Some(text.as_str());
            messages = rest;
        }

        let mut out = Vec::new();
        self.write_inkling2_tool_declarations(&mut out, request)?;
        let mut effort = Some(effort);
        for message in messages {
            if !matches!(
                message,
                ChatMessage::System { .. } | ChatMessage::Developer { .. }
            ) && let Some(effort) = effort.take()
            {
                self.write_inkling2_effort(&mut out, effort)?;
            }

            match message {
                ChatMessage::System { content } | ChatMessage::Developer { content, .. } => {
                    self.write_inkling2_content(&mut out, self.special.message_system, content)?;
                }
                ChatMessage::User { content } => {
                    self.write_inkling2_content(&mut out, self.special.message_user, content)?;
                }
                ChatMessage::Assistant { content } => {
                    self.write_inkling2_assistant(&mut out, content)?;
                }
                ChatMessage::ToolResponse { content, .. } => {
                    self.write_inkling2_content(&mut out, self.special.message_tool, content)?;
                }
                ChatMessage::Custom { role, .. } => {
                    return Err(Error::UnsupportedChatRole { role: role.clone() });
                }
            }
        }
        if let Some(effort) = effort {
            self.write_inkling2_effort(&mut out, effort)?;
        }

        if let Some(prefix) = assistant_prefix {
            out.push(self.special.message_model);
            out.push(self.special.content_text);
            self.write_text_tokens(&mut out, prefix)?;
        }

        Ok(RenderedPrompt {
            prompt: Prompt::TokenIds(out),
            media_order: None,
            effective_template_kwargs: request.chat_options.template_kwargs.clone(),
        })
    }

    /// Resolve the effort as the reference does: the request's effort (or
    /// the deployment default) by name or number, numeric strings included;
    /// thinking toggles do not apply.
    fn inkling2_reasoning_effort(&self, request: &ChatRequest) -> Result<f64> {
        let kwarg = |kwargs: &HashMap<String, Value>| {
            kwargs.get("reasoning_effort").filter(|value| !value.is_null()).cloned()
        };
        let effort = match &request.chat_options.reasoning_effort {
            Some(effort) => Some(effort.clone()),
            None => kwarg(&request.chat_options.template_kwargs)
                .or_else(|| kwarg(&self.default_template_kwargs))
                .map(EffortValue::try_from)
                .transpose()?,
        };
        let value = match &effort {
            Some(EffortValue::String(text)) => {
                resolve_reasoning_effort(effort.as_ref()).or_else(|| text.parse().ok())
            }
            effort => resolve_reasoning_effort(effort.as_ref()),
        };
        value
            .filter(|value| (0.0..=MAX_REASONING_EFFORT).contains(value))
            .ok_or_else(|| {
                Error::InvalidReasoningEffort(format!(
                    "invalid Inkling reasoning_effort: {effort:?}"
                ))
            })
    }

    fn write_inkling2_effort(&self, out: &mut Vec<u32>, effort: f64) -> Result<()> {
        let millieffort = (effort * 1000.0).round_ties_even() as i64;
        let effort = ((millieffort as f64 / 1000.0) * 100.0).round_ties_even() / 100.0;
        self.write_text_block(
            out,
            self.special.message_system,
            None,
            &format!("Thinking effort level: {effort}"),
        )
    }

    fn write_inkling2_tool_declarations(
        &self,
        out: &mut Vec<u32>,
        request: &ChatRequest,
    ) -> Result<()> {
        let tools = request.tools();
        if tools.is_empty() {
            return Ok(());
        }
        let specs = tools
            .iter()
            .map(|tool| {
                let parameters = match &tool.parameters {
                    Value::Null => Value::Object(Map::new()),
                    parameters @ Value::Object(_) => parameters.clone(),
                    _ => {
                        return Err(Error::ChatTemplate(
                            "Inkling tool parameters must be a JSON object".to_string(),
                        ));
                    }
                };
                let mut spec = Map::new();
                spec.insert(
                    "description".to_string(),
                    Value::from(tool.description.clone().unwrap_or_default()),
                );
                spec.insert("name".to_string(), Value::from(tool.name.clone()));
                spec.insert("parameters".to_string(), parameters);
                spec.insert("type".to_string(), Value::from("function"));
                Ok(Value::Object(spec))
            })
            .collect::<Result<Vec<_>>>()?;
        let payload = tml_json(&Value::Array(specs), false)?;

        self.write_message_start(out, self.special.message_system, Some("tool_declare"))?;
        out.push(self.special.content_xml);
        self.write_text_tokens(out, &payload)?;
        out.push(self.special.end_message);
        Ok(())
    }

    fn write_inkling2_content(
        &self,
        out: &mut Vec<u32>,
        role_token_id: u32,
        content: &ChatContent,
    ) -> Result<()> {
        match content {
            ChatContent::Text(text) => self.write_text_block(out, role_token_id, None, text),
            ChatContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ChatContentPart::Text { text } => {
                            self.write_text_block(out, role_token_id, None, text)?;
                        }
                        ChatContentPart::ImageUrl { .. } => {
                            self.write_image_block(out, role_token_id);
                        }
                        ChatContentPart::InputAudio { .. } => {
                            return Err(Error::UnsupportedMultimodalContent("input_audio"));
                        }
                        ChatContentPart::AudioUrl { .. } => {
                            return Err(Error::UnsupportedMultimodalContent("audio_url"));
                        }
                        ChatContentPart::VideoUrl { .. } => {
                            return Err(Error::UnsupportedMultimodalContent("video_url"));
                        }
                        ChatContentPart::ToolReference { .. } => {
                            return Err(Error::UnsupportedContentPart("tool_reference"));
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn write_inkling2_assistant(
        &self,
        out: &mut Vec<u32>,
        content: &[AssistantContentBlock],
    ) -> Result<()> {
        // An empty string answer beside reasoning or tool calls is no block.
        let num_text = content
            .iter()
            .filter(|block| matches!(block, AssistantContentBlock::Text { .. }))
            .count();
        let drop_empty_text = num_text == 1 && num_text < content.len();
        let mut wrote_block = false;
        for block in content {
            match block {
                AssistantContentBlock::Reasoning { text } => {
                    out.push(self.special.message_model);
                    out.push(self.special.content_thinking);
                    self.write_text_tokens(out, text)?;
                    out.push(self.special.end_message);
                }
                AssistantContentBlock::Text { text } => {
                    if drop_empty_text && text.is_empty() {
                        continue;
                    }
                    self.write_text_block(out, self.special.message_model, None, text)?;
                }
                AssistantContentBlock::ToolCall(tool_call) => {
                    let (name, args) = tool_call_json_value(tool_call);
                    let payload = format!(
                        "{{\"name\":{},\"args\":{}}}",
                        compact_json(&Value::from(name))?,
                        tml_json(&args, true)?
                    );
                    out.push(self.special.message_model);
                    out.push(self.special.content_invoke_tool_json);
                    self.write_text_tokens(out, &payload)?;
                    out.push(self.special.end_message);
                }
            }
            wrote_block = true;
        }
        if wrote_block {
            out.push(self.special.content_model_end_sampling);
        }
        Ok(())
    }
}

/// Compact, key-sorted JSON with the Inkling2 reference float spelling:
/// Python's shortest `repr`, positional for decimal exponents in `[-5, 16)`,
/// and otherwise scientific with an explicit `+` unless `legacy_floats`.
fn tml_json(value: &Value, legacy_floats: bool) -> Result<String> {
    Ok(match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            let items = keys
                .into_iter()
                .map(|key| {
                    Ok(format!(
                        "{}:{}",
                        compact_json(&Value::from(key.as_str()))?,
                        tml_json(&map[key], legacy_floats)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            format!("{{{}}}", items.join(","))
        }
        Value::Array(items) => {
            let items = items
                .iter()
                .map(|item| tml_json(item, legacy_floats))
                .collect::<Result<Vec<_>>>()?;
            format!("[{}]", items.join(","))
        }
        Value::Number(number) if number.is_f64() => {
            tml_float(number.as_f64().unwrap_or_default(), legacy_floats)
        }
        _ => compact_json(value)?,
    })
}

fn tml_float(value: f64, legacy_floats: bool) -> String {
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("`{:e}` has an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` exponent is an integer");
    if !(-5..16).contains(&exponent) {
        let sign = if exponent >= 0 && !legacy_floats {
            "+"
        } else {
            ""
        };
        return format!("{mantissa}e{sign}{exponent}");
    }
    let negative = mantissa.starts_with('-');
    let digits = mantissa.trim_start_matches('-').replace('.', "");
    let decimal = exponent + 1;
    let body = if decimal <= 0 {
        format!("0.{}{digits}", "0".repeat(decimal.unsigned_abs() as usize))
    } else if decimal as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(decimal as usize - digits.len()))
    } else {
        let (int, frac) = digits.split_at(decimal as usize);
        format!("{int}.{frac}")
    };
    format!("{}{body}", if negative { "-" } else { "" })
}

#[cfg(test)]
mod tests;
