// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Inkling2 rendering.
//!
//! Inkling2 keeps Inkling's block format, but its reference renderer differs:
//! no generation prefix (the model emits its own `<|message_model|>`), empty
//! string contents kept as empty blocks, payload text encoded without
//! special-token matching, effort quantized to integer milli-effort, and its
//! own canonical JSON spelling for floats.

use std::collections::HashMap;
use std::fmt::Write as _;

use serde_json::{Map, Value};
use vllm_text::Prompt;

use super::{
    InklingChatRenderer, MAX_REASONING_EFFORT, compact_json, resolve_reasoning_effort,
    tool_call_arguments_or_empty,
};
use crate::error::{Error, Result};
use crate::request::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatTool, ChatToolChoice,
};
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
        let tools = inkling2_declared_tools(request);
        self.write_inkling2_tool_declarations(&mut out, &tools)?;

        let mut effort_text = Some(inkling2_effort_text(effort));
        let mut first_system_content = None;
        if !tools.is_empty()
            && let Some(instruction) =
                tool_choice_instruction(request.tool_choice(), request.parallel_tool_calls())
        {
            match messages.first() {
                Some(ChatMessage::System { content } | ChatMessage::Developer { content, .. }) => {
                    first_system_content = Some(with_appended_text(content, &instruction));
                }
                _ => {
                    effort_text = effort_text.map(|text| format!("{text}\n\n{instruction}"));
                }
            }
        }

        let mut tool_call_id_to_name = HashMap::new();
        for (index, message) in messages.iter().enumerate() {
            if !matches!(
                message,
                ChatMessage::System { .. } | ChatMessage::Developer { .. }
            ) && let Some(text) = effort_text.take()
            {
                self.write_text_block(&mut out, self.special.message_system, None, &text)?;
            }

            match message {
                ChatMessage::System { content } | ChatMessage::Developer { content, .. } => {
                    let content = match &first_system_content {
                        Some(first) if index == 0 => first,
                        _ => content,
                    };
                    self.write_inkling2_content(
                        &mut out,
                        self.special.message_system,
                        None,
                        content,
                    )?;
                }
                ChatMessage::User { content } => {
                    self.write_inkling2_content(
                        &mut out,
                        self.special.message_user,
                        None,
                        content,
                    )?;
                }
                ChatMessage::Assistant { content } => {
                    self.write_inkling2_assistant(&mut out, content, &mut tool_call_id_to_name)?;
                }
                ChatMessage::ToolResponse {
                    content,
                    tool_call_id,
                } => {
                    let name = tool_call_id_to_name.get(tool_call_id).map(String::as_str);
                    self.write_inkling2_content(
                        &mut out,
                        self.special.message_tool,
                        name,
                        content,
                    )?;
                }
                ChatMessage::Custom { role, .. } => {
                    return Err(Error::UnsupportedChatRole { role: role.clone() });
                }
            }
        }
        if let Some(text) = effort_text {
            self.write_text_block(&mut out, self.special.message_system, None, &text)?;
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

    fn write_inkling2_tool_declarations(
        &self,
        out: &mut Vec<u32>,
        tools: &[&ChatTool],
    ) -> Result<()> {
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
        let payload = tml_json(&Value::Array(specs), false, true)?;

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
        author_name: Option<&str>,
        content: &ChatContent,
    ) -> Result<()> {
        match content {
            ChatContent::Text(text) => self.write_text_block(out, role_token_id, author_name, text),
            ChatContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ChatContentPart::Text { text } => {
                            self.write_text_block(out, role_token_id, author_name, text)?;
                        }
                        ChatContentPart::ImageUrl { .. } => {
                            self.write_image_block(out, role_token_id, author_name)?;
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
        tool_call_id_to_name: &mut HashMap<String, String>,
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
                    if !tool_call.id.is_empty() {
                        tool_call_id_to_name.insert(tool_call.id.clone(), tool_call.name.clone());
                    }
                    let mut args = tool_call_arguments_or_empty(&tool_call.arguments);
                    args.sort_keys();
                    let payload = format!(
                        "{{\"name\":{},\"args\":{}}}",
                        compact_json(&Value::from(tool_call.name.as_str()))?,
                        tml_json(&Value::Object(args), true, false)?
                    );
                    self.write_message_start(
                        out,
                        self.special.message_model,
                        Some(&tool_call.name),
                    )?;
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

fn inkling2_effort_text(effort: f64) -> String {
    let millieffort = (effort * 1000.0).round_ties_even() as i64;
    let effort = ((millieffort as f64 / 1000.0) * 100.0).round_ties_even() / 100.0;
    format!("Thinking effort level: {effort}")
}

fn inkling2_declared_tools(request: &ChatRequest) -> Vec<&ChatTool> {
    match request.tool_choice() {
        ChatToolChoice::None => request
            .messages
            .iter()
            .filter_map(ChatMessage::declared_tools)
            .flatten()
            .collect(),
        ChatToolChoice::Function { name } => request.tool(name).into_iter().collect(),
        ChatToolChoice::Auto | ChatToolChoice::Required => request.tools().iter().collect(),
    }
}

fn tool_choice_instruction(
    tool_choice: &ChatToolChoice,
    parallel_tool_calls: bool,
) -> Option<String> {
    match tool_choice {
        ChatToolChoice::None => None,
        ChatToolChoice::Auto if parallel_tool_calls => None,
        ChatToolChoice::Auto => Some(
            "If you call a function, call at most one function in your next response.".to_string(),
        ),
        ChatToolChoice::Required => {
            let quantity = if parallel_tool_calls {
                "at least one"
            } else {
                "exactly one"
            };
            Some(format!(
                "Call {quantity} of the provided functions in your next response."
            ))
        }
        ChatToolChoice::Function { name } => {
            let once = if parallel_tool_calls {
                ""
            } else {
                " exactly once"
            };
            Some(format!(
                "Call the function named {}{once} in your next response.",
                ascii_json_string(name)
            ))
        }
    }
}

fn ascii_json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for ch in Value::from(text).to_string().chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            for unit in ch.encode_utf16(&mut [0; 2]).iter() {
                write!(out, "\\u{unit:04x}").expect("writing to a String cannot fail");
            }
        }
    }
    out
}

fn with_appended_text(content: &ChatContent, text: &str) -> ChatContent {
    match content {
        ChatContent::Text(existing) if !existing.is_empty() => {
            ChatContent::Text(format!("{existing}\n\n{text}"))
        }
        ChatContent::Parts(parts) if !parts.is_empty() => {
            let mut parts = parts.clone();
            parts.push(ChatContentPart::text(text));
            ChatContent::Parts(parts)
        }
        ChatContent::Text(_) | ChatContent::Parts(_) => ChatContent::Text(text.to_string()),
    }
}

/// Compact JSON with the Inkling2 reference float spelling: Python's shortest
/// `repr`, positional for decimal exponents in `[-5, 16)`, and otherwise
/// scientific with an explicit `+` unless `legacy_floats`.
fn tml_json(value: &Value, legacy_floats: bool, sort_keys: bool) -> Result<String> {
    Ok(match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            if sort_keys {
                keys.sort();
            }
            let items = keys
                .into_iter()
                .map(|key| {
                    Ok(format!(
                        "{}:{}",
                        compact_json(&Value::from(key.as_str()))?,
                        tml_json(&map[key], legacy_floats, sort_keys)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            format!("{{{}}}", items.join(","))
        }
        Value::Array(items) => {
            let items = items
                .iter()
                .map(|item| tml_json(item, legacy_floats, sort_keys))
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
