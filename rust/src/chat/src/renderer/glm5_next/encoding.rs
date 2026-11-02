// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{self, Write};

use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};
use serde_json::{Value, json};
use thiserror_ext::AsReport as _;

use super::writer::Glm5PromptWriter;
use crate::error::{Error, Result, bail_chat_template_thrown, chat_template_thrown};
use crate::renderer::MediaPartSource;
use crate::request::{ChatContent, ChatContentPart, ChatMessage, ChatTool, GenerationPromptMode};
use crate::{AssistantContentBlock, AssistantMessageExt};

const TOOLS_PREFIX: &str = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
const TOOLS_SUFFIX: &str = "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

pub(super) fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

#[derive(Default)]
struct MediaContext {
    enabled: bool,
    order: Vec<MediaPartSource>,
}

fn visible_text(
    out: &mut Glm5PromptWriter<'_>,
    value: &Value,
    media: &mut MediaContext,
) -> Result<()> {
    match value {
        Value::Null => {}
        Value::String(value) => out.ordinary(value),
        Value::Array(parts) => {
            for part in parts {
                if let Some(part) = part.as_str() {
                    out.ordinary(part);
                    continue;
                }
                match part["type"].as_str().unwrap_or_default() {
                    "text" => out.ordinary(&part.get("text").map(text).unwrap_or_default()),
                    kind @ ("image" | "image_url" | "video" | "video_url" | "audio"
                    | "audio_url" | "input_audio") => {
                        let kind = kind.replace("_url", "").replace("input_", "");
                        if media.enabled {
                            out.control(&format!("<|begin_of_{kind}|>"))?;
                            if kind != "audio" {
                                out.control(&format!("<|{kind}|>"))?;
                            }
                            out.control(&format!("<|end_of_{kind}|>"))?;
                            if let (Some(message_index), Some(content_part_index)) = (
                                part["message_index"].as_u64(),
                                part["content_part_index"].as_u64(),
                            ) {
                                media.order.push(MediaPartSource {
                                    message_index: message_index as usize,
                                    content_part_index: content_part_index as usize,
                                });
                            }
                        } else {
                            out.control(&format!("<reminder>You are unable to process this {kind} because you don't have multi-modal input ability. Try different methods.</reminder>"))?;
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => out.ordinary(&text(value)),
    }
    Ok(())
}

fn tool_definition(tool: &Value) -> Result<String> {
    let Some(fields) = tool.as_object() else {
        bail_chat_template_thrown!("Each tool must be a JSON object");
    };
    let mut out = String::from("{");
    for (key, value) in fields {
        if key == "strict" || key == "defer_loading" {
            continue;
        }
        if out.len() > 1 {
            out.push_str(", ");
        }
        write!(out, "\"{key}\": {}", json(value)?).unwrap();
    }
    out.push('}');
    Ok(out)
}

fn unwrap_tool(tool: &Value) -> &Value {
    tool.get("function").unwrap_or(tool)
}

fn tool_references(value: &Value) -> bool {
    value
        .as_array()
        .and_then(|parts| parts.first())
        .is_some_and(|part| part["type"] == "tool_reference")
}

fn reference_response(out: &mut Glm5PromptWriter<'_>, refs: &Value, tools: &[Value]) -> Result<()> {
    if tools.is_empty() {
        bail_chat_template_thrown!("Rendering a tool_reference requires tools");
    }
    out.control("<tool_response><tools>\n")?;
    for reference in refs.as_array().into_iter().flatten() {
        for tool in tools.iter().map(unwrap_tool) {
            if tool.get("name") == reference.get("name") {
                out.ordinary(&tool_definition(tool)?);
                out.control("\n")?;
            }
        }
    }
    out.control("</tools></tool_response>")
}

fn response(
    out: &mut Glm5PromptWriter<'_>,
    value: &Value,
    tools: &[Value],
    media: &mut MediaContext,
) -> Result<()> {
    if tool_references(value) {
        reference_response(out, value, tools)
    } else {
        out.control("<tool_response>")?;
        visible_text(out, value, media)?;
        out.control("</tool_response>")
    }
}

fn bundled_outputs(value: &Value) -> Result<bool> {
    if !truthy(value) || value.is_string() {
        return Ok(false);
    }
    let Some(parts) = value.as_array() else {
        bail_chat_template_thrown!("A tool message content must be a string or a list");
    };
    Ok(parts.first().is_some_and(|part| part.get("output").is_some()))
}

fn tool_response(
    out: &mut Glm5PromptWriter<'_>,
    message: &Value,
    tools: &[Value],
    media: &mut MediaContext,
) -> Result<()> {
    let value = &message["content"];
    if tool_references(value) {
        return reference_response(out, value, tools);
    }
    if bundled_outputs(value)? {
        for part in value.as_array().unwrap() {
            response(out, &part["output"], tools, media)?;
        }
        Ok(())
    } else {
        response(out, value, tools, media)
    }
}

fn id(value: &Value) -> String {
    ["tool_call_id", "id"]
        .into_iter()
        .filter_map(|key| value.get(key))
        .find(|value| truthy(value))
        .map(text)
        .unwrap_or_default()
}

fn ordered_tool_block(
    out: &mut Glm5PromptWriter<'_>,
    block: &[Value],
    assistant: Option<&Value>,
    tools: &[Value],
    media: &mut MediaContext,
) -> Result<()> {
    let calls = assistant
        .filter(|message| message["role"] == "assistant")
        .and_then(|message| message["tool_calls"].as_array())
        .filter(|calls| !calls.is_empty());
    if let Some(calls) = calls {
        let mut can_sort = true;
        let mut responses = HashMap::new();
        for message in block {
            if bundled_outputs(&message["content"])? {
                for entry in message["content"].as_array().unwrap() {
                    let key = id(entry);
                    if key.is_empty() || responses.insert(key, &entry["output"]).is_some() {
                        can_sort = false;
                    }
                }
            } else {
                let key = id(message);
                if key.is_empty() || responses.insert(key, &message["content"]).is_some() {
                    can_sort = false;
                }
            }
        }
        let call_ids: Vec<_> = calls.iter().map(id).collect();
        let mut seen = HashSet::new();
        for key in &call_ids {
            if key.is_empty() || !seen.insert(key) {
                can_sort = false;
            }
        }
        if can_sort && responses.keys().all(|key| seen.contains(key)) {
            for value in call_ids.iter().filter_map(|key| responses.get(key)) {
                response(out, value, tools, media)?;
            }
            return Ok(());
        }
    }
    for message in block {
        tool_response(out, message, tools, media)?;
    }
    Ok(())
}

fn tool_call(out: &mut Glm5PromptWriter<'_>, call: &Value) -> Result<()> {
    let call = call.get("function").filter(|value| truthy(value)).unwrap_or(call);
    let Some(name) = call["name"].as_str() else {
        bail_chat_template_thrown!("Each tool_call needs a string name");
    };
    let parsed;
    let arguments = if let Some(raw) = call["arguments"].as_str() {
        parsed = serde_json::from_str::<Value>(raw).map_err(|error| {
            chat_template_thrown!("Invalid tool_call arguments: {}", error.as_report())
        })?;
        &parsed
    } else {
        &call["arguments"]
    };
    let Some(arguments) = arguments.as_object() else {
        bail_chat_template_thrown!("tool_call arguments must be a JSON object");
    };
    out.control("<tool_call>")?;
    out.ordinary(name);
    for (key, value) in arguments {
        out.control("<arg_key>")?;
        out.ordinary(key);
        out.control("</arg_key><arg_value>")?;
        if let Some(value) = value.as_str() {
            out.ordinary(value);
        } else {
            out.ordinary(&json(value)?);
        }
        out.control("</arg_value>")?;
    }
    out.control("</tool_call>")
}

#[cfg(test)]
pub(super) fn render(
    messages: &[Value],
    tools: &[Value],
    mode: GenerationPromptMode,
    kwargs: &HashMap<String, Value>,
) -> Result<String> {
    let mut out = Glm5PromptWriter::new(None);
    render_into(&mut out, messages, tools, mode, kwargs, false)?;
    Ok(out.text().to_owned())
}

/// Render into `out`, returning media parts in placeholder order.
pub(super) fn render_into(
    out: &mut Glm5PromptWriter<'_>,
    messages: &[Value],
    tools: &[Value],
    mode: GenerationPromptMode,
    kwargs: &HashMap<String, Value>,
    multimodal: bool,
) -> Result<Vec<MediaPartSource>> {
    let mut media = MediaContext {
        enabled: multimodal,
        order: Vec::new(),
    };
    render_inner(out, messages, tools, mode, kwargs, &mut media)?;
    Ok(media.order)
}

fn render_inner(
    out: &mut Glm5PromptWriter<'_>,
    messages: &[Value],
    tools: &[Value],
    mode: GenerationPromptMode,
    kwargs: &HashMap<String, Value>,
    media: &mut MediaContext,
) -> Result<()> {
    let effort = match kwargs.get("reasoning_effort").and_then(Value::as_str) {
        Some("low") => "Low",
        Some("high") => "High",
        _ => "Max",
    };
    let clear_thinking = kwargs.get("clear_thinking").is_some_and(truthy);
    out.control(&format!("[gMASK]<sop><|system|>Reasoning Effort: {effort}"))?;
    if !tools.is_empty() {
        out.control(TOOLS_PREFIX)?;
        for tool in tools.iter().map(unwrap_tool) {
            if !truthy(&tool["defer_loading"]) {
                out.ordinary(&tool_definition(tool)?);
                out.control("\n")?;
            }
        }
        out.control(TOOLS_SUFFIX)?;
    }
    let last_user = messages.iter().rposition(|message| message["role"] == "user");
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        match message["role"].as_str().unwrap_or_default() {
            "user" | "system" => {
                out.control(&format!("<|{}|>", message["role"].as_str().unwrap()))?;
                visible_text(out, &message["content"], media)?;
            }
            "assistant" => {
                out.control("<|assistant|><think>")?;
                let mut content = {
                    let mut text = Glm5PromptWriter::new(None);
                    visible_text(&mut text, &message["content"], media)?;
                    text.text().to_owned()
                };
                let original = content.clone();
                let mut reasoning = message["reasoning_content"].as_str().map(str::to_owned);
                if reasoning.is_none() && content.contains("</think>") {
                    reasoning = Some(
                        content
                            .split("</think>")
                            .next()
                            .unwrap()
                            .rsplit("<think>")
                            .next()
                            .unwrap()
                            .to_owned(),
                    );
                    content = content.rsplit("</think>").next().unwrap().to_owned();
                }
                if (!clear_thinking || last_user.is_none_or(|last| index > last))
                    && let Some(reasoning) = reasoning
                {
                    out.ordinary(&reasoning);
                }
                out.control("</think>")?;
                let trim = |ch: char| ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch);
                if mode == GenerationPromptMode::ContinueFinalAssistant
                    && index + 1 == messages.len()
                {
                    // Match HF continuation, which truncates the template output
                    // at the end of the final content: tool calls are dropped,
                    // trailing whitespace is trimmed, and content the template
                    // rewrote is rejected.
                    out.ordinary(content.trim_matches(trim));
                    let original = original.trim_matches(trim);
                    if !out.text().contains(original) {
                        return Err(Error::ChatTemplate(format!(
                            "continue_final_message is set but the final message does not appear \
                             in the prompt after applying the chat template! This can happen if \
                             the chat template deletes portions of the final message. Final \
                             message to continue: {original}"
                        )));
                    }
                    break;
                }
                out.ordinary(content.trim_matches(trim));
                for call in message["tool_calls"].as_array().into_iter().flatten() {
                    tool_call(out, call)?;
                }
            }
            "tool" => {
                let start = index;
                while index + 1 < messages.len() && messages[index + 1]["role"] == "tool" {
                    index += 1;
                }
                out.control("<|observation|>")?;
                ordered_tool_block(
                    out,
                    &messages[start..=index],
                    start.checked_sub(1).map(|i| &messages[i]),
                    tools,
                    media,
                )?;
            }
            _ => {}
        }
        index += 1;
    }
    if mode == GenerationPromptMode::StartNewAssistant {
        out.control("<|assistant|><think>")?;
    }
    Ok(())
}

// TODO: Expose tool_reference and bundled outputs in ChatContentPart.

fn content(value: &ChatContent, message_index: usize) -> Value {
    match value {
        ChatContent::Text(text) => json!(text),
        ChatContent::Parts(parts) => Value::Array(
            parts
                .iter()
                .enumerate()
                .map(|(content_part_index, part)| match part {
                    ChatContentPart::Text { text } => json!({"type": "text", "text": text}),
                    ChatContentPart::ImageUrl { .. } => json!({"type": "image", "message_index": message_index, "content_part_index": content_part_index}),
                    ChatContentPart::VideoUrl { .. } => json!({"type": "video", "message_index": message_index, "content_part_index": content_part_index}),
                    ChatContentPart::InputAudio { .. } | ChatContentPart::AudioUrl { .. } => {
                        json!({"type": "audio", "message_index": message_index, "content_part_index": content_part_index})
                    }
                })
                .collect(),
        ),
    }
}

pub(super) fn messages(messages: &[ChatMessage]) -> Vec<Value> {
    messages.iter().enumerate().map(|(index, message)| match message {
        ChatMessage::System { content: value } => json!({"role": "system", "content": content(value, index)}),
        ChatMessage::Developer { content: value, .. } => json!({"role": "developer", "content": content(value, index)}),
        ChatMessage::User { content: value } => json!({"role": "user", "content": content(value, index)}),
        ChatMessage::Custom { role, content: value } => json!({"role": role, "content": content(value, index)}),
        ChatMessage::ToolResponse { content: value, tool_call_id } => {
            json!({"role": "tool", "content": content(value, index), "tool_call_id": tool_call_id})
        }
        ChatMessage::Assistant { content } => {
            let mut message = json!({"role": "assistant", "content": content.text()});
            if content.iter().any(|block| matches!(block, AssistantContentBlock::Reasoning { .. })) {
                message["reasoning_content"] = json!(content.iter().filter_map(|block| match block {
                    AssistantContentBlock::Reasoning { text } => Some(text.as_str()),
                    _ => None,
                }).collect::<String>());
            }
            if content.has_tool_calls() {
                message["tool_calls"] = Value::Array(content.tool_calls().map(|call| {
                    json!({"id": call.id, "function": {"name": call.name, "arguments": call.arguments}})
                }).collect());
            }
            message
        }
    }).collect()
}

pub(super) fn tools(tools: &[ChatTool]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({"function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "defer_loading": tool.defer_loading,
            }})
        })
        .collect()
}

struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        let magnitude = value.abs();
        if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
            let scientific = format!("{value:e}");
            let (mantissa, exponent) = scientific.split_once('e').unwrap();
            let exponent: i32 = exponent.parse().unwrap();
            write!(writer, "{mantissa}e{exponent:+03}")
        } else {
            let decimal = value.to_string();
            writer.write_all(decimal.as_bytes())?;
            if !decimal.contains('.') {
                writer.write_all(b".0")?;
            }
            Ok(())
        }
    }
}

fn json(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    value
        .serialize(&mut Serializer::with_formatter(&mut bytes, PythonFormatter))
        .map_err(|error| {
            chat_template_thrown!("GLM JSON serialization failed: {}", error.as_report())
        })?;
    Ok(String::from_utf8(bytes).expect("JSON serializer must write UTF-8"))
}
