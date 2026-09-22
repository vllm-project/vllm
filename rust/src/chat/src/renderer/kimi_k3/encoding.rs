// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Kimi K3 XTML prompt renderer.
//!
//! Port of Moonshot remote-code `encoding_k3.py::build_chat_segments()`.

use std::collections::HashMap;

use serde_json::{Map, Value, json};
use vllm_tokenizer::Tokenizer;

use crate::error::{Error, Result};
use crate::request::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatTool, ChatToolChoice,
};
use crate::{AssistantContentBlock, AssistantToolCall};

pub(super) const OPEN: &str = "<|open|>";
pub(super) const CLOSE: &str = "<|close|>";
pub(super) const SEP: &str = "<|sep|>";
pub(super) const END_OF_MSG: &str = "<|end_of_msg|>";
pub(super) const IMAGE_PLACEHOLDER: &str = "<|media_pad|>";

const DEFAULT_THINKING_EFFORT: &str = "max";
const VALID_THINKING_EFFORTS: &[&str] = &["low", "high", "max"];

/// K3 prompt encoder preserving Python's per-segment tokenization boundaries.
pub(super) struct K3TokenWriter<'a> {
    tokenizer: &'a dyn Tokenizer,
    token_ids: Vec<u32>,
}

impl<'a> K3TokenWriter<'a> {
    pub(super) fn new(tokenizer: &'a dyn Tokenizer) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
        }
    }

    /// Encode one trusted segment with normal added-token recognition.
    pub(super) fn control(&mut self, text: &str) -> Result<()> {
        if !text.is_empty() {
            self.token_ids.extend(self.tokenizer.encode(text, false)?);
        }
        Ok(())
    }

    /// Encode one literal segment while bypassing every added-token matcher.
    pub(super) fn ordinary(&mut self, text: &str) -> Result<()> {
        if !text.is_empty() {
            self.token_ids.extend(self.tokenizer.encode_ordinary(text)?);
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Vec<u32> {
        self.token_ids
    }
}

/// Render and tokenize one chat request using K3's segment-aware contract.
pub(super) fn render_request(request: &ChatRequest, tokenizer: &dyn Tokenizer) -> Result<Vec<u32>> {
    let thinking = thinking_enabled(request)?;
    let thinking_effort = thinking.then(|| thinking_effort(request)).transpose()?;
    let tools = request_tools(request);
    let mut out = K3TokenWriter::new(tokenizer);

    if !tools.is_empty() {
        write_tool_declare(&mut out, tools, false)?;
    }

    if let Some(effort) = thinking_effort {
        // Preserve the checkpoint's literal guidance text: it still names
        // `medium`, although the validator above no longer accepts it.
        write_internal_system(
            &mut out,
            "thinking-effort",
            &format!(
                "`thinking_effort` guides on how much to think in your \
                 thinking channel (not including the response channel), \
                 supported values include `low`, `medium`, `high`, and `max`.\n\
                 Now the system is invoked with `thinking_effort={effort}`."
            ),
        )?;
    }

    // Track prior assistant tool-call ids for tool-result reordering / naming.
    let mut tool_call_id_index: HashMap<String, (usize, String)> = HashMap::new();
    let mut pending_tool_run: Vec<(usize, ChatMessage)> = Vec::new();

    let flush_tool_run = |out: &mut K3TokenWriter<'_>,
                          run: &mut Vec<(usize, ChatMessage)>,
                          id_index: &HashMap<String, (usize, String)>|
     -> Result<()> {
        if run.is_empty() {
            return Ok(());
        }

        let mut resolved = Vec::with_capacity(run.len());
        let mut unresolved = false;
        for (offset, message) in run.drain(..) {
            let ChatMessage::ToolResponse {
                content,
                tool_call_id,
            } = message
            else {
                unreachable!("pending tool run only holds tool responses");
            };
            match id_index.get(&tool_call_id) {
                Some(&(position, ref name)) => {
                    resolved.push((position, offset, content, Some(name.clone())));
                }
                None => {
                    unresolved = true;
                    resolved.push((usize::MAX, offset, content, None));
                }
            }
        }

        if unresolved {
            // Preserve original order when the run cannot be fully matched.
            resolved.sort_by_key(|item| item.1);
        } else {
            resolved.sort_by_key(|item| (item.0, item.1));
        }

        for (position, _, content, name) in resolved {
            let tool_name = name.as_deref().ok_or_else(|| {
                Error::ChatTemplate(
                    "Kimi K3 tool messages need a resolvable tool name: \
                         carry a matching tool_call_id against a preceding \
                         assistant tool_call"
                        .to_string(),
                )
            })?;
            write_tool_message(out, tool_name, position, &content)?;
        }
        Ok(())
    };

    for (message_index, message) in request.messages.iter().enumerate() {
        match message {
            ChatMessage::ToolResponse { .. } => {
                pending_tool_run.push((message_index, message.clone()));
                continue;
            }
            _ => {
                flush_tool_run(&mut out, &mut pending_tool_run, &tool_call_id_index)?;
            }
        }

        match message {
            // Python: role=system with a `tools` field renders a dynamic
            // tool-declare (`## New Tools Available`). Map that to Developer
            // messages that carry tools (OpenAI "developer" / system-tools).
            ChatMessage::Developer {
                content,
                tools: Some(local_tools),
            } if !local_tools.is_empty() => {
                write_tool_declare(&mut out, local_tools, true)?;
                if !content_is_empty(content) {
                    write_role_message(&mut out, "system", None, content)?;
                }
            }
            ChatMessage::System { content } | ChatMessage::Developer { content, .. } => {
                write_role_message(&mut out, "system", None, content)?;
            }
            ChatMessage::User { content } => {
                write_role_message(&mut out, "user", None, content)?;
            }
            ChatMessage::Assistant { content } => {
                tool_call_id_index.clear();
                let mut call_position = 0usize;
                for block in content {
                    if let AssistantContentBlock::ToolCall(call) = block {
                        call_position += 1;
                        if !call.id.is_empty() {
                            tool_call_id_index
                                .entry(call.id.clone())
                                .or_insert((call_position, call.name.clone()));
                        }
                    }
                }

                write_assistant_message(&mut out, content, thinking)?;
            }
            ChatMessage::ToolResponse { .. } => unreachable!("handled above"),
        }
    }
    flush_tool_run(&mut out, &mut pending_tool_run, &tool_call_id_index)?;

    match request.tool_choice() {
        ChatToolChoice::Required => {
            write_internal_system(
                &mut out,
                "tool-choice",
                "The system is invoked with `tool_choice=required`.\n\
                 You MUST call tools in the next message.",
            )?;
        }
        // Emit only when tools are present: Rust defaults tool_choice to None
        // for tool-free requests, which must not inject a tool-choice message.
        ChatToolChoice::None if !request.tools().is_empty() => {
            write_internal_system(
                &mut out,
                "tool-choice",
                "The system is invoked with `tool_choice=none`.\n\
                 You MUST NOT call any tools in the next message.",
            )?;
        }
        ChatToolChoice::None | ChatToolChoice::Auto | ChatToolChoice::Function { .. } => {}
    }

    write_response_format(&mut out, request)?;

    if request.chat_options.add_generation_prompt() {
        write_open_tag(&mut out, "message", &[("role", "assistant")])?;
        write_open_tag(&mut out, if thinking { "think" } else { "response" }, &[])?;
    }

    Ok(out.finish())
}

fn request_tools(request: &ChatRequest) -> &[ChatTool] {
    // Declare tools whenever the request carries them. K3 tool-declare is
    // independent of tool_choice; tool_choice only injects control messages.
    request.initial_tools()
}

fn thinking_enabled(request: &ChatRequest) -> Result<bool> {
    if let Some(thinking) = request.parse_template_bool("thinking")? {
        return Ok(thinking);
    }
    if let Some(enable_thinking) = request.parse_template_bool("enable_thinking")? {
        return Ok(enable_thinking);
    }
    Ok(request
        .chat_options
        .reasoning_effort
        .map(|effort| effort != crate::request::ReasoningEffort::None)
        .unwrap_or(true))
}

fn thinking_effort(request: &ChatRequest) -> Result<String> {
    let effort = if let Some(value) = request.chat_options.template_kwargs.get("thinking_effort") {
        value.as_str().ok_or_else(|| {
            Error::ChatTemplate(format!(
                "template kwarg `thinking_effort` must be a string, got {value}"
            ))
        })?
    } else if let Some(effort) = request.chat_options.reasoning_effort {
        effort.as_str()
    } else if let Some(value) = request.chat_options.template_kwargs.get("reasoning_effort") {
        value.as_str().ok_or_else(|| {
            Error::ChatTemplate(format!(
                "template kwarg `reasoning_effort` must be a string, got {value}"
            ))
        })?
    } else {
        DEFAULT_THINKING_EFFORT
    };

    if !VALID_THINKING_EFFORTS.contains(&effort) {
        return Err(Error::ChatTemplate(format!(
            "unsupported thinking_effort={effort:?}; supported values are `low`, `high`, and `max`"
        )));
    }
    Ok(effort.to_string())
}

fn content_is_empty(content: &ChatContent) -> bool {
    match content {
        ChatContent::Text(text) => text.is_empty(),
        ChatContent::Parts(parts) => parts.iter().all(|part| match part {
            ChatContentPart::Text { text } => text.is_empty(),
            ChatContentPart::ImageUrl { .. }
            | ChatContentPart::VideoUrl { .. }
            | ChatContentPart::InputAudio { .. }
            | ChatContentPart::AudioUrl { .. } => false,
        }),
    }
}

fn write_tool_declare(
    out: &mut K3TokenWriter<'_>,
    tools: &[ChatTool],
    dynamic: bool,
) -> Result<()> {
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        let mut function = Map::new();
        if let Some(description) = &tool.description {
            function.insert(
                "description".to_string(),
                Value::String(description.clone()),
            );
        }
        function.insert("name".to_string(), Value::String(tool.name.clone()));
        function.insert("parameters".to_string(), sort_json(&tool.parameters));
        if let Some(strict) = tool.strict {
            function.insert("strict".to_string(), Value::Bool(strict));
        }
        specs.push(json!({
            "function": Value::Object(function),
            "type": "function",
        }));
    }
    let payload = compact_json(&sort_json(&Value::Array(specs)))?;

    let body = if dynamic {
        format!(
            "## New Tools Available\n\
             The system dynamically extends the toolset via lazy-loading.\n\
             You have access to all existing and extended tools.\n\
             Here are the specs for the extended tools.\n\n\
             ```json\n\
             {payload}\n\
             ```"
        )
    } else {
        format!(
            "# Tools\n\
             Here are the available tools, described in JSONSchema.\n\n\
             ```json\n\
             {payload}\n\
             ```"
        )
    };

    write_internal_system(out, "tool-declare", &body)
}

fn write_internal_system(
    out: &mut K3TokenWriter<'_>,
    message_type: &str,
    body: &str,
) -> Result<()> {
    write_open_tag(
        out,
        "message",
        &[("role", "system"), ("type", message_type)],
    )?;
    out.ordinary(body.trim())?;
    write_close_tag(out, "message")?;
    out.control(END_OF_MSG)
}

fn write_role_message(
    out: &mut K3TokenWriter<'_>,
    role: &str,
    name: Option<&str>,
    content: &ChatContent,
) -> Result<()> {
    let mut attrs = vec![("role", role.to_string())];
    if let Some(name) = name.filter(|name| !name.is_empty()) {
        attrs.push(("name", name.to_string()));
    }
    let attr_refs: Vec<(&str, &str)> = attrs.iter().map(|(k, v)| (*k, v.as_str())).collect();
    write_open_tag(out, "message", &attr_refs)?;
    write_content(out, content)?;
    write_close_tag(out, "message")?;
    out.control(END_OF_MSG)
}

fn write_tool_message(
    out: &mut K3TokenWriter<'_>,
    tool_name: &str,
    index: usize,
    content: &ChatContent,
) -> Result<()> {
    let index_str = index.to_string();
    write_open_tag(
        out,
        "message",
        &[("role", "tool"), ("tool", tool_name), ("index", &index_str)],
    )?;
    write_content(out, content)?;
    write_close_tag(out, "message")?;
    out.control(END_OF_MSG)
}

fn write_assistant_message(
    out: &mut K3TokenWriter<'_>,
    content: &[AssistantContentBlock],
    thinking: bool,
) -> Result<()> {
    write_open_tag(out, "message", &[("role", "assistant")])?;

    let mut reasoning = String::new();
    let mut response = String::new();
    let mut tool_calls = Vec::new();
    for block in content {
        match block {
            AssistantContentBlock::Reasoning { text } => reasoning.push_str(text),
            AssistantContentBlock::Text { text } => response.push_str(text),
            AssistantContentBlock::ToolCall(call) => tool_calls.push(call),
        }
    }

    // The think channel is structural: in thinking mode every assistant
    // message carries open/close tags even when there is no reasoning content.
    // In non-thinking mode the channel is dropped entirely.
    if thinking {
        write_open_tag(out, "think", &[])?;
        if !reasoning.trim().is_empty() {
            out.ordinary(&reasoning)?;
        }
        write_close_tag(out, "think")?;
    }

    write_open_tag(out, "response", &[])?;
    out.ordinary(&response)?;
    write_close_tag(out, "response")?;

    if !tool_calls.is_empty() {
        write_open_tag(out, "tools", &[])?;
        for (index, tool_call) in tool_calls.into_iter().enumerate() {
            write_assistant_tool_call(out, tool_call, index + 1)?;
        }
        write_close_tag(out, "tools")?;
    }

    write_close_tag(out, "message")?;
    out.control(END_OF_MSG)
}

fn write_assistant_tool_call(
    out: &mut K3TokenWriter<'_>,
    tool_call: &AssistantToolCall,
    index: usize,
) -> Result<()> {
    let index_str = index.to_string();
    write_open_tag(
        out,
        "call",
        &[
            ("tool", tool_call.name.as_str()),
            ("index", index_str.as_str()),
        ],
    )?;

    let (args, json_block) = normalize_tool_arguments(&tool_call.arguments)?;
    if let Some(raw) = json_block {
        write_open_tag(out, "json", &[("type", "object")])?;
        out.ordinary(&raw)?;
        write_close_tag(out, "json")?;
    } else {
        for (key, value) in args {
            let typ = xtml_type(&value);
            write_open_tag(out, "argument", &[("key", key.as_str()), ("type", typ)])?;
            out.ordinary(&xtml_value(&value))?;
            write_close_tag(out, "argument")?;
        }
    }

    write_close_tag(out, "call")
}

fn write_content(out: &mut K3TokenWriter<'_>, content: &ChatContent) -> Result<()> {
    match content {
        ChatContent::Text(text) => write_text_with_images(out, text),
        ChatContent::Parts(parts) => {
            for part in parts {
                match part {
                    ChatContentPart::Text { text } => write_text_with_images(out, text)?,
                    ChatContentPart::ImageUrl { .. } => out.control(IMAGE_PLACEHOLDER)?,
                    ChatContentPart::VideoUrl { .. } => {
                        return Err(Error::UnsupportedMultimodalContent("video_url"));
                    }
                    ChatContentPart::InputAudio { .. } => {
                        return Err(Error::UnsupportedMultimodalContent("input_audio"));
                    }
                    ChatContentPart::AudioUrl { .. } => {
                        return Err(Error::UnsupportedMultimodalContent("audio_url"));
                    }
                }
            }
            Ok(())
        }
    }
}

fn write_text_with_images(out: &mut K3TokenWriter<'_>, text: &str) -> Result<()> {
    // Placeholder expansion is left as the literal K3 image token; multimodal
    // preprocessing can replace it once image prompts are known.
    out.ordinary(text)
}

fn write_response_format(out: &mut K3TokenWriter<'_>, request: &ChatRequest) -> Result<()> {
    let Some(rf) = request.chat_options.response_format.as_ref() else {
        return Ok(());
    };

    let rf_type = rf.get("type").and_then(Value::as_str).or_else(|| rf.as_str()).unwrap_or("");

    match rf_type {
        "json_object" => {
            write_internal_system(
                out,
                "response-format",
                "The system is invoked with `response_format=json_object`.\n\
                 Your response must be raw JSON data without markdown code \
                 blocks (```json) or any additional formatting.",
            )?;
        }
        "json_schema" => {
            let schema = extract_response_schema(rf);
            let schema_json = compact_json(&sort_json(&schema.unwrap_or(Value::Null)))?;
            write_internal_system(
                out,
                "response-format",
                &format!(
                    "The system is invoked with `response_format=json_schema`.\n\
                     Your response must be raw JSON data without markdown code \
                     blocks (```json) or any additional formatting.\n\
                     The JSON data must match the following schema:\n\
                     ```json\n\
                     {schema_json}\n\
                     ```"
                ),
            )?;
        }
        _ => {}
    }
    Ok(())
}

fn extract_response_schema(response_format: &Value) -> Option<Value> {
    let json_schema = response_format.get("json_schema")?;
    if let Some(schema) = json_schema.get("schema") {
        return Some(schema.clone());
    }
    if let Some(schema) = json_schema.get("json_schema") {
        return Some(schema.clone());
    }
    Some(json_schema.clone())
}

fn normalize_tool_arguments(arguments: &str) -> Result<(Map<String, Value>, Option<String>)> {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return Ok((Map::new(), None));
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Object(map)) => Ok((map, None)),
        Ok(_) => Err(Error::ChatTemplate(
            "Kimi K3 tool call arguments must be a JSON object".to_string(),
        )),
        Err(_) => Ok((Map::new(), Some(arguments.to_string()))),
    }
}

fn xtml_type(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}

fn xtml_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => compact_json(other).unwrap_or_else(|_| other.to_string()),
    }
}

fn write_open_tag(out: &mut K3TokenWriter<'_>, tag: &str, attrs: &[(&str, &str)]) -> Result<()> {
    out.control(OPEN)?;
    out.ordinary(tag)?;
    for (key, value) in attrs {
        out.ordinary(&format!(" {key}"))?;
        out.ordinary("=\"")?;
        out.ordinary(&escape_attr_value(value))?;
        out.ordinary("\"")?;
    }
    out.control(SEP)
}

fn write_close_tag(out: &mut K3TokenWriter<'_>, tag: &str) -> Result<()> {
    out.control(CLOSE)?;
    out.ordinary(tag)?;
    out.control(SEP)
}

fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

fn compact_json(value: &Value) -> Result<String> {
    serde_json::to_string(value).map_err(|error| Error::ChatTemplate(error.to_string()))
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
