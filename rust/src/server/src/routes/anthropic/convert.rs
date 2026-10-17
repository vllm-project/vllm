// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::Value;
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart,
    ChatMessage as VllmChatMessage, ChatOptions, ChatRequest, ChatTool, ChatToolChoice,
    GenerationPromptMode, ResolvedToolContext, SamplingParams,
};
use vllm_text::output::TextDecodeOptions;

use super::types::{
    AnthropicContentBlock, AnthropicCountTokensRequest, AnthropicMessage, AnthropicRole,
    AnthropicTool, AnthropicToolChoice, ImageSource, MessageContent, SystemPrompt, SystemTextBlock,
    ToolResultContent,
};
use crate::error::{ApiError, bail_invalid_request, chat_submit_error};

/// Claude Code attaches an attribution block whose per-request hash defeats
/// prefix caching; Python strips it and so do we. The oracle has no named
/// constant — it inlines this literal at every check site.
const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header";

// ============================================================================
// Oracle-mirrored conversions
// ============================================================================
// Each fn below maps 1:1 to a named method of the Python vLLM serving class,
// listed in oracle definition order.

/// Convert an Anthropic image source to a URL string.
///
/// Mirrors the Python vLLM `_convert_image_source_to_url`: `url` sources
/// pass through (empty string when the URL is missing); everything else
/// becomes a base64 data URI with `image/jpeg` as the fallback media type.
pub(super) fn convert_image_source_to_url(source: &ImageSource) -> String {
    if source.source_type.as_deref() == Some("url") {
        return source.url.clone().unwrap_or_default();
    }
    let media_type = source.media_type.as_deref().unwrap_or("image/jpeg");
    let data = source.data.as_deref().unwrap_or("");
    format!("data:{media_type};base64,{data}")
}

/// Extract the effective text of one inline `role: system` message.
///
/// Mirrors the Python vLLM `_extract_system_text`: strings and text blocks
/// pass through with billing headers stripped; `None` when nothing remains.
fn extract_inline_system_text(content: &MessageContent) -> Option<String> {
    let text = match content {
        MessageContent::Text(text) => clean_system_text(text)?.to_string(),
        MessageContent::Blocks(blocks) => {
            let mut parts = String::new();
            for block in blocks {
                if let AnthropicContentBlock::Text { text } = block
                    && !text.is_empty()
                    && let Some(clean) = clean_system_text(text)
                {
                    parts.push_str(clean);
                }
            }
            parts
        }
    };
    (!text.is_empty()).then_some(text)
}

/// Lower the Anthropic system prompt plus conversation history into internal
/// chat messages.
///
/// Mirrors the Python vLLM `_convert_messages`, folding in
/// `_convert_system_message`: the leading system block (system prompt plus
/// hoisted inline system messages) is built here rather than by the caller.
///
/// `merge_inline_system` mirrors Python's `_merge_inline_system` switch:
/// when `true`, inline `role: system` messages are hoisted into the leading
/// system block (for templates that reject mid-conversation system messages);
/// when `false`, they stay at their original position to keep prefix-cache
/// hits intact. Python chooses the flag per model via a startup template
/// probe (`_detect_merge_inline_system`); PR 1 passes the conservative
/// always-merge default from the route layer and defers the probe to PR 2.
pub(super) fn convert_messages(
    system: Option<SystemPrompt>,
    messages: Vec<AnthropicMessage>,
    merge_inline_system: bool,
) -> Result<Vec<VllmChatMessage>, ApiError> {
    let mut out = Vec::new();

    let mut system_parts = String::new();
    match &system {
        Some(SystemPrompt::Text(text)) => system_parts.push_str(text),
        Some(SystemPrompt::Blocks(blocks)) => {
            system_parts.push_str(&system_text_from_blocks(blocks))
        }
        None => {}
    }
    if merge_inline_system {
        for message in &messages {
            if matches!(message.role, AnthropicRole::System)
                && let Some(text) = extract_inline_system_text(&message.content)
            {
                system_parts.push_str(&text);
            }
        }
    }
    if !system_parts.is_empty() {
        out.push(VllmChatMessage::system(system_parts));
    }

    for message in messages {
        match message.role {
            AnthropicRole::System => {
                // Already hoisted above when merging; otherwise keep the
                // message at its original position (billing headers stripped,
                // dropped entirely if nothing remains).
                if !merge_inline_system
                    && let Some(text) = extract_inline_system_text(&message.content)
                {
                    out.push(VllmChatMessage::system(text));
                }
            }
            AnthropicRole::User => convert_user_message(message.content, &mut out)?,
            AnthropicRole::Assistant => convert_assistant_message(message.content, &mut out)?,
        }
    }

    Ok(out)
}

/// Lower one `tool_result` block into a `ToolResponse` message plus an
/// optional follow-up user message carrying any images.
///
/// Mirrors the Python vLLM `_convert_user_tool_result`: text parts join with
/// `\n`; `tool_reference` entries inside the result are dropped (Python
/// re-emits them as a second tool message in a shape the internal chat layer
/// has no analog for).
fn convert_user_tool_result(
    tool_use_id: String,
    content: Option<ToolResultContent>,
    out: &mut Vec<VllmChatMessage>,
) {
    let mut image_urls: Vec<String> = Vec::new();
    let text = match content {
        None => String::new(),
        Some(ToolResultContent::Text(text)) => text,
        Some(ToolResultContent::Blocks(blocks)) => {
            let mut text_parts: Vec<String> = Vec::new();
            for block in blocks {
                match block {
                    AnthropicContentBlock::Text { text } => text_parts.push(text),
                    AnthropicContentBlock::Image { source } => {
                        let url = convert_image_source_to_url(&source);
                        if !url.is_empty() {
                            image_urls.push(url);
                        }
                    }
                    _ => {}
                }
            }
            text_parts.join("\n")
        }
    };

    out.push(VllmChatMessage::tool_response(text, tool_use_id));

    if !image_urls.is_empty() {
        out.push(VllmChatMessage::user(ChatContent::Parts(
            image_urls
                .into_iter()
                .map(|url| ChatContentPart::ImageUrl {
                    image_url: url,
                    detail: None,
                    uuid: None,
                })
                .collect(),
        )));
    }
}

/// Map Anthropic tool choice onto the internal enum
/// (`auto`→Auto, `any`→Required, `none`→None, `tool`→Function).
///
/// Mirrors the Python vLLM `_convert_tool_choice`: when tools are present
/// without an explicit choice, Python defaults to `auto`. The
/// `disable_parallel_tool_use` half of that method lives in
/// [`parallel_tool_calls`].
pub(super) fn convert_tool_choice(
    tool_choice: Option<&AnthropicToolChoice>,
    has_tools: bool,
) -> Option<ChatToolChoice> {
    match tool_choice {
        Some(AnthropicToolChoice::Auto { .. }) => Some(ChatToolChoice::Auto),
        Some(AnthropicToolChoice::Any { .. }) => Some(ChatToolChoice::Required),
        Some(AnthropicToolChoice::None) => Some(ChatToolChoice::None),
        Some(AnthropicToolChoice::Tool { name, .. }) => {
            Some(ChatToolChoice::Function { name: name.clone() })
        }
        None if has_tools => Some(ChatToolChoice::Auto),
        None => None,
    }
}

/// Convert Anthropic tool definitions.
///
/// Mirrors the Python vLLM `_convert_tools`, plus the protocol validator's
/// schema normalization: Python inserts `"type": "object"` into schemas that
/// omit it at validation time; mirrored here so rendering sees the same
/// schema.
pub(super) fn convert_tools(tools: Option<Vec<AnthropicTool>>) -> Vec<ChatTool> {
    tools
        .unwrap_or_default()
        .into_iter()
        .map(|tool| {
            let mut parameters = tool.input_schema;
            if let Value::Object(map) = &mut parameters
                && !map.contains_key("type")
            {
                map.insert("type".to_string(), Value::String("object".to_string()));
            }
            ChatTool {
                name: tool.name,
                description: tool.description,
                parameters,
                strict: tool.strict,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// Logic the oracle inlines in its larger methods, or decomposes differently;
// each doc comment names the Python method it lifts from.

/// Drop billing-header text.
///
/// Lifts the `startswith` checks the oracle inlines in
/// `_convert_system_message` and `_extract_system_text`.
fn clean_system_text(text: &str) -> Option<&str> {
    (!text.starts_with(BILLING_HEADER_PREFIX)).then_some(text)
}

/// Concatenate the text of block-form system content, skipping billing
/// headers and empty blocks.
///
/// Lifts the block filter/join loop inlined in the Python vLLM
/// `_convert_system_message`; Python joins with an empty separator.
fn system_text_from_blocks(blocks: &[SystemTextBlock]) -> String {
    blocks
        .iter()
        .filter(|block| block.block_type == "text" && !block.text.is_empty())
        .filter_map(|block| clean_system_text(&block.text))
        .collect()
}

/// Lower one user message.
///
/// Covers the user-role half of the Python vLLM `_convert_message_content` /
/// `_convert_block` pair (Python dispatches per block type; we split per role
/// instead). Tool results split out into `ToolResponse` messages (each
/// followed by one image-carrying user message when the result contains
/// images), emitted before the residual user content — matching the message
/// ordering the Python converter produces. A user message whose blocks were
/// fully consumed (or empty) is dropped, matching Python.
fn convert_user_message(
    content: MessageContent,
    out: &mut Vec<VllmChatMessage>,
) -> Result<(), ApiError> {
    let blocks = match content {
        MessageContent::Text(text) => {
            out.push(VllmChatMessage::user(text));
            return Ok(());
        }
        MessageContent::Blocks(blocks) => blocks,
    };

    let mut parts: Vec<ChatContentPart> = Vec::new();
    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                if !text.is_empty() {
                    parts.push(ChatContentPart::text(text));
                }
            }
            AnthropicContentBlock::Image { source } => parts.push(ChatContentPart::ImageUrl {
                image_url: convert_image_source_to_url(&source),
                detail: None,
                uuid: None,
            }),
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error: _,
            } => convert_user_tool_result(tool_use_id, content, out),
            // Accepted and dropped (opaque), matching Python.
            AnthropicContentBlock::RedactedThinking { .. } => {}
            // Standalone references are ignored, matching Python; references
            // inside tool_result content are handled there.
            AnthropicContentBlock::ToolReference { .. } => {}
            AnthropicContentBlock::Thinking { .. } | AnthropicContentBlock::ToolUse { .. } => {
                bail_invalid_request!(
                    "thinking and tool_use blocks are only valid in assistant messages."
                );
            }
        }
    }

    match parts.len() {
        0 => {} // fully consumed by tool results (or empty): drop, matching Python
        // Python collapses a single text part back to plain string content.
        1 if matches!(parts[0], ChatContentPart::Text { .. }) => {
            let Some(ChatContentPart::Text { text }) = parts.pop() else {
                unreachable!("checked above");
            };
            out.push(VllmChatMessage::user(ChatContent::Text(text)));
        }
        _ => out.push(VllmChatMessage::user(ChatContent::Parts(parts))),
    }
    Ok(())
}

/// Lower one assistant message, preserving thinking/text/tool_use block order
/// (Python concatenates all thinking into one `reasoning` field and appends
/// tool calls after content; we keep the original interleaving).
///
/// Covers the assistant-role half of the Python vLLM
/// `_convert_message_content` / `_convert_block` pair, including
/// `_convert_tool_use_block`'s lowering of tool calls.
fn convert_assistant_message(
    content: MessageContent,
    out: &mut Vec<VllmChatMessage>,
) -> Result<(), ApiError> {
    let blocks = match content {
        MessageContent::Text(text) => {
            out.push(VllmChatMessage::assistant_blocks(vec![
                AssistantContentBlock::Text { text },
            ]));
            return Ok(());
        }
        MessageContent::Blocks(blocks) => blocks,
    };

    let mut assistant_blocks: Vec<AssistantContentBlock> = Vec::new();
    for block in blocks {
        match block {
            AnthropicContentBlock::Thinking { thinking, .. } => {
                // Input signatures are accepted and dropped; vLLM cannot
                // verify them (#47753 open question covers the output side).
                assistant_blocks.push(AssistantContentBlock::Reasoning { text: thinking });
            }
            AnthropicContentBlock::Text { text } => {
                if !text.is_empty() {
                    assistant_blocks.push(AssistantContentBlock::Text { text });
                }
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                let arguments = serde_json::to_string(
                    &input.unwrap_or_else(|| Value::Object(Default::default())),
                )
                .expect("JSON value serializes");
                assistant_blocks.push(AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: id.unwrap_or_else(fabricate_tool_call_id),
                    name,
                    arguments,
                }));
            }
            // Assistant-side tool_result is nonsensical but tolerated by
            // Python, which renders it into the content via `str()` (Python
            // repr); we render block content as JSON instead.
            AnthropicContentBlock::ToolResult { content, .. } => {
                let rendered = match content {
                    None => String::new(),
                    Some(ToolResultContent::Text(text)) => text,
                    Some(ToolResultContent::Blocks(blocks)) => {
                        serde_json::to_string(&blocks).expect("blocks serialize")
                    }
                };
                assistant_blocks.push(AssistantContentBlock::Text {
                    text: format!("Tool result: {rendered}"),
                });
            }
            AnthropicContentBlock::RedactedThinking { .. }
            | AnthropicContentBlock::ToolReference { .. } => {}
            AnthropicContentBlock::Image { .. } => {
                bail_invalid_request!("image blocks are not supported in assistant messages.");
            }
        }
    }

    if assistant_blocks.is_empty() {
        bail_invalid_request!(
            "Assistant messages must contain text, reasoning content, or tool_calls."
        );
    }
    out.push(VllmChatMessage::assistant_blocks(assistant_blocks));
    Ok(())
}

/// Fabricate a tool-call ID for `tool_use` blocks that arrive without one.
///
/// Covers the fallback inlined in the Python vLLM `_convert_tool_use_block`:
/// Python uses `call_{int(time.time())}`, which collides when one response
/// carries several tool_use blocks (#35667); we keep the UUID form the Rust
/// stack already uses for generated tool calls.
fn fabricate_tool_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

/// Parallel tool use defaults to enabled; `disable_parallel_tool_use: true`
/// switches it off.
///
/// Lifts the `parallel_tool_calls` assignment inlined in the Python vLLM
/// `_convert_tool_choice` (`not disable_parallel_tool_use`).
pub(super) fn parallel_tool_calls(tool_choice: Option<&AnthropicToolChoice>) -> bool {
    let disable = match tool_choice {
        Some(
            AnthropicToolChoice::Auto {
                disable_parallel_tool_use,
            }
            | AnthropicToolChoice::Any {
                disable_parallel_tool_use,
            }
            | AnthropicToolChoice::Tool {
                disable_parallel_tool_use,
                ..
            },
        ) => disable_parallel_tool_use.unwrap_or(false),
        Some(AnthropicToolChoice::None) | None => false,
    };
    !disable
}

/// Lower one `count_tokens` body into a [`ChatRequest`] for template
/// rendering, following `tokenize`'s precedent: only fields that affect
/// rendering are set; sampling and decode options stay at default because
/// counting never generates.
///
/// Covers the request-preparation half of the Python vLLM `count_tokens`
/// route (the prompt-shaping subset of `_convert_anthropic_to_openai_request`).
pub(super) fn prepare_count_tokens_request(
    request: AnthropicCountTokensRequest,
    request_id: String,
    merge_inline_system: bool,
) -> Result<ChatRequest, ApiError> {
    let has_tools = request.tools.is_some();
    let parallel = parallel_tool_calls(request.tool_choice.as_ref());
    let tool_choice = convert_tool_choice(request.tool_choice.as_ref(), has_tools);
    let messages = convert_messages(request.system, request.messages, merge_inline_system)?;

    let tool_context = ResolvedToolContext::new(
        &messages,
        convert_tools(request.tools),
        tool_choice,
        parallel,
    )
    .map_err(|error| chat_submit_error("failed to resolve request tools", error))?;

    Ok(ChatRequest {
        request_id,
        messages,
        sampling_params: SamplingParams::default(),
        chat_options: ChatOptions {
            generation_prompt_mode: GenerationPromptMode::StartNewAssistant,
            chat_template: None,
            reasoning_effort: None,
            response_format: None,
            template_kwargs: request.chat_template_kwargs.unwrap_or_default(),
        },
        tool_context,
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        priority: 0,
        documents: None,
        cache_salt: None,
        add_special_tokens: false,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    })
}
