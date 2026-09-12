// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared DeepSeek V4 and V4.1 prompt rendering.
//!
//! Official Python reference:
//! <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py>

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Write as _;

use serde::Serialize;
use serde_json::{Map, Value};
use serde_json_fmt::JsonFormat;

use llm_multimodal::DEEPSEEK_V41_IMAGE_PLACEHOLDER;

use crate::error::{Error, Result};
use crate::request::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatTool, ReasoningEffort,
};
use crate::{AssistantContentBlock, AssistantMessageExt, AssistantToolCall};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";
const USER_SP_TOKEN: &str = "<｜User｜>";
const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const REASONING_EFFORT_HIGH: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);
const REASONING_EFFORT_MAX: &str = concat!(
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n",
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n",
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n",
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingMode {
    Chat,
    Thinking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DsDialect {
    V4,
    V41,
}

impl DsDialect {
    fn tool_calls_tag(self) -> &'static str {
        match self {
            Self::V4 => "tool_calls",
            Self::V41 => " calls",
        }
    }

    fn invoke_tag(self) -> &'static str {
        match self {
            Self::V4 => "invoke",
            Self::V41 => " invoke",
        }
    }

    fn parameter_tag(self) -> &'static str {
        match self {
            Self::V4 => "parameter",
            Self::V41 => " parameter",
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Serialize)]
struct RenderedToolSchema<'a> {
    name: &'a str,
    description: Option<&'a str>,
    parameters: &'a Value,
    strict: Option<bool>,
}

/// Render one chat request into the final prompt string.
pub(super) fn render_request(request: &ChatRequest, dialect: DsDialect) -> Result<String> {
    let (thinking_mode, reasoning_effort_prompt) = match dialect {
        DsDialect::V4 => {
            resolve_thinking_options(request).map(|(mode, prompt)| (mode, Cow::Borrowed(prompt)))?
        }
        DsDialect::V41 => resolve_v41_thinking_options(request)?,
    };
    let request_tools = request_tools(request);
    let synthetic_tool_system = needs_synthetic_tool_system(request, request_tools);
    let drop_thinking = request.parse_template_bool("drop_thinking")?.unwrap_or(true)
        && !rendered_tools_present(request, request_tools);
    let drop_historical_developers = thinking_mode == ThinkingMode::Thinking && drop_thinking;
    let last_user_like_message_index = request
        .messages
        .iter()
        .enumerate()
        .rfind(|(index, message)| is_user_like_entry(message, *index, dialect))
        .map(|(index, _)| index);
    let last_user_render_index = find_last_user_render_index(
        request.messages.as_slice(),
        synthetic_tool_system,
        drop_historical_developers,
        last_user_like_message_index,
        dialect,
    );
    let mut out = String::from(BOS_TOKEN);
    if dialect == DsDialect::V41
        && (thinking_mode == ThinkingMode::Thinking
            || synthetic_tool_system
            || matches!(request.messages.first(), Some(ChatMessage::System { .. })))
    {
        out.push_str(SYSTEM_SP_TOKEN);
    }
    if thinking_mode == ThinkingMode::Thinking {
        out.push_str(&reasoning_effort_prompt);
    }

    let mut request_tools_attached = false;
    let mut render_index = 0isize;
    if synthetic_tool_system {
        render_system_message(&mut out, None, request_tools, dialect)?;
        request_tools_attached = true;
        render_index += 1;
    }

    let mut last_tool_call_order = HashMap::new();
    for (message_index, message) in request.messages.iter().enumerate() {
        if is_following_user_content(request.messages.as_slice(), message_index) {
            continue;
        }

        if drop_historical_developers
            && is_historical_developer(message, message_index, last_user_like_message_index)
        {
            continue;
        }

        let current_render_index = render_index;
        render_index += 1;

        match message {
            ChatMessage::System { content } => {
                if dialect == DsDialect::V41 && current_render_index > 0 {
                    out.push_str(SYSTEM_SP_TOKEN);
                }
                let tools = if !request_tools_attached {
                    request_tools_attached = true;
                    request_tools
                } else {
                    &[]
                };
                render_system_message(&mut out, Some(content), tools, dialect)?;
            }
            ChatMessage::Developer { content, tools } => {
                render_developer_message(
                    &mut out,
                    content,
                    tools.as_deref().unwrap_or(&[]),
                    dialect,
                )?;
            }
            ChatMessage::User { .. } | ChatMessage::ToolResponse { .. } => {
                render_user_content_block(
                    &mut out,
                    request.messages.as_slice(),
                    message_index,
                    &last_tool_call_order,
                    dialect,
                )?;
            }
            ChatMessage::Assistant { content } => {
                // Mirror Python: thinking block (reasoning + </think>) is
                // emitted whenever thinking is active and reasoning isn't
                // dropped - i.e. drop_thinking is off OR this turn lies
                // strictly after the last user turn.
                let emit_thinking_block = thinking_mode == ThinkingMode::Thinking
                    && (!drop_thinking || current_render_index > last_user_render_index);
                let append_eos = !(message_index + 1 == request.messages.len()
                    && request.chat_options.continue_final_message());
                render_assistant_message(
                    &mut out,
                    emit_thinking_block,
                    append_eos,
                    content,
                    dialect,
                )?;

                if content.has_tool_calls() {
                    last_tool_call_order.clear();
                    last_tool_call_order.extend(
                        content
                            .tool_calls()
                            .enumerate()
                            .map(|(index, tool_call)| (tool_call.id.clone(), index)),
                    );
                }
            }
        }

        if (is_user_like_entry(message, current_render_index as usize, dialect)
            || (dialect == DsDialect::V4 && matches!(message, ChatMessage::System { .. })))
            && next_rendered_entry_is_assistant_or_end(
                request.messages.as_slice(),
                message_index,
                drop_historical_developers,
                last_user_like_message_index,
            )
        {
            write_assistant_transition(
                &mut out,
                thinking_mode,
                drop_thinking,
                current_render_index >= last_user_render_index,
            );
        }
    }

    Ok(out)
}

/// Resolve DeepSeek V4's thinking controls. Unlike the Python tokenizer
/// wrapper, the Rust renderer only consumes the typed top-level
/// `reasoning_effort`; the generic template-kwargs map is left for HF
/// templates.
fn resolve_thinking_options(request: &ChatRequest) -> Result<(ThinkingMode, &'static str)> {
    let mut thinking_mode = match request.enable_thinking()?.unwrap_or(true) {
        true => ThinkingMode::Thinking,
        false => ThinkingMode::Chat,
    };
    let mut reasoning_effort_prompt = REASONING_EFFORT_HIGH;

    match request.chat_options.reasoning_effort {
        Some(ReasoningEffort::None) => thinking_mode = ThinkingMode::Chat,
        Some(ReasoningEffort::Max) => {
            reasoning_effort_prompt = REASONING_EFFORT_MAX;
        }
        Some(ReasoningEffort::XHigh | ReasoningEffort::High) => {
            reasoning_effort_prompt = REASONING_EFFORT_HIGH;
        }
        Some(ReasoningEffort::Minimal | ReasoningEffort::Medium | ReasoningEffort::Low) => {
            reasoning_effort_prompt = "";
        }
        None => {}
    }

    Ok((thinking_mode, reasoning_effort_prompt))
}

/// Resolve V4.1's numeric reasoning effort using the top-level value before template kwargs.
fn resolve_v41_thinking_options(
    request: &ChatRequest,
) -> Result<(ThinkingMode, Cow<'static, str>)> {
    let mut thinking = request.enable_thinking()?.unwrap_or(true);
    let budget = match request.chat_options.reasoning_effort {
        Some(ReasoningEffort::None) => {
            thinking = false;
            50
        }
        Some(ReasoningEffort::Low) => 25,
        Some(ReasoningEffort::High) => 50,
        Some(ReasoningEffort::XHigh) => 75,
        Some(ReasoningEffort::Max) => 100,
        Some(ReasoningEffort::Minimal | ReasoningEffort::Medium) => {
            return Err(invalid_v41_reasoning_effort());
        }
        None => match request.chat_options.template_kwargs.get("reasoning_effort") {
            None | Some(Value::Null) => 50,
            Some(Value::String(effort)) => match effort.as_str() {
                "low" => 25,
                "high" => 50,
                "xhigh" => 75,
                "max" => 100,
                _ => return Err(invalid_v41_reasoning_effort()),
            },
            Some(Value::Number(budget)) => budget
                .as_u64()
                .filter(|budget| (1..=100).contains(budget))
                .ok_or_else(invalid_v41_reasoning_effort)?,
            Some(_) => return Err(invalid_v41_reasoning_effort()),
        },
    };
    if thinking {
        Ok((
            ThinkingMode::Thinking,
            Cow::Owned(format!(
                "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
            )),
        ))
    } else {
        Ok((ThinkingMode::Chat, Cow::Borrowed("")))
    }
}

fn invalid_v41_reasoning_effort() -> Error {
    Error::InvalidReasoningEffort(
        "DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, or an integer within [1, 100] in chat_template_kwargs".to_string(),
    )
}

/// Return request-level tools only when native tool parsing is enabled.
fn request_tools(request: &ChatRequest) -> &[ChatTool] {
    if request.tool_parsing_enabled() {
        request.initial_tools()
    } else {
        &[]
    }
}

/// Return whether request tools need a synthetic leading system entry.
fn needs_synthetic_tool_system(request: &ChatRequest, request_tools: &[ChatTool]) -> bool {
    !request_tools.is_empty()
        && !request
            .messages
            .iter()
            .any(|message| matches!(message, ChatMessage::System { .. }))
}

/// Return whether any rendered message carries tool schemas.
fn rendered_tools_present(request: &ChatRequest, request_tools: &[ChatTool]) -> bool {
    !request_tools.is_empty()
        || request.messages.iter().any(|message| {
            matches!(
                message,
                ChatMessage::Developer {
                    tools: Some(tools),
                    ..
                } if !tools.is_empty()
            )
        })
}

/// Find the last user-like turn after inline tool-response merging.
fn find_last_user_render_index(
    messages: &[ChatMessage],
    synthetic_tool_system: bool,
    drop_historical_developers: bool,
    last_user_like_message_index: Option<usize>,
    dialect: DsDialect,
) -> isize {
    let mut render_index = isize::from(synthetic_tool_system);
    let mut last_user_index = -1;

    for (message_index, message) in messages.iter().enumerate() {
        if is_following_user_content(messages, message_index)
            || (drop_historical_developers
                && is_historical_developer(message, message_index, last_user_like_message_index))
        {
            continue;
        }

        if is_user_like_entry(message, render_index as usize, dialect) {
            last_user_index = render_index;
        }
        render_index += 1;
    }

    last_user_index
}

/// Return whether this message is already covered by a previous user-content
/// entry.
fn is_following_user_content(messages: &[ChatMessage], message_index: usize) -> bool {
    is_user_content_entry(&messages[message_index])
        && message_index > 0
        && is_user_content_entry(&messages[message_index - 1])
}

/// Return whether one message contributes content to a V4 user turn.
fn is_user_content_entry(message: &ChatMessage) -> bool {
    matches!(
        message,
        ChatMessage::User { .. } | ChatMessage::ToolResponse { .. }
    )
}

/// Return whether one rendered entry should be treated as user-like.
fn is_user_like_entry(message: &ChatMessage, message_index: usize, dialect: DsDialect) -> bool {
    matches!(
        message,
        ChatMessage::Developer { .. } | ChatMessage::User { .. } | ChatMessage::ToolResponse { .. }
    ) || (dialect == DsDialect::V41
        && message_index > 0
        && matches!(message, ChatMessage::System { .. }))
}

/// Return whether a developer entry precedes another user-like turn.
fn is_historical_developer(
    message: &ChatMessage,
    message_index: usize,
    last_user_like_message_index: Option<usize>,
) -> bool {
    matches!(message, ChatMessage::Developer { .. })
        && last_user_like_message_index.is_some_and(|last_index| message_index < last_index)
}

/// Return whether the next rendered entry is assistant, or there is no next
/// entry.
fn next_rendered_entry_is_assistant_or_end(
    messages: &[ChatMessage],
    message_index: usize,
    drop_historical_developers: bool,
    last_user_like_message_index: Option<usize>,
) -> bool {
    let mut next_index = message_index + 1;
    while next_index < messages.len()
        && (is_following_user_content(messages, next_index)
            || (drop_historical_developers
                && is_historical_developer(
                    &messages[next_index],
                    next_index,
                    last_user_like_message_index,
                )))
    {
        next_index += 1;
    }

    messages
        .get(next_index)
        .map(|message| matches!(message, ChatMessage::Assistant { .. }))
        .unwrap_or(true)
}

/// Render the tool preamble shown to the model for one DeepSeek dialect.
fn render_tools(out: &mut String, tools: &[ChatTool], dialect: DsDialect) -> Result<()> {
    let tool_calls_tag = dialect.tool_calls_tag();
    let invoke_tag = dialect.invoke_tag();
    let parameter_tag = dialect.parameter_tag();
    write!(
        out,
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜{tool_calls_tag}>" block like the following:

<｜DSML｜{tool_calls_tag}>
<｜DSML｜{invoke_tag} name="$TOOL_NAME">
<｜DSML｜{parameter_tag} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜{parameter_tag}>
...
</｜DSML｜{invoke_tag}>
<｜DSML｜{invoke_tag} name="$TOOL_NAME2">
...
</｜DSML｜{invoke_tag}>
</｜DSML｜{tool_calls_tag}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"#,
    )
    .expect("writing to String cannot fail");

    for (index, tool) in tools.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        render_tool_schema(out, tool)?;
    }

    out.push_str(
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n",
    );
    Ok(())
}

/// Serialize one typed tool schema into the JSON shape embedded in the prompt.
fn render_tool_schema(out: &mut String, tool: &ChatTool) -> Result<()> {
    out.push_str(&json_dumps(&RenderedToolSchema {
        name: &tool.name,
        description: tool.description.as_deref(),
        parameters: &tool.parameters,
        strict: tool.strict,
    })?);
    Ok(())
}

/// Render a system turn, optionally followed by the V4 tool preamble.
fn render_system_message(
    out: &mut String,
    content: Option<&ChatContent>,
    tools: &[ChatTool],
    dialect: DsDialect,
) -> Result<()> {
    if let Some(content) = content {
        write_chat_content(out, content, dialect)?;
    }
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools, dialect)?;
    }
    Ok(())
}

/// Developer messages are rendered as user-like turns with optional tools.
fn render_developer_message(
    out: &mut String,
    content: &ChatContent,
    tools: &[ChatTool],
    dialect: DsDialect,
) -> Result<()> {
    if content.is_empty() {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek developer message: empty content".to_string(),
        ));
    }

    out.push_str(USER_SP_TOKEN);
    write_chat_content(out, content, dialect)?;
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools, dialect)?;
    }
    Ok(())
}

/// Render contiguous user and tool-response messages as one V4 user turn.
fn render_user_content_block(
    out: &mut String,
    messages: &[ChatMessage],
    message_index: usize,
    tool_call_order: &HashMap<String, usize>,
    dialect: DsDialect,
) -> Result<()> {
    let (block_start, block_end) = user_content_block_bounds(messages, message_index);
    let mut sorted_tool_indices =
        sorted_tool_response_indices(messages, block_start, block_end, tool_call_order).into_iter();

    out.push_str(USER_SP_TOKEN);
    for (offset, message_index) in (block_start..block_end).enumerate() {
        if offset > 0 {
            out.push_str("\n\n");
        }
        match &messages[message_index] {
            ChatMessage::User { content } => write_chat_content(out, content, dialect)?,
            ChatMessage::ToolResponse { .. } => {
                let sorted_index = sorted_tool_indices
                    .next()
                    .expect("tool response block should include this tool message");
                let ChatMessage::ToolResponse { content, .. } = &messages[sorted_index] else {
                    unreachable!("sorted tool response index should reference a tool message");
                };
                write_tool_result(out, content, dialect)?;
            }
            _ => unreachable!("user content block should only contain user content messages"),
        }
    }

    Ok(())
}

/// Return the contiguous user-content block containing `actual_index`.
fn user_content_block_bounds(messages: &[ChatMessage], actual_index: usize) -> (usize, usize) {
    let mut block_start = actual_index;
    while block_start > 0 && is_user_content_entry(&messages[block_start - 1]) {
        block_start -= 1;
    }

    let mut block_end = actual_index + 1;
    while block_end < messages.len() && is_user_content_entry(&messages[block_end]) {
        block_end += 1;
    }

    (block_start, block_end)
}

fn sorted_tool_response_indices(
    messages: &[ChatMessage],
    block_start: usize,
    block_end: usize,
    tool_call_order: &HashMap<String, usize>,
) -> Vec<usize> {
    let mut indices = (block_start..block_end)
        .filter(|index| matches!(messages[*index], ChatMessage::ToolResponse { .. }))
        .collect::<Vec<_>>();
    if indices.len() <= 1 || tool_call_order.is_empty() {
        return indices;
    }

    indices.sort_by_key(|index| {
        let ChatMessage::ToolResponse { tool_call_id, .. } = &messages[*index] else {
            unreachable!("tool response block should only contain tool messages");
        };
        tool_call_order.get(tool_call_id.as_str()).copied().unwrap_or(0)
    });
    indices
}

/// Render one tool response payload inside a V4 `<tool_result>` block.
fn write_tool_result(out: &mut String, content: &ChatContent, dialect: DsDialect) -> Result<()> {
    out.push_str("<tool_result>");
    write_chat_content(out, content, dialect)?;
    out.push_str("</tool_result>");
    Ok(())
}

/// Append the assistant transition token after a user-like or system turn.
fn write_assistant_transition(
    out: &mut String,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    opens_thinking: bool,
) {
    out.push_str(ASSISTANT_SP_TOKEN);
    if thinking_mode == ThinkingMode::Thinking && (!drop_thinking || opens_thinking) {
        out.push_str(THINKING_START_TOKEN);
    } else {
        out.push_str(THINKING_END_TOKEN);
    }
}

/// Render one assistant turn, including optional reasoning, DSML tool calls,
/// and the trailing EOS marker.
fn render_assistant_message(
    out: &mut String,
    emit_thinking_block: bool,
    append_eos: bool,
    content: &[AssistantContentBlock],
    dialect: DsDialect,
) -> Result<()> {
    let has_tool_calls = content.has_tool_calls();

    if emit_thinking_block {
        if content.has_reasoning() {
            write_assistant_reasoning(out, content);
        }
        out.push_str(THINKING_END_TOKEN);
    }

    write_assistant_text(out, content);

    if has_tool_calls {
        let tool_calls_tag = dialect.tool_calls_tag();
        writeln!(out, "\n\n<{DSML_TOKEN}{tool_calls_tag}>").expect("writing to String cannot fail");
        for (index, tool_call) in content.tool_calls().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            render_tool_call(out, tool_call, dialect)?;
        }
        write!(out, "\n</{DSML_TOKEN}{tool_calls_tag}>").expect("writing to String cannot fail");
    }

    if append_eos {
        out.push_str(EOS_TOKEN);
    }
    Ok(())
}

/// Render one assistant tool call in DSML XML-like format.
fn render_tool_call(
    out: &mut String,
    tool_call: &AssistantToolCall,
    dialect: DsDialect,
) -> Result<()> {
    let invoke_tag = dialect.invoke_tag();
    writeln!(
        out,
        "<{DSML_TOKEN}{invoke_tag} name=\"{}\">",
        tool_call.name
    )
    .expect("writing to String cannot fail");
    encode_arguments_to_dsml(out, tool_call, dialect)?;
    write!(out, "\n</{DSML_TOKEN}{invoke_tag}>").expect("writing to String cannot fail");
    Ok(())
}

/// Convert one assistant tool-call arguments object into DSML parameter form.
///
/// String values are emitted raw with `string="true"`, while all other JSON
/// values are rendered with JSON syntax and `string="false"`.
fn encode_arguments_to_dsml(
    out: &mut String,
    tool_call: &AssistantToolCall,
    dialect: DsDialect,
) -> Result<()> {
    let parameter_tag = dialect.parameter_tag();
    // Match deepseek-recipe's render_tool_arguments for both V4 and V4.1:
    // https://github.com/deepseek-ai/deepseek-recipe/blob/8cadfede7063c896b944e7bae05daa3549ae97ea/deepseek-recipe-encoding/src/v4/mod.rs#L48-L58
    let arguments = serde_json::from_str::<Map<String, Value>>(&tool_call.arguments)
        .unwrap_or_else(|_| {
            Map::from_iter([(
                "arguments".to_owned(),
                Value::String(tool_call.arguments.clone()),
            )])
        });

    let mut wrote_parameter = false;
    for (key, value) in &arguments {
        if wrote_parameter {
            out.push('\n');
        }

        let is_string = matches!(value, Value::String(_));
        write!(
            out,
            "<{DSML_TOKEN}{parameter_tag} name=\"{key}\" string=\"{}\">",
            if is_string { "true" } else { "false" }
        )
        .expect("writing to String cannot fail");

        match value {
            Value::String(value) => out.push_str(value),
            value => out.push_str(&json_dumps(value)?),
        }

        write!(out, "</{DSML_TOKEN}{parameter_tag}>").expect("writing to String cannot fail");
        wrote_parameter = true;
    }

    Ok(())
}

/// Write chat content directly into the destination buffer without flattening
/// it into an intermediate `String`.
///
/// The V4.1 dialect inlines the image placeholder at each image part's
/// position unconditionally (matching the Python encoding's
/// `IMAGE_PLACEHOLDER`); other dialects reject multimodal parts.
fn write_chat_content(out: &mut String, content: &ChatContent, dialect: DsDialect) -> Result<()> {
    match content {
        ChatContent::Text(text) => out.push_str(text),
        ChatContent::Parts(parts) => {
            for (index, part) in parts.iter().enumerate() {
                if index > 0 && dialect == DsDialect::V41 {
                    out.push_str("\n\n");
                }
                match part {
                    ChatContentPart::ImageUrl { .. } if dialect == DsDialect::V41 => {
                        out.push_str(DEEPSEEK_V41_IMAGE_PLACEHOLDER);
                    }
                    _ => out.push_str(part.as_text()?),
                }
            }
        }
    }
    Ok(())
}

/// Write all reasoning blocks in encounter order.
fn write_assistant_reasoning(out: &mut String, content: &[AssistantContentBlock]) {
    for block in content {
        if let AssistantContentBlock::Reasoning { text } = block {
            out.push_str(text);
        }
    }
}

/// Write all visible assistant text blocks in encounter order.
fn write_assistant_text(out: &mut String, content: &[AssistantContentBlock]) {
    for block in content {
        if let AssistantContentBlock::Text { text } = block {
            out.push_str(text);
        }
    }
}

/// Compact JSON serialization used by this renderer for exact prompt text.
fn json_dumps<T: Serialize>(value: &T) -> Result<String> {
    JsonFormat::new()
        .comma(", ")
        .expect("literal comma separator is valid JSON")
        .colon(": ")
        .expect("literal colon separator is valid JSON")
        .ascii(false)
        .format_to_string(value)
        .map_err(|error| {
            Error::ChatTemplate(format!(
                "failed to serialize DeepSeek JSON payload: {error}"
            ))
        })
}
