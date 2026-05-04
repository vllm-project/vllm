//! DeepSeek V4 prompt renderer.
//!
//! Original Python implementation:
//! <https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/deepseek_v4_encoding.py>

use std::collections::HashMap;
use std::fmt::Write as _;

use serde::Serialize;
use serde_json::Value;
use serde_json_fmt::JsonFormat;

use crate::error::{Error, Result};
use crate::request::{ChatContent, ChatMessage, ChatRequest, ChatTool, ReasoningEffort};
use crate::{AssistantContentBlock, AssistantMessageExt, AssistantToolCall};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const REASONING_EFFORT_MAX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingMode {
    Chat,
    Thinking,
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
pub(super) fn render_request(request: &ChatRequest) -> Result<String> {
    let (thinking_mode, max_reasoning_effort) = resolve_thinking_options(request)?;
    let request_tools = request_tools(request);
    let synthetic_tool_system = needs_synthetic_tool_system(request, request_tools);
    let drop_thinking = request
        .parse_template_bool("drop_thinking")?
        .unwrap_or(true)
        && !rendered_tools_present(request, request_tools);
    let last_user_render_index =
        find_last_user_render_index(request.messages.as_slice(), synthetic_tool_system);
    let mut out = String::from(BOS_TOKEN);
    if thinking_mode == ThinkingMode::Thinking && max_reasoning_effort {
        out.push_str(REASONING_EFFORT_MAX);
    }

    let mut request_tools_attached = false;
    let mut render_index = 0isize;
    if synthetic_tool_system {
        render_system_message(&mut out, None, request_tools)?;
        request_tools_attached = true;
        render_index += 1;
    }

    for (message_index, message) in request.messages.iter().enumerate() {
        if is_following_tool_response(request.messages.as_slice(), message_index) {
            continue;
        }

        let current_render_index = render_index;
        render_index += 1;

        match message {
            ChatMessage::System { content } => {
                let tools = if !request_tools_attached {
                    request_tools_attached = true;
                    request_tools
                } else {
                    &[]
                };
                render_system_message(&mut out, Some(content), tools)?;
            }
            ChatMessage::Developer { content, tools } => {
                render_developer_message(&mut out, content, tools.as_deref().unwrap_or(&[]))?;
            }
            ChatMessage::User { content } => render_user_message(&mut out, content)?,
            ChatMessage::Assistant { content } => {
                // Mirror Python: thinking block (reasoning + </think>) is
                // emitted whenever thinking is active and reasoning isn't
                // dropped - i.e. drop_thinking is off OR this turn lies
                // strictly after the last user turn.
                let emit_thinking_block = thinking_mode == ThinkingMode::Thinking
                    && (!drop_thinking || current_render_index > last_user_render_index);
                let append_eos = !(message_index + 1 == request.messages.len()
                    && request.chat_options.continue_final_message());
                render_assistant_message(&mut out, emit_thinking_block, append_eos, content)?;
            }
            ChatMessage::ToolResponse { .. } => {
                render_tool_response_block(&mut out, request.messages.as_slice(), message_index)?;
            }
        }

        if is_user_like_entry(message)
            && next_rendered_entry_is_assistant_or_end(request.messages.as_slice(), message_index)
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

/// Resolve DeepSeek V4's thinking controls. Unlike the Python tokenizer wrapper,
/// the Rust renderer only consumes the typed top-level `reasoning_effort`; the
/// generic template-kwargs map is left for HF templates.
fn resolve_thinking_options(request: &ChatRequest) -> Result<(ThinkingMode, bool)> {
    let mut thinking_mode = match request.enable_thinking()?.unwrap_or(false) {
        true => ThinkingMode::Thinking,
        false => ThinkingMode::Chat,
    };
    let mut max_reasoning_effort = false;

    match request.chat_options.reasoning_effort {
        Some(ReasoningEffort::None) => thinking_mode = ThinkingMode::Chat,
        Some(ReasoningEffort::Max | ReasoningEffort::XHigh) => max_reasoning_effort = true,
        Some(_) | None => {}
    }

    Ok((thinking_mode, max_reasoning_effort))
}

/// Return request-level tools only when native tool parsing is enabled.
fn request_tools(request: &ChatRequest) -> &[ChatTool] {
    if request.tool_parsing_enabled() {
        request.tools.as_slice()
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
fn find_last_user_render_index(messages: &[ChatMessage], synthetic_tool_system: bool) -> isize {
    let mut render_index = isize::from(synthetic_tool_system);
    let mut last_user_index = -1;

    for (message_index, message) in messages.iter().enumerate() {
        if is_following_tool_response(messages, message_index) {
            continue;
        }

        if is_user_like_entry(message) {
            last_user_index = render_index;
        }
        render_index += 1;
    }

    last_user_index
}

/// Return whether this tool message is already covered by a previous tool run.
fn is_following_tool_response(messages: &[ChatMessage], message_index: usize) -> bool {
    matches!(messages[message_index], ChatMessage::ToolResponse { .. })
        && message_index > 0
        && matches!(
            messages[message_index - 1],
            ChatMessage::ToolResponse { .. }
        )
}

/// Return whether one rendered entry should be treated as user-like.
fn is_user_like_entry(message: &ChatMessage) -> bool {
    matches!(
        message,
        ChatMessage::Developer { .. } | ChatMessage::User { .. } | ChatMessage::ToolResponse { .. }
    )
}

/// Return whether the next rendered entry is assistant, or there is no next entry.
fn next_rendered_entry_is_assistant_or_end(messages: &[ChatMessage], message_index: usize) -> bool {
    let mut next_index = message_index + 1;
    if matches!(messages[message_index], ChatMessage::ToolResponse { .. }) {
        while next_index < messages.len()
            && matches!(messages[next_index], ChatMessage::ToolResponse { .. })
        {
            next_index += 1;
        }
    }

    messages
        .get(next_index)
        .map(|message| matches!(message, ChatMessage::Assistant { .. }))
        .unwrap_or(true)
}

/// Render the tool preamble shown to the model, V4 flavor.
fn render_tools(out: &mut String, tools: &[ChatTool]) -> Result<()> {
    out.push_str(
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"#,
    );

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
) -> Result<()> {
    if let Some(content) = content {
        write_chat_content(out, content)?;
    }
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools)?;
    }
    Ok(())
}

/// Developer messages are rendered as user-like turns with optional tools.
fn render_developer_message(
    out: &mut String,
    content: &ChatContent,
    tools: &[ChatTool],
) -> Result<()> {
    if content.is_empty() {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V4 developer message: empty content".to_string(),
        ));
    }

    out.push_str(USER_SP_TOKEN);
    write_chat_content(out, content)?;
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools)?;
    }
    Ok(())
}

/// Render one plain user turn.
fn render_user_message(out: &mut String, content: &ChatContent) -> Result<()> {
    out.push_str(USER_SP_TOKEN);
    write_chat_content(out, content)?;
    Ok(())
}

/// Render a contiguous tool-response run as one synthetic user turn.
fn render_tool_response_block(
    out: &mut String,
    messages: &[ChatMessage],
    message_index: usize,
) -> Result<()> {
    let (block_start, block_end) = tool_response_block_bounds(messages, message_index);
    let sorted_indices = sorted_tool_response_indices(messages, block_start, block_end);

    out.push_str(USER_SP_TOKEN);
    for (offset, message_index) in sorted_indices.iter().enumerate() {
        if offset > 0 {
            out.push_str("\n\n");
        }
        let ChatMessage::ToolResponse { content, .. } = &messages[*message_index] else {
            unreachable!("tool response block should only contain tool messages");
        };
        write_tool_result(out, content)?;
    }

    Ok(())
}

/// Return the contiguous tool-response block containing `actual_index`.
fn tool_response_block_bounds(messages: &[ChatMessage], actual_index: usize) -> (usize, usize) {
    let mut block_start = actual_index;
    while block_start > 0 && matches!(messages[block_start - 1], ChatMessage::ToolResponse { .. }) {
        block_start -= 1;
    }

    let mut block_end = actual_index + 1;
    while block_end < messages.len()
        && matches!(messages[block_end], ChatMessage::ToolResponse { .. })
    {
        block_end += 1;
    }

    (block_start, block_end)
}

fn sorted_tool_response_indices(
    messages: &[ChatMessage],
    block_start: usize,
    block_end: usize,
) -> Vec<usize> {
    let Some(tool_call_order) = last_tool_call_order_before(messages, block_start) else {
        return (block_start..block_end).collect();
    };

    let mut indices = (block_start..block_end).collect::<Vec<_>>();
    indices.sort_by_key(|index| {
        let ChatMessage::ToolResponse { tool_call_id, .. } = &messages[*index] else {
            unreachable!("tool response block should only contain tool messages");
        };
        tool_call_order
            .get(tool_call_id.as_str())
            .copied()
            .unwrap_or(0)
    });
    indices
}

fn last_tool_call_order_before(
    messages: &[ChatMessage],
    message_index: usize,
) -> Option<HashMap<&str, usize>> {
    let mut tool_call_order = None;
    for message in &messages[..message_index] {
        if let ChatMessage::Assistant { content } = message {
            let order = content
                .tool_calls()
                .enumerate()
                .map(|(index, tool_call)| (tool_call.id.as_str(), index))
                .collect::<HashMap<_, _>>();
            if !order.is_empty() {
                tool_call_order = Some(order);
            }
        }
    }
    tool_call_order
}

/// Render one tool response payload inside a V4 `<tool_result>` block.
fn write_tool_result(out: &mut String, content: &ChatContent) -> Result<()> {
    out.push_str("<tool_result>");
    write_chat_content(out, content)?;
    out.push_str("</tool_result>");
    Ok(())
}

/// Append the assistant transition token after a user-like turn.
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
        out.push_str("\n\n<｜DSML｜tool_calls>\n");
        for (index, tool_call) in content.tool_calls().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            render_tool_call(out, tool_call)?;
        }
        out.push_str("\n</｜DSML｜tool_calls>");
    }

    if append_eos {
        out.push_str(EOS_TOKEN);
    }
    Ok(())
}

/// Render one assistant tool call in DSML XML-like format.
fn render_tool_call(out: &mut String, tool_call: &AssistantToolCall) -> Result<()> {
    writeln!(out, "<{DSML_TOKEN}invoke name=\"{}\">", tool_call.name)
        .expect("writing to String cannot fail");
    encode_arguments_to_dsml(out, tool_call)?;
    write!(out, "\n</{DSML_TOKEN}invoke>").expect("writing to String cannot fail");
    Ok(())
}

/// Convert one assistant tool-call arguments object into DSML parameter form.
///
/// String values are emitted raw with `string="true"`, while all other JSON
/// values are rendered with JSON syntax and `string="false"`.
fn encode_arguments_to_dsml(out: &mut String, tool_call: &AssistantToolCall) -> Result<()> {
    let arguments: Value = serde_json::from_str(&tool_call.arguments).map_err(|error| {
        Error::ChatTemplate(format!(
            "assistant tool call has invalid JSON arguments for DeepSeek V4: {error}"
        ))
    })?;
    let Some(arguments) = arguments.as_object() else {
        return Err(Error::ChatTemplate(
            "assistant tool call arguments for DeepSeek V4 must be a JSON object".to_string(),
        ));
    };

    let mut wrote_parameter = false;
    for (key, value) in arguments {
        if wrote_parameter {
            out.push('\n');
        }

        let is_string = matches!(value, Value::String(_));
        write!(
            out,
            "<{DSML_TOKEN}parameter name=\"{key}\" string=\"{}\">",
            if is_string { "true" } else { "false" }
        )
        .expect("writing to String cannot fail");

        match value {
            Value::String(value) => out.push_str(value),
            value => out.push_str(&json_dumps(value)?),
        }

        write!(out, "</{DSML_TOKEN}parameter>").expect("writing to String cannot fail");
        wrote_parameter = true;
    }

    Ok(())
}

/// Write chat content directly into the destination buffer without flattening
/// it into an intermediate `String`.
fn write_chat_content(out: &mut String, content: &ChatContent) -> Result<()> {
    match content {
        ChatContent::Text(text) => out.push_str(text),
        ChatContent::Parts(parts) => {
            for part in parts {
                out.push_str(part.as_text());
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
                "failed to serialize DeepSeek V4 JSON payload: {error}"
            ))
        })
}
