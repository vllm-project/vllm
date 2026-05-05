//! DeepSeek V3.2 prompt renderer.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use serde::Serialize;
use serde_json::Value;
use serde_json_fmt::JsonFormat;

use crate::error::{Error, Result};
use crate::request::{ChatContent, ChatMessage, ChatRequest, ChatRole, ChatTool};
use crate::{AssistantContentBlock, AssistantMessageExt, AssistantToolCall};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";

/// DeepSeek uses `"chat"` vs `"thinking"` mode names. Keep the split explicit
/// here so the render branches stay easy to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingMode {
    Chat,
    Thinking,
}

/// Tool schema shape rendered inside the `<functions>` block.
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
    let thinking_mode = match request.enable_thinking()?.unwrap_or(false) {
        true => ThinkingMode::Thinking,
        false => ThinkingMode::Chat,
    };
    let drop_thinking = matches!(
        request.messages.last().map(ChatMessage::role),
        Some(ChatRole::User | ChatRole::Developer)
    );
    let render_offset = isize::from(request.tool_parsing_enabled());
    let last_user_render_index =
        find_last_user_render_index(request.messages.as_slice(), render_offset);
    let last_user_actual_index = find_last_user_actual_index(request.messages.as_slice());
    let mut prompt = String::from(BOS_TOKEN);

    if request.tool_parsing_enabled() {
        render_system_message(&mut prompt, None, &request.tools)?;
    }

    for (message_index, message) in request.messages.iter().enumerate() {
        render_message(
            &mut prompt,
            request.messages.as_slice(),
            message_index,
            message,
            render_offset,
            last_user_render_index,
            last_user_actual_index,
            thinking_mode,
            drop_thinking,
        )?;
    }

    Ok(prompt)
}

/// Find the last user-like turn in render order.
///
/// `render_offset` is `1` when a synthetic tool-only system turn is rendered
/// before the real request messages, and `0` otherwise.
fn find_last_user_render_index(messages: &[ChatMessage], render_offset: isize) -> isize {
    messages
        .iter()
        .rposition(|message| matches!(message.role(), ChatRole::User | ChatRole::Developer))
        .map(|index| index as isize + render_offset)
        .unwrap_or(-1)
}

/// Render one real request message, using `render_offset` to account for any
/// synthetic tool-only system turn that was already emitted before the loop.
fn render_message(
    out: &mut String,
    messages: &[ChatMessage],
    message_index: usize,
    message: &ChatMessage,
    render_offset: isize,
    last_user_render_index: isize,
    last_user_actual_index: usize,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
) -> Result<()> {
    let render_index = message_index as isize + render_offset;
    let opens_thinking = render_index == last_user_render_index;
    let after_last_user_turn = render_index > last_user_render_index;
    let after_or_at_last_user_turn = render_index >= last_user_render_index;

    match message {
        ChatMessage::System { content } => render_system_message(out, Some(content), &[]),
        ChatMessage::Developer { content, tools } => render_developer_message(
            out,
            content,
            tools.as_deref().unwrap_or(&[]),
            thinking_mode == ThinkingMode::Thinking && opens_thinking,
        ),
        ChatMessage::User { content } => render_user_message(
            out,
            content,
            thinking_mode == ThinkingMode::Thinking && opens_thinking,
        ),
        ChatMessage::Assistant { content } => render_assistant_message(
            out,
            thinking_mode == ThinkingMode::Thinking && after_last_user_turn,
            content,
            should_keep_assistant_reasoning(
                message_index,
                last_user_actual_index,
                thinking_mode,
                drop_thinking,
            ),
            // TODO: Respect `continue_final_message` and map it to DeepSeek's
            // prefix-style final-assistant continuation behavior.
            false,
        ),
        ChatMessage::ToolResponse { content, .. } => render_tool_message(
            out,
            messages,
            message_index,
            thinking_mode == ThinkingMode::Thinking && after_or_at_last_user_turn,
            content,
        ),
    }
}

/// Historical assistant reasoning is dropped in thinking mode when the final
/// request turn is a new user-like message.
fn should_keep_assistant_reasoning(
    actual_index: usize,
    last_user_actual_index: usize,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
) -> bool {
    !(thinking_mode == ThinkingMode::Thinking
        && drop_thinking
        && actual_index < last_user_actual_index)
}

/// Return the last user/developer turn in the real request message list.
fn find_last_user_actual_index(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .rposition(|message| matches!(message.role(), ChatRole::User | ChatRole::Developer))
        .unwrap_or(usize::MAX)
}

/// Render a system turn, optionally followed by the tool preamble.
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

/// Developer messages are wrapped into the same user-like turn shape as real
/// user messages, but can also carry message-local tools.
fn render_developer_message(
    out: &mut String,
    content: &ChatContent,
    tools: &[ChatTool],
    opens_thinking: bool,
) -> Result<()> {
    if content.is_empty() {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 developer message: empty content".to_string(),
        ));
    }

    out.push_str("<｜User｜>");
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools)?;
    }
    out.push_str("\n\n# The user's message is: ");
    write_chat_content(out, content)?;
    write_user_like_suffix(out, opens_thinking);
    Ok(())
}

/// Plain user turns share the same wrapper shape as developer turns without the
/// developer-specific preamble.
fn render_user_message(
    out: &mut String,
    content: &ChatContent,
    opens_thinking: bool,
) -> Result<()> {
    out.push_str("<｜User｜>");
    write_chat_content(out, content)?;
    write_user_like_suffix(out, opens_thinking);
    Ok(())
}

/// Shared trailing wrapper used by both real user turns and native developer
/// turns after their content has already been written.
// TODO: respect `add_generation_prompt` option
fn write_user_like_suffix(out: &mut String, opens_thinking: bool) {
    out.push_str("<｜Assistant｜>");
    if opens_thinking {
        out.push_str(THINKING_START_TOKEN);
    } else {
        out.push_str(THINKING_END_TOKEN);
    }
}

/// Render one tool result turn and decide whether it opens or closes the shared
/// `<function_results>` block for the preceding assistant tool-call message.
fn render_tool_message(
    out: &mut String,
    messages: &[ChatMessage],
    message_index: usize,
    resumes_thinking: bool,
    _content: &ChatContent,
) -> Result<()> {
    let (block_start, block_end) = tool_response_block_bounds(messages, message_index);
    if message_index != block_start {
        return Ok(());
    }

    let Some(prev_assistant_idx) = previous_assistant_actual_index(messages, block_start) else {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: missing previous assistant message".to_string(),
        ));
    };

    let ChatMessage::Assistant {
        content: assistant_content,
    } = &messages[prev_assistant_idx]
    else {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: previous non-tool message is not assistant"
                .to_string(),
        ));
    };

    let assistant_tool_calls = assistant_content.tool_calls().collect::<Vec<_>>();
    if assistant_tool_calls.is_empty() {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool message: previous assistant message has no tool calls"
                .to_string(),
        ));
    }

    let mut expected_tool_call_ids = HashSet::with_capacity(assistant_tool_calls.len());
    for tool_call in &assistant_tool_calls {
        if !expected_tool_call_ids.insert(tool_call.id.as_str()) {
            return Err(Error::ChatTemplate(
                "invalid DeepSeek V3.2 assistant tool calls: duplicate tool_call_id".to_string(),
            ));
        }
    }

    let mut tool_results_by_id = HashMap::with_capacity(assistant_tool_calls.len());
    for message in &messages[block_start..block_end] {
        let ChatMessage::ToolResponse {
            content,
            tool_call_id,
        } = message
        else {
            unreachable!("tool response block should only contain tool messages");
        };

        if !expected_tool_call_ids.contains(tool_call_id.as_str()) {
            return Err(Error::ChatTemplate(format!(
                "invalid DeepSeek V3.2 tool message: unknown tool_call_id `{tool_call_id}`"
            )));
        }

        if tool_results_by_id.insert(tool_call_id.as_str(), content).is_some() {
            return Err(Error::ChatTemplate(format!(
                "invalid DeepSeek V3.2 tool message: duplicate tool_call_id `{tool_call_id}`"
            )));
        }
    }

    if tool_results_by_id.len() != assistant_tool_calls.len() {
        return Err(Error::ChatTemplate(
            "invalid DeepSeek V3.2 tool messages: missing tool result for assistant tool call"
                .to_string(),
        ));
    }

    out.push_str("\n\n<function_results>");
    for tool_call in assistant_tool_calls {
        let content = tool_results_by_id
            .get(tool_call.id.as_str())
            .expect("validated tool_call_id set should be complete");
        out.push_str("\n<result>");
        write_chat_content(out, content)?;
        out.push_str("</result>");
    }

    out.push_str("\n</function_results>");
    out.push_str("\n\n");
    if resumes_thinking {
        out.push_str(THINKING_START_TOKEN);
    } else {
        out.push_str(THINKING_END_TOKEN);
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

/// Return the most recent assistant turn before `actual_index`.
fn previous_assistant_actual_index(messages: &[ChatMessage], actual_index: usize) -> Option<usize> {
    messages[..actual_index]
        .iter()
        .rposition(|message| matches!(message, ChatMessage::Assistant { .. }))
}

/// Render one assistant turn, including optional reasoning, DSML tool calls,
/// and the trailing EOS marker.
fn render_assistant_message(
    out: &mut String,
    after_last_user_turn: bool,
    content: &[AssistantContentBlock],
    keep_reasoning: bool,
    prefix: bool,
) -> Result<()> {
    let has_reasoning = keep_reasoning && content.has_reasoning();
    let has_tool_calls = content.has_tool_calls();

    if !has_tool_calls && prefix {
        write_assistant_text(out, content);
        return Ok(());
    }

    if after_last_user_turn {
        if !has_reasoning && !has_tool_calls {
            return Err(Error::ChatTemplate(
                "invalid DeepSeek V3.2 assistant message after last user message: expected reasoning or tool calls"
                    .to_string(),
            ));
        }

        if has_reasoning {
            write_assistant_reasoning(out, content);
        }
        out.push_str(THINKING_END_TOKEN);
    }

    write_assistant_text(out, content);

    if has_tool_calls {
        out.push_str("\n\n<｜DSML｜function_calls>\n");
        for (index, tool_call) in content.tool_calls().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            render_tool_call(out, tool_call)?;
        }
        out.push_str("\n</｜DSML｜function_calls>");
    }

    out.push_str(EOS_TOKEN);
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
            "assistant tool call has invalid JSON arguments for DeepSeek V3.2: {error}"
        ))
    })?;
    let Some(arguments) = arguments.as_object() else {
        return Err(Error::ChatTemplate(
            "assistant tool call arguments for DeepSeek V3.2 must be a JSON object".to_string(),
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

/// Render the full tool preamble shown to the model.
fn render_tools(out: &mut String, tools: &[ChatTool]) -> Result<()> {
    out.push_str(
        r#"## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a "<｜DSML｜function_calls>" block like the following as part of your reply to the user:
<｜DSML｜function_calls>
<｜DSML｜invoke name="$FUNCTION_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$FUNCTION_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The "string" attribute should be set to "true" for string type parameters and "false" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<｜DSML｜function_calls>
...
</｜DSML｜function_calls>

<function_results>
...
</function_results>

<think>...thinking about results</think>

Here are the functions available in JSONSchema format:
<functions>
"#,
    );

    for (index, tool) in tools.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        render_tool_schema(out, tool)?;
    }

    out.push_str("\n</functions>\n");
    Ok(())
}

/// Serialize one typed tool schema into the JSON shape embedded inside
/// `<functions>`.
fn render_tool_schema(out: &mut String, tool: &ChatTool) -> Result<()> {
    out.push_str(&json_dumps(&RenderedToolSchema {
        name: &tool.name,
        description: tool.description.as_deref(),
        parameters: &tool.parameters,
        strict: tool.strict,
    })?);
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
                "failed to serialize DeepSeek V3.2 JSON payload: {error}"
            ))
        })
}
