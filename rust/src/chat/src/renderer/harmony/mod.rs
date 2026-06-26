//! Native Harmony chat renderer for `gpt_oss`.

pub(crate) mod encoding;

use openai_harmony::HarmonyEncoding;
use openai_harmony::chat::{
    Author, Conversation, DeveloperContent, Message, ReasoningEffort as HarmonyReasoningEffort,
    Role, SystemContent, ToolDescription,
};
use thiserror_ext::AsReport as _;
use time::macros::format_description;
use vllm_text::Prompt;

use self::encoding::harmony_encoding;
use super::{ChatRenderer, RenderedPrompt, request_template_kwargs};
use crate::error::{Error, Result};
use crate::event::AssistantContentBlock;
use crate::request::{ChatContent, ChatMessage, ChatRequest, ChatTool, GenerationPromptMode};
use crate::{AssistantMessageExt as _, ReasoningEffort};

const SYSTEM_START_DATE_ENV: &str = "VLLM_SYSTEM_START_DATE";
const HARMONY_SYSTEM_INSTRUCTIONS_ENV: &str = "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS";

/// GPT-OSS renderer backed by the official Harmony encoding.
pub struct HarmonyChatRenderer {
    encoding: &'static HarmonyEncoding,
    options: Options,
}

struct Options {
    system_start_date: String,
    use_system_instructions: bool,
}

impl HarmonyChatRenderer {
    /// Create a new Harmony renderer with options loaded from the environment.
    ///
    /// - `VLLM_SYSTEM_START_DATE`: If set, this date will be used as the system start date.
    /// - `VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS`: If set to a non-zero integer, system
    ///   instructions will be included in the prompt.
    pub fn new() -> Result<Self> {
        Self::with_options(
            env_system_start_date(),
            env_use_harmony_system_instructions(),
        )
    }

    /// Create a new Harmony renderer with the given options.
    pub fn with_options(
        system_start_date: impl Into<String>,
        use_system_instructions: bool,
    ) -> Result<Self> {
        Ok(Self {
            encoding: harmony_encoding()?,
            options: Options {
                system_start_date: system_start_date.into(),
                use_system_instructions,
            },
        })
    }

    fn render_token_ids(&self, request: &ChatRequest) -> Result<Vec<u32>> {
        if request.has_multimodal() {
            return Err(Error::UnsupportedMultimodalContent("image_url"));
        }
        if matches!(
            request.chat_options.generation_prompt_mode,
            GenerationPromptMode::ContinueFinalAssistant
        ) {
            return Err(Error::ChatTemplate(
                "Harmony renderer does not support continue_final_message".to_string(),
            ));
        }

        let messages = auto_drop_analysis_messages(to_harmony_messages(request, &self.options)?);
        let conversation = Conversation::from_messages(messages);
        // Pass `None` so oss-harmony does not apply its narrower built-in
        // analysis-drop policy after the Rust-side Python-parity cleanup above.
        let token_ids = match request.chat_options.generation_prompt_mode {
            GenerationPromptMode::StartNewAssistant => self
                .encoding
                .render_conversation_for_completion(&conversation, Role::Assistant, None),
            GenerationPromptMode::NoGenerationPrompt => {
                self.encoding.render_conversation(&conversation, None)
            }
            GenerationPromptMode::ContinueFinalAssistant => unreachable!("checked above"),
        }
        .map_err(|error| {
            Error::ChatTemplate(format!(
                "failed to render Harmony prompt: {}",
                error.as_report()
            ))
        })?;

        Ok(token_ids)
    }
}

impl ChatRenderer for HarmonyChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        Ok(RenderedPrompt {
            prompt: Prompt::TokenIds(self.render_token_ids(request)?),
            effective_template_kwargs: request_template_kwargs(request),
        })
    }
}

fn to_harmony_messages(request: &ChatRequest, options: &Options) -> Result<Vec<Message>> {
    let (instructions, leading_developer_tools, remaining_messages) =
        peel_leading_instructions(&request.messages)?;
    let tool_call_names = tool_call_names(&request.messages);
    let mut messages =
        build_harmony_preamble(request, instructions, leading_developer_tools, options)?;

    for message in remaining_messages {
        messages.extend(to_harmony_message(message, &tool_call_names, options)?);
    }

    Ok(messages)
}

#[allow(clippy::type_complexity)]
fn peel_leading_instructions(
    messages: &[ChatMessage],
) -> Result<(Option<String>, Option<&[ChatTool]>, &[ChatMessage])> {
    let Some(first) = messages.first() else {
        return Ok((None, None, messages));
    };

    match first {
        ChatMessage::System { content } => Ok((Some(flatten_text(content)?), None, &messages[1..])),
        ChatMessage::Developer { content, tools } => Ok((
            Some(flatten_text(content)?),
            tools.as_deref(),
            &messages[1..],
        )),
        ChatMessage::User { .. }
        | ChatMessage::Assistant { .. }
        | ChatMessage::ToolResponse { .. } => Ok((None, None, messages)),
    }
}

fn build_harmony_preamble(
    request: &ChatRequest,
    instructions: Option<String>,
    leading_developer_tools: Option<&[ChatTool]>,
    options: &Options,
) -> Result<Vec<Message>> {
    let mut messages = vec![Message::from_role_and_content(
        Role::System,
        system_content(
            instructions.as_deref().filter(|_| options.use_system_instructions),
            request.chat_options.reasoning_effort,
            &options.system_start_date,
        )?,
    )];

    let mut developer = DeveloperContent::new();
    let mut has_developer_content = false;

    if !options.use_system_instructions
        && let Some(instructions) = instructions.as_deref().filter(|text| !text.is_empty())
    {
        developer = developer.with_instructions(instructions);
        has_developer_content = true;
    }

    let tool_descriptions = preamble_tool_descriptions(request, leading_developer_tools);
    if !tool_descriptions.is_empty() {
        developer = developer.with_function_tools(tool_descriptions);
        has_developer_content = true;
    }

    if has_developer_content {
        messages.push(Message::from_role_and_content(Role::Developer, developer));
    }

    Ok(messages)
}

fn preamble_tool_descriptions(
    request: &ChatRequest,
    leading_developer_tools: Option<&[ChatTool]>,
) -> Vec<ToolDescription> {
    let mut tools = Vec::new();
    if request.tool_parsing_enabled() {
        tools.extend(to_tool_descriptions(&request.tools));
    }
    if let Some(leading_developer_tools) = leading_developer_tools {
        tools.extend(to_tool_descriptions(leading_developer_tools));
    }
    tools
}

fn system_content(
    instructions: Option<&str>,
    reasoning_effort: Option<ReasoningEffort>,
    system_start_date: &str,
) -> Result<SystemContent> {
    let mut content =
        SystemContent::new().with_conversation_start_date(system_start_date.to_string());

    if let Some(reasoning_effort) = reasoning_effort {
        content = content.with_reasoning_effort(to_harmony_reasoning_effort(reasoning_effort)?);
    }

    if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
        let model_identity = match content.model_identity.as_deref() {
            Some(identity) if !identity.is_empty() => format!("{identity}\n{instructions}"),
            _ => instructions.to_string(),
        };
        content = content.with_model_identity(model_identity);
    }

    Ok(content)
}

fn to_harmony_message(
    message: &ChatMessage,
    tool_call_names: &std::collections::HashMap<String, String>,
    options: &Options,
) -> Result<Vec<Message>> {
    Ok(match message {
        ChatMessage::System { content } => {
            let instructions = flatten_text(content)?;
            vec![system_or_developer_message(
                "system",
                instructions,
                None,
                options,
            )?]
        }
        ChatMessage::Developer { content, tools } => {
            let instructions = flatten_text(content)?;
            vec![developer_message(Some(instructions), tools.as_deref())]
        }
        ChatMessage::User { content } => {
            vec![Message::from_role_and_content(
                Role::User,
                flatten_text(content)?,
            )]
        }
        ChatMessage::Assistant { content } => assistant_messages(content),
        ChatMessage::ToolResponse {
            content,
            tool_call_id,
        } => {
            let name = tool_call_names.get(tool_call_id).ok_or_else(|| {
                Error::ChatTemplate(format!(
                    "invalid Harmony tool message: unknown tool_call_id `{tool_call_id}`"
                ))
            })?;
            vec![
                Message::from_author_and_content(
                    Author::new(Role::Tool, format!("functions.{name}")),
                    flatten_text(content)?,
                )
                .with_channel("commentary")
                .with_recipient("assistant"),
            ]
        }
    })
}

fn system_or_developer_message(
    role: &str,
    instructions: String,
    tools: Option<&[ChatTool]>,
    options: &Options,
) -> Result<Message> {
    if role == "system" && options.use_system_instructions {
        return Ok(Message::from_role_and_content(
            Role::System,
            system_content(Some(&instructions), None, &options.system_start_date)?,
        ));
    }

    Ok(developer_message(Some(instructions), tools))
}

fn developer_message(instructions: Option<String>, tools: Option<&[ChatTool]>) -> Message {
    let mut content = DeveloperContent::new();
    if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
        content = content.with_instructions(instructions);
    }
    if let Some(tools) = tools {
        let tools = to_tool_descriptions(tools);
        if !tools.is_empty() {
            content = content.with_function_tools(tools);
        }
    }
    Message::from_role_and_content(Role::Developer, content)
}

fn assistant_messages(content: &[AssistantContentBlock]) -> Vec<Message> {
    let mut messages = Vec::new();
    let has_tool_calls = content.has_tool_calls();

    if has_tool_calls {
        let text = content.text();
        if !text.is_empty() {
            messages.push(
                Message::from_role_and_content(Role::Assistant, text).with_channel("commentary"),
            );
        }
    }

    if let Some(reasoning) = content.reasoning() {
        messages.push(
            Message::from_role_and_content(Role::Assistant, reasoning).with_channel("analysis"),
        );
    }

    if has_tool_calls {
        for tool_call in content.tool_calls() {
            messages.push(
                Message::from_role_and_content(Role::Assistant, tool_call.arguments.clone())
                    .with_channel("commentary")
                    .with_recipient(format!("functions.{}", tool_call.name))
                    .with_content_type("<|constrain|>json"),
            );
        }
    } else {
        let text = content.text();
        if !text.is_empty() {
            messages
                .push(Message::from_role_and_content(Role::Assistant, text).with_channel("final"));
        }
    }

    messages
}

fn tool_call_names(messages: &[ChatMessage]) -> std::collections::HashMap<String, String> {
    let mut names = std::collections::HashMap::new();
    for message in messages {
        let ChatMessage::Assistant { content } = message else {
            continue;
        };
        for tool_call in content.tool_calls() {
            names.insert(tool_call.id.clone(), tool_call.name.clone());
        }
    }
    names
}

fn auto_drop_analysis_messages(messages: Vec<Message>) -> Vec<Message> {
    // Match vLLM Python's Harmony cleanup: once an assistant final message exists,
    // previous assistant analysis messages are stale chain-of-thought and should
    // be removed. oss-harmony can also drop analysis with `Some(Default::default())`,
    // but that built-in path only triggers when the last assistant message is final
    // and drops relative to the first final message, which misses longer multi-turn
    // histories with later user/tool turns.
    let Some(last_assistant_final_index) = messages.iter().rposition(|message| {
        message.author.role == Role::Assistant && message.channel.as_deref() == Some("final")
    }) else {
        return messages;
    };

    messages
        .into_iter()
        .enumerate()
        .filter_map(|(index, message)| {
            (index >= last_assistant_final_index || message.channel.as_deref() != Some("analysis"))
                .then_some(message)
        })
        .collect()
}

fn flatten_text(content: &ChatContent) -> Result<String> {
    content.try_flatten_to_text()
}

fn to_tool_descriptions(tools: &[ChatTool]) -> Vec<ToolDescription> {
    tools
        .iter()
        .map(|tool| {
            ToolDescription::new(
                tool.name.clone(),
                tool.description.clone().unwrap_or_default(),
                Some(tool.parameters.clone()),
            )
        })
        .collect()
}

fn to_harmony_reasoning_effort(
    reasoning_effort: ReasoningEffort,
) -> Result<HarmonyReasoningEffort> {
    match reasoning_effort {
        ReasoningEffort::Low => Ok(HarmonyReasoningEffort::Low),
        ReasoningEffort::Medium => Ok(HarmonyReasoningEffort::Medium),
        ReasoningEffort::High => Ok(HarmonyReasoningEffort::High),
        ReasoningEffort::None
        | ReasoningEffort::Minimal
        | ReasoningEffort::XHigh
        | ReasoningEffort::Max => Err(Error::ChatTemplate(format!(
            "reasoning_effort={:?} is not supported by Harmony. Supported values are: low, medium, high.",
            reasoning_effort.as_str()
        ))),
    }
}

fn env_system_start_date() -> String {
    std::env::var(SYSTEM_START_DATE_ENV)
        .ok()
        .filter(|date| !date.is_empty())
        .unwrap_or_else(current_date)
}

fn current_date() -> String {
    const DATE_FORMAT: &[time::format_description::FormatItem<'static>] =
        format_description!("[year]-[month]-[day]");
    let now = time::OffsetDateTime::now_local().unwrap_or_else(|_| time::OffsetDateTime::now_utc());
    now.format(DATE_FORMAT).expect("static date format should be valid")
}

fn env_use_harmony_system_instructions() -> bool {
    std::env::var(HARMONY_SYSTEM_INSTRUCTIONS_ENV)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .is_some_and(|value| value != 0)
}

#[cfg(test)]
mod tests;
