// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::{Value, json};
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage,
    ChatOptions, ChatRequest, ChatTool, ChatToolChoice, EffortValue, GenerationPromptMode,
    ResolvedToolContext, SamplingParams,
};

use super::types::{
    AssistantContent, AssistantInputBlock, ClientTool, ClientToolKind, CountTokensRequest,
    ImageSource, InputMessage, MessagesInput, MessagesRequest, OutputConfig, OutputEffort,
    OutputFormat, TextBlock, TextContent, ThinkingConfig, ToolChoice, ToolChoiceKind,
    ToolResultBlock, ToolResultContent, UserContent, UserInputBlock,
};
use super::validate::{
    validate_count_tokens_request, validate_messages_input, validate_messages_request,
};
use crate::error::{ApiError, chat_submit_error, invalid_request};
use crate::lora::LoraModelResolution;
use crate::utils::ResolvedRequestContext;

pub(super) fn prepare_messages_request(
    body: MessagesRequest,
    model: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<ChatRequest, ApiError> {
    validate_messages_request(&body)?;
    let effort = body.output_config.and_then(|config| config.effort);
    let mut request = prepare_chat_request(body.input, effort, model, ctx)?;
    request.sampling_params.max_tokens = Some(body.max_tokens);
    request.sampling_params.temperature = body.temperature;
    request.sampling_params.top_p = body.top_p;
    request.sampling_params.top_k = body.top_k;
    request.decode_options.stop_strings = body.stop_sequences;
    request.intermediate = body.stream == Some(true);
    request.cache_salt = body.cache_salt;
    Ok(request)
}

pub(super) fn prepare_count_tokens_request(
    body: CountTokensRequest,
    model: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<ChatRequest, ApiError> {
    validate_count_tokens_request(&body)?;
    let OutputConfig { effort, format } = body.output_config.unwrap_or_default();
    let mut request = prepare_chat_request(body.input, effort, model, ctx)?;
    request.chat_options.template_kwargs = body.chat_template_kwargs.unwrap_or_default();
    if let Some(OutputFormat::JsonSchema { schema }) = format {
        request.chat_options.response_format = Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "json_schema", "schema": schema},
        }));
    }
    Ok(request)
}

fn prepare_chat_request(
    mut body: MessagesInput,
    effort: Option<OutputEffort>,
    model: &LoraModelResolution,
    ctx: ResolvedRequestContext,
) -> Result<ChatRequest, ApiError> {
    body.strip_system_billing_headers();
    validate_messages_input(&body, model)?;
    let (reasoning_effort, thinking_token_budget) = resolve_thinking(body.thinking, effort);
    let mut messages = Vec::new();
    if let Some(system) = body.system {
        let text = match system {
            TextContent::Text(text) => text,
            TextContent::Blocks(blocks) => {
                blocks.into_iter().map(|TextBlock::Text { text, .. }| text).collect()
            }
        };
        if !text.is_empty() {
            messages.push(ChatMessage::system(text));
        }
    }
    for message in body.messages {
        append_message(&mut messages, message);
    }
    let tool_context = resolve_tool_context(&messages, body.tools, body.tool_choice)?;
    let continuation = matches!(messages.last(), Some(ChatMessage::Assistant { content })
        if content.iter().any(|b| matches!(b, AssistantContentBlock::Text { text } if !text.is_empty())));
    Ok(ChatRequest {
        request_id: format!("msg_{}", ctx.request_id),
        messages,
        sampling_params: SamplingParams {
            thinking_token_budget,
            ..Default::default()
        },
        chat_options: ChatOptions {
            reasoning_effort,
            generation_prompt_mode: if continuation {
                GenerationPromptMode::ContinueFinalAssistant
            } else {
                GenerationPromptMode::StartNewAssistant
            },
            ..Default::default()
        },
        tool_context,
        decode_options: Default::default(),
        intermediate: false,
        prompt_truncation: None,
        priority: ctx.priority.unwrap_or(0),
        documents: None,
        cache_salt: None,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id: ctx.session_id,
        lora_request: model.lora_request.clone(),
        return_prompt_text: false,
    })
}

fn resolve_thinking(
    thinking: Option<ThinkingConfig>,
    effort: Option<OutputEffort>,
) -> (Option<EffortValue>, Option<i64>) {
    let reasoning_effort = effort.map(EffortValue::from);
    match thinking {
        Some(ThinkingConfig::Disabled | ThinkingConfig::BetweenTools) => {
            // Like Python, between-tools thinking starts with thinking disabled.
            (Some(EffortValue::from("none")), None)
        }
        Some(ThinkingConfig::Enabled { budget_tokens, .. }) => (reasoning_effort, budget_tokens),
        Some(ThinkingConfig::Adaptive { .. }) | None => (reasoning_effort, None),
    }
}

fn resolve_tool_context(
    messages: &[ChatMessage],
    tools: Option<Vec<ClientTool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<ResolvedToolContext, ApiError> {
    let tools = tools
        .unwrap_or_default()
        .into_iter()
        .map(ChatTool::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    let (tool_choice, parallel) = match tool_choice {
        Some(choice) => (
            Some(match choice.kind {
                ToolChoiceKind::Auto => ChatToolChoice::Auto,
                ToolChoiceKind::Any => ChatToolChoice::Required,
                ToolChoiceKind::None => ChatToolChoice::None,
                ToolChoiceKind::Tool { name } => ChatToolChoice::Function { name },
            }),
            !choice.disable_parallel_tool_use,
        ),
        None => (None, true),
    };
    ResolvedToolContext::new(messages, tools, tool_choice, parallel)
        .map_err(|error| chat_submit_error("invalid Messages tools", error))
}

impl MessagesInput {
    fn strip_system_billing_headers(&mut self) {
        if let Some(system @ TextContent::Blocks(_)) = &mut self.system {
            system.strip_billing_headers();
        }
        self.messages.retain_mut(|message| match message {
            InputMessage::System { content } => content.strip_billing_headers(),
            _ => true,
        });
    }
}

fn append_message(messages: &mut Vec<ChatMessage>, message: InputMessage) {
    match message {
        // The renderer handles inline instructions without moving them to the front.
        InputMessage::System { content } => {
            messages.push(ChatMessage::developer(content.into_content(), None));
        }
        InputMessage::Assistant { content } => messages.push(match content {
            AssistantContent::Text(text) => ChatMessage::assistant_text(text),
            AssistantContent::Blocks(blocks) => ChatMessage::assistant_blocks(
                blocks
                    .into_iter()
                    .filter_map(|block| match block {
                        AssistantInputBlock::Text { text } => {
                            Some(AssistantContentBlock::Text { text })
                        }
                        AssistantInputBlock::Thinking { thinking, .. } => {
                            Some(AssistantContentBlock::Reasoning { text: thinking })
                        }
                        AssistantInputBlock::ToolUse { id, name, input } => {
                            Some(AssistantContentBlock::ToolCall(AssistantToolCall {
                                id,
                                name,
                                arguments: Value::Object(input).to_string(),
                            }))
                        }
                        AssistantInputBlock::RedactedThinking { .. } => None,
                    })
                    .collect(),
            ),
        }),
        InputMessage::User {
            content: UserContent::Text(text),
        } => {
            if !text.is_empty() {
                messages.push(ChatMessage::user(text));
            }
        }
        InputMessage::User {
            content: UserContent::Blocks(blocks),
        } => append_user_blocks(messages, blocks),
    }
}

fn append_user_blocks(messages: &mut Vec<ChatMessage>, blocks: Vec<UserInputBlock>) {
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            UserInputBlock::Text { text } if !text.is_empty() => {
                parts.push(ChatContentPart::text(text));
            }
            UserInputBlock::Text { .. } => {}
            UserInputBlock::Image { source } => parts.push(source.into_part()),
            UserInputBlock::ToolResult {
                tool_use_id,
                content,
            } => {
                if !parts.is_empty() {
                    messages.push(ChatMessage::user(parts_to_content(std::mem::take(
                        &mut parts,
                    ))));
                }
                append_tool_result(messages, tool_use_id, content);
            }
        }
    }
    if !parts.is_empty() {
        messages.push(ChatMessage::user(parts_to_content(parts)));
    }
}

fn parts_to_content(mut parts: Vec<ChatContentPart>) -> ChatContent {
    if matches!(parts.as_slice(), [ChatContentPart::Text { .. }]) {
        let ChatContentPart::Text { text } = parts.pop().unwrap() else {
            unreachable!()
        };
        ChatContent::Text(text)
    } else {
        ChatContent::Parts(parts)
    }
}

impl ImageSource {
    fn into_part(self) -> ChatContentPart {
        ChatContentPart::image_url(match self {
            Self::Url { url } => url,
            Self::Base64 { media_type, data } => {
                format!("data:{};base64,{data}", media_type.as_str())
            }
        })
    }
}

fn append_tool_result(
    messages: &mut Vec<ChatMessage>,
    id: String,
    content: Option<ToolResultContent>,
) {
    let mut images = Vec::new();
    let content = match content {
        None => ChatContent::Text(String::new()),
        Some(ToolResultContent::Text(text)) => ChatContent::Text(text),
        Some(ToolResultContent::Blocks(blocks)) => {
            let mut text = Vec::new();
            let mut references = Vec::new();
            for block in blocks {
                match block {
                    ToolResultBlock::Text { text: part } => text.push(part),
                    ToolResultBlock::Image { source } => images.push(source.into_part()),
                    ToolResultBlock::ToolReference { tool_name } => {
                        references.push(ChatContentPart::tool_reference(tool_name))
                    }
                }
            }
            let text = text.join("\n");
            if references.is_empty() {
                ChatContent::Text(text)
            } else {
                if !text.is_empty() {
                    references.push(ChatContentPart::text(text));
                }
                ChatContent::Parts(references)
            }
        }
    };
    messages.push(ChatMessage::tool_response(content, id));
    // Match Python's rendering of image-bearing results for templates whose tool role is text-only.
    if !images.is_empty() {
        messages.push(ChatMessage::user(ChatContent::Parts(images)));
    }
}

impl TryFrom<ClientTool> for ChatTool {
    type Error = ApiError;

    fn try_from(tool: ClientTool) -> Result<Self, ApiError> {
        if matches!(tool.kind, Some(ClientToolKind::Unsupported)) {
            return Err(invalid_request!(
                param = "tools",
                "Only client tools are supported"
            ));
        }
        if tool.name.is_empty()
            || tool.name.len() > 128
            || !tool.name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err(invalid_request!(param = "tools", "Invalid tool name"));
        }
        let mut input_schema = tool
            .input_schema
            .ok_or_else(|| invalid_request!(param = "tools", "Tool input_schema is required"))?;
        let schema_type = input_schema.entry("type").or_insert_with(|| Value::from("object"));
        if schema_type.as_str() != Some("object") {
            return Err(invalid_request!(
                param = "tools",
                "Tool input schemas must have type object"
            ));
        }
        Ok(Self {
            name: tool.name,
            description: tool.description,
            parameters: Value::Object(input_schema),
            strict: tool.strict,
            defer_loading: tool.defer_loading,
        })
    }
}

impl From<OutputEffort> for EffortValue {
    fn from(effort: OutputEffort) -> Self {
        Self::from(match effort {
            OutputEffort::Low => "low",
            OutputEffort::Medium => "medium",
            OutputEffort::High => "high",
            OutputEffort::XHigh => "xhigh",
            OutputEffort::Max => "max",
        })
    }
}

impl TextContent {
    /// Filter system text blocks and return whether to keep the content.
    fn strip_billing_headers(&mut self) -> bool {
        fn keep_system_text(text: &str) -> bool {
            !text.is_empty() && !text.starts_with("x-anthropic-billing-header")
        }

        match self {
            Self::Text(text) => keep_system_text(text),
            Self::Blocks(blocks) => {
                blocks.retain(|TextBlock::Text { text, .. }| keep_system_text(text));
                !blocks.is_empty()
            }
        }
    }

    fn into_content(self) -> ChatContent {
        match self {
            Self::Text(text) => ChatContent::Text(text),
            Self::Blocks(blocks) => parts_to_content(
                blocks
                    .into_iter()
                    .map(|TextBlock::Text { text, .. }| ChatContentPart::text(text))
                    .collect(),
            ),
        }
    }
}
