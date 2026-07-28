// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::event::{AssistantContentBlock, AssistantToolCall};
use crate::request::{
    ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatTool, ChatToolChoice,
    GenerationPromptMode, ReasoningEffort,
};

/// Options for constructing a [`ChatRequest`] from a fixture file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FixtureRequestOptions {
    /// Whether to set the template kwarg `[enable_]thinking=true`.
    pub enable_thinking: bool,
    /// Whether fixtures ending in an assistant message should omit the
    /// trailing generation prompt.
    pub no_generation_prompt_when_last_assistant: bool,
}

/// Read a fixture file from the given path and convert it into a [`ChatRequest`]
/// using the provided options.
pub(crate) fn fixture_chat_request(path: &Path, options: FixtureRequestOptions) -> ChatRequest {
    let fixture = fs::read_to_string(path).unwrap();
    let fixture: FixtureFile = serde_json::from_str(&fixture).unwrap();
    fixture.into_request().into_chat_request(options)
}

/// Fixture file format for chat-renderer tests.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum FixtureFile {
    WithRequest(FixtureRequest),
    MessagesOnly(Vec<FixtureMessage>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct FixtureRequest {
    #[serde(default)]
    tools: Vec<FixtureTool>,
    messages: Vec<FixtureMessage>,
    add_generation_prompt: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
}

impl FixtureFile {
    fn into_request(self) -> FixtureRequest {
        match self {
            Self::WithRequest(request) => request,
            Self::MessagesOnly(messages) => FixtureRequest {
                tools: Vec::new(),
                messages,
                add_generation_prompt: None,
                reasoning_effort: None,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub(crate) enum FixtureMessage {
    System {
        content: FixtureContent,
    },
    Developer {
        content: FixtureContent,
        #[serde(default)]
        tools: Vec<FixtureTool>,
    },
    User {
        content: FixtureContent,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(default)]
        reasoning_content: String,
        #[serde(default)]
        tool_calls: Vec<FixtureToolCall>,
    },
    Tool {
        content: FixtureContent,
        #[serde(default)]
        tool_call_id: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum FixtureContent {
    Text(String),
    Parts(Vec<FixtureContentPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum FixtureContentPart {
    Text { text: String },
    ImageUrl { image_url: String },
    InputAudio { data: String },
    AudioUrl { audio_url: String },
}

#[derive(Debug, Deserialize)]
pub(crate) struct FixtureTool {
    function: FixtureToolFunction,
}

#[derive(Debug, Deserialize)]
struct FixtureToolFunction {
    name: String,
    description: Option<String>,
    parameters: Value,
    #[serde(default)]
    strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FixtureToolCall {
    #[serde(default)]
    id: Option<String>,
    function: FixtureToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct FixtureToolCallFunction {
    name: String,
    arguments: String,
}

impl FixtureRequest {
    fn into_chat_request(self, options: FixtureRequestOptions) -> ChatRequest {
        let mut request = ChatRequest {
            request_id: "renderer-fixture".to_string(),
            messages: self
                .messages
                .into_iter()
                .enumerate()
                .map(|(index, message)| fixture_message_to_chat_message(index, message))
                .collect(),
            tools: to_chat_tools(&self.tools),
            tool_choice: if self.tools.is_empty() {
                ChatToolChoice::None
            } else {
                ChatToolChoice::Auto
            },
            ..ChatRequest::for_test()
        };

        if options.no_generation_prompt_when_last_assistant
            && matches!(request.messages.last(), Some(ChatMessage::Assistant { .. }))
        {
            request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
        }
        if self.add_generation_prompt == Some(false) {
            request.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
        }
        request.chat_options.reasoning_effort = self.reasoning_effort;
        if options.enable_thinking {
            for key in ["thinking", "enable_thinking"] {
                request.chat_options.template_kwargs.insert(key.to_string(), Value::Bool(true));
            }
        }

        request
    }
}

fn fixture_message_to_chat_message(index: usize, message: FixtureMessage) -> ChatMessage {
    match message {
        FixtureMessage::System { content } => ChatMessage::system(to_chat_content(content)),
        FixtureMessage::Developer { content, tools } => ChatMessage::developer(
            to_chat_content(content),
            (!tools.is_empty()).then(|| to_chat_tools(&tools)),
        ),
        FixtureMessage::User { content } => ChatMessage::user(to_chat_content(content)),
        FixtureMessage::Assistant {
            content,
            reasoning_content,
            tool_calls,
        } => {
            let mut blocks = Vec::new();
            if !reasoning_content.is_empty() {
                blocks.push(AssistantContentBlock::Reasoning {
                    text: reasoning_content,
                });
            }
            if !content.is_empty() {
                blocks.push(AssistantContentBlock::Text { text: content });
            }
            blocks.extend(
                tool_calls.into_iter().enumerate().map(|(tool_index, tool_call)| {
                    AssistantContentBlock::ToolCall(AssistantToolCall {
                        id: tool_call
                            .id
                            .unwrap_or_else(|| format!("fixture-tool-call-{index}-{tool_index}")),
                        name: tool_call.function.name,
                        arguments: tool_call.function.arguments,
                    })
                }),
            );
            ChatMessage::assistant_blocks(blocks)
        }
        FixtureMessage::Tool {
            content,
            tool_call_id,
        } => ChatMessage::tool_response(
            to_chat_content(content),
            tool_call_id.unwrap_or_else(|| format!("fixture-tool-response-{index}")),
        ),
    }
}

fn to_chat_content(content: FixtureContent) -> ChatContent {
    match content {
        FixtureContent::Text(text) => ChatContent::Text(text),
        FixtureContent::Parts(parts) => ChatContent::Parts(
            parts
                .into_iter()
                .map(|part| match part {
                    FixtureContentPart::Text { text } => ChatContentPart::text(text),
                    FixtureContentPart::ImageUrl { image_url } => {
                        ChatContentPart::image_url(image_url)
                    }
                    FixtureContentPart::InputAudio { data } => ChatContentPart::InputAudio {
                        data,
                        format: None,
                        uuid: None,
                    },
                    FixtureContentPart::AudioUrl { audio_url } => ChatContentPart::AudioUrl {
                        audio_url,
                        uuid: None,
                    },
                })
                .collect(),
        ),
    }
}

fn to_chat_tools(tools: &[FixtureTool]) -> Vec<ChatTool> {
    tools
        .iter()
        .map(|tool| ChatTool {
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}
