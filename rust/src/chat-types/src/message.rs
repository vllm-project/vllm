// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};

use crate::{AssistantContentBlock, AssistantMessage, AssistantMessageExt as _, ChatContent, Tool};

/// Role label for one chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    /// System instructions.
    System,
    /// Developer instructions.
    Developer,
    /// User input.
    User,
    /// Assistant history.
    Assistant,
    /// Result of an assistant tool call.
    ToolResponse,
}

impl ChatRole {
    /// Return the role string exposed to chat templates.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::ToolResponse => "tool_response",
        }
    }
}

/// One chat message.
///
/// Original Python API reference:
/// <https://github.com/vllm-project/vllm/blob/bc2c0c86efb28e77677a3cfb8687e976914a313a/vllm/entrypoints/chat_utils.py#L309-L333>
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatMessage {
    /// System message.
    System {
        /// Message content.
        content: ChatContent,
    },
    /// Developer message with optional message-local tools.
    Developer {
        /// Message content.
        content: ChatContent,
        /// Tools introduced by this developer message.
        tools: Option<Vec<Tool>>,
    },
    /// User message.
    User {
        /// Message content.
        content: ChatContent,
    },
    /// Assistant history assembled from structured blocks.
    Assistant {
        /// Structured assistant content.
        content: Vec<AssistantContentBlock>,
    },
    /// Tool response associated with one prior assistant tool call.
    ToolResponse {
        /// Tool response content.
        content: ChatContent,
        /// Identifier of the assistant tool call being answered.
        tool_call_id: String,
    },
}

impl ChatMessage {
    /// Construct one chat message with plain string content.
    ///
    /// # Panics
    ///
    /// Panics for [`ChatRole::ToolResponse`], which requires a tool-call ID.
    /// Use [`Self::tool_response`] for tool responses.
    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        let content: String = text.into();

        match role {
            ChatRole::System => Self::system(content),
            ChatRole::Developer => Self::developer(content, None),
            ChatRole::User => Self::user(content),
            ChatRole::Assistant => Self::assistant_text(content),
            ChatRole::ToolResponse => {
                panic!(
                    "tool response messages require a tool_call_id; \
                     use ChatMessage::tool_response() instead"
                )
            }
        }
    }

    /// Construct one system message.
    pub fn system(content: impl Into<ChatContent>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    /// Construct one developer message.
    pub fn developer(content: impl Into<ChatContent>, tools: Option<Vec<Tool>>) -> Self {
        Self::Developer {
            content: content.into(),
            tools,
        }
    }

    /// Construct one user message.
    pub fn user(content: impl Into<ChatContent>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    /// Construct one assistant message with plain string content.
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![AssistantContentBlock::Text { text: text.into() }],
        }
    }

    /// Construct one assistant message with structured content blocks.
    pub fn assistant_blocks(content: Vec<AssistantContentBlock>) -> Self {
        Self::Assistant { content }
    }

    /// Construct one tool-response message.
    pub fn tool_response(content: impl Into<ChatContent>, tool_call_id: impl Into<String>) -> Self {
        Self::ToolResponse {
            content: content.into(),
            tool_call_id: tool_call_id.into(),
        }
    }

    /// Return the role of this message.
    pub fn role(&self) -> ChatRole {
        match self {
            Self::System { .. } => ChatRole::System,
            Self::Developer { .. } => ChatRole::Developer,
            Self::User { .. } => ChatRole::User,
            Self::Assistant { .. } => ChatRole::Assistant,
            Self::ToolResponse { .. } => ChatRole::ToolResponse,
        }
    }

    /// Concatenate the visible text carried by this message.
    ///
    /// Returns the static content-part type when a non-assistant message
    /// contains multimodal content.
    pub fn text_content(&self) -> Result<String, &'static str> {
        match self {
            Self::System { content }
            | Self::Developer { content, .. }
            | Self::User { content }
            | Self::ToolResponse { content, .. } => content.try_flatten_to_text(),
            Self::Assistant { content } => Ok(content.text()),
        }
    }

    /// Concatenate assistant reasoning text when present.
    pub fn reasoning_content(&self) -> Option<String> {
        match self {
            Self::Assistant { content } => content.reasoning(),
            Self::System { .. }
            | Self::Developer { .. }
            | Self::User { .. }
            | Self::ToolResponse { .. } => None,
        }
    }

    /// Return whether this message contains multimodal content.
    pub fn has_multimodal(&self) -> bool {
        match self {
            Self::System { content }
            | Self::Developer { content, .. }
            | Self::User { content }
            | Self::ToolResponse { content, .. } => content.has_multimodal(),
            Self::Assistant { .. } => false,
        }
    }
}

impl From<AssistantMessage> for ChatMessage {
    fn from(value: AssistantMessage) -> Self {
        Self::Assistant {
            content: value.content,
        }
    }
}
