// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ops::Deref;

use serde::{Deserialize, Serialize};

/// One finalized assistant tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantToolCall {
    /// Stable tool-call identifier.
    pub id: String,
    /// Function name selected by the assistant.
    pub name: String,
    /// Serialized function arguments.
    pub arguments: String,
}

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantBlockKind {
    /// Visible final-answer text.
    Text,
    /// Extracted reasoning content.
    Reasoning,
    /// One finalized tool call.
    ToolCall,
}

/// One structured assistant output block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantContentBlock {
    /// Visible final-answer text.
    Text {
        /// Visible text.
        text: String,
    },
    /// Extracted reasoning content.
    Reasoning {
        /// Reasoning text.
        text: String,
    },
    /// One finalized tool call.
    ToolCall(AssistantToolCall),
}

impl AssistantContentBlock {
    /// Return the semantic kind of this block.
    pub fn kind(&self) -> AssistantBlockKind {
        match self {
            Self::Text { .. } => AssistantBlockKind::Text,
            Self::Reasoning { .. } => AssistantBlockKind::Reasoning,
            Self::ToolCall(..) => AssistantBlockKind::ToolCall,
        }
    }

    /// Return this block as one finalized tool call when applicable.
    pub fn as_tool_call(&self) -> Option<&AssistantToolCall> {
        match self {
            Self::ToolCall(call) => Some(call),
            _ => None,
        }
    }

    /// Trim whitespace from text and tool arguments.
    ///
    /// Returns `None` when trimming makes a text or reasoning block empty.
    pub fn trim(mut self) -> Option<Self> {
        match &mut self {
            Self::Text { text } | Self::Reasoning { text } => {
                let trimmed_text = text.trim();
                if trimmed_text.is_empty() {
                    return None;
                }
                *text = trimmed_text.to_string();
            }
            Self::ToolCall(call) => {
                call.arguments = call.arguments.trim().to_string();
            }
        }
        Some(self)
    }
}

#[easy_ext::ext(AssistantMessageExt)]
impl [AssistantContentBlock] {
    /// Concatenate all visible final-answer text blocks.
    pub fn text(&self) -> String {
        self.iter()
            .filter_map(|block| match block {
                AssistantContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Concatenate all extracted reasoning blocks.
    pub fn reasoning(&self) -> Option<String> {
        Some(
            self.iter()
                .filter_map(|block| match block {
                    AssistantContentBlock::Reasoning { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
        )
        .filter(|text: &String| !text.is_empty())
    }

    /// Return whether this assistant message contains reasoning text.
    pub fn has_reasoning(&self) -> bool {
        self.iter().any(|block| match block {
            AssistantContentBlock::Reasoning { text } => !text.is_empty(),
            _ => false,
        })
    }

    /// Iterate over finalized assistant tool calls in encounter order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &AssistantToolCall> {
        self.iter().filter_map(AssistantContentBlock::as_tool_call)
    }

    /// Return whether this assistant message contains any tool-call blocks.
    pub fn has_tool_calls(&self) -> bool {
        self.iter().any(|block| matches!(block, AssistantContentBlock::ToolCall(_)))
    }
}

/// Final structured assistant message assembled from parsed output.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// Assistant content blocks in emission order.
    pub content: Vec<AssistantContentBlock>,
}

impl Deref for AssistantMessage {
    type Target = [AssistantContentBlock];

    fn deref(&self) -> &Self::Target {
        &self.content
    }
}

impl AssistantMessage {
    /// Push one new block to the end of the message content.
    pub fn push_block(&mut self, block: AssistantContentBlock) {
        self.content.push(block);
    }

    /// Trim all blocks and remove text blocks that become empty.
    pub fn trim(mut self) -> Self {
        self.content = self.content.into_iter().filter_map(AssistantContentBlock::trim).collect();
        self
    }
}
