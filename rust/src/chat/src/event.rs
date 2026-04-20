use std::ops::Deref;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use vllm_text::{DecodedLogprobs, DecodedPromptLogprobs};

use crate::FinishReason;

/// One finalized assistant tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    pub name: String,
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
    Text { text: String },
    /// Extracted reasoning content.
    Reasoning { text: String },
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

    /// Return this block as one finalized tool call, if applicable.
    pub fn as_tool_call(&self) -> Option<&AssistantToolCall> {
        match self {
            Self::ToolCall(call) => Some(call),
            _ => None,
        }
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

    /// Concatenate all extracted reasoning blocks, if any.
    pub fn reasoning(&self) -> Option<String> {
        Some(
            self.iter()
                .filter_map(|block| match block {
                    AssistantContentBlock::Reasoning { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect(),
        )
        .filter(|s: &String| !s.is_empty())
    }

    /// Return whether this assistant message contains any non-empty reasoning text blocks.
    pub fn has_reasoning(&self) -> bool {
        self.iter().any(|block| match block {
            AssistantContentBlock::Reasoning { text } => !text.is_empty(),
            _ => false,
        })
    }

    /// Return finalized assistant tool calls in encounter order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &AssistantToolCall> {
        self.iter().filter_map(AssistantContentBlock::as_tool_call)
    }

    /// Return whether this assistant message contains any tool-call blocks.
    pub fn has_tool_calls(&self) -> bool {
        self.iter()
            .any(|block| matches!(block, AssistantContentBlock::ToolCall(_)))
    }
}

/// Final structured assistant message assembled from the event stream.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantMessage {
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
    pub(crate) fn push_block(&mut self, block: AssistantContentBlock) {
        self.content.push(block);
    }
}

/// Streamed chat event emitted by [`crate::ChatEventStream`].
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    /// The request was accepted, streaming has started, and prompt metadata is ready.
    Start {
        /// The actual prompt token IDs for this request.
        prompt_token_ids: Arc<[u32]>,
        /// Once-only prompt logprobs metadata, when requested.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    /// A new assistant output block has started.
    BlockStart {
        index: usize,
        kind: AssistantBlockKind,
    },
    /// A newly observed delta for one open assistant output block.
    BlockDelta {
        index: usize,
        kind: AssistantBlockKind,
        delta: String,
    },
    /// Per-decoded-update sample metadata: logprobs and/or output token IDs.
    LogprobsDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
    },
    /// One assistant output block has ended.
    BlockEnd {
        index: usize,
        block: AssistantContentBlock,
    },
    /// One tool call has started.
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// One incremental tool-call arguments delta.
    ToolCallArgumentsDelta {
        index: usize,
        id: String,
        delta: String,
    },
    /// One tool call has ended.
    ToolCallEnd {
        index: usize,
        call: AssistantToolCall,
    },
    /// Terminal event carrying the final assembled assistant message and finish metadata.
    Done {
        message: AssistantMessage,
        /// Number of prompt tokens actually sent to the engine after chat
        /// template rendering and tokenization.
        prompt_token_count: usize,
        /// Number of output tokens generated.
        output_token_count: usize,
        finish_reason: FinishReason,
        /// Connector-specific KV transfer parameters for disaggregated serving.
        kv_transfer_params: Option<serde_json::Value>,
    },
}
