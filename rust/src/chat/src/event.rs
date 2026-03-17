use std::ops::Deref;

use serde::{Deserialize, Serialize};
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantBlockKind {
    Text,
    Reasoning,
    // ToolCall,
}

/// One structured assistant output block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantContentBlock {
    Text { text: String },
    Reasoning { text: String },
    // ToolCall { ... },
}

impl AssistantContentBlock {
    /// Return the semantic kind of this block.
    pub fn kind(&self) -> AssistantBlockKind {
        match self {
            Self::Text { .. } => AssistantBlockKind::Text,
            Self::Reasoning { .. } => AssistantBlockKind::Reasoning,
        }
    }

    pub(crate) fn from_text_delta(kind: AssistantBlockKind, text: String) -> Self {
        match kind {
            AssistantBlockKind::Text => Self::Text { text },
            AssistantBlockKind::Reasoning => Self::Reasoning { text },
        }
    }
}

#[easy_ext::ext(AssistantContentBlocksExt)]
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
    /// The request was accepted and streaming has started.
    Start,
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
    /// One assistant output block has ended.
    BlockEnd {
        index: usize,
        block: AssistantContentBlock,
    },
    /// Terminal event carrying the final assembled assistant message and finish metadata.
    Done {
        message: AssistantMessage,
        /// Raw cumulative output token IDs, including a terminal stop token when
        /// the engine emitted one.
        token_ids: Vec<u32>,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    },
}
