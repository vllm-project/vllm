use vllm_engine_core_client::protocol::{FinishReason, StopReason};

/// Semantic kind of one assistant output block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantBlockKind {
    Text,
    Reasoning,
    ToolCall,
}

/// One structured assistant output block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantContentBlock {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
}

impl AssistantContentBlock {
    /// Return the semantic kind of this block.
    pub fn kind(&self) -> AssistantBlockKind {
        match self {
            Self::Text { .. } => AssistantBlockKind::Text,
            Self::Reasoning { .. } => AssistantBlockKind::Reasoning,
            Self::ToolCall { .. } => AssistantBlockKind::ToolCall,
        }
    }

    pub(crate) fn from_text_delta(kind: AssistantBlockKind, text: String) -> Self {
        match kind {
            AssistantBlockKind::Text => Self::Text { text },
            AssistantBlockKind::Reasoning => Self::Reasoning { text },
            AssistantBlockKind::ToolCall => {
                unreachable!("tool call cannot be constructed from text deltas")
            }
        }
    }
}

/// Final structured assistant message assembled from the event stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssistantMessage {
    pub content: Vec<AssistantContentBlock>,
}

impl AssistantMessage {
    /// Concatenate all visible final-answer text blocks.
    pub fn text(&self) -> String {
        self.blocks_of_kind(AssistantBlockKind::Text)
            .filter_map(|block| match block {
                AssistantContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Concatenate all extracted reasoning blocks.
    pub fn reasoning(&self) -> String {
        self.blocks_of_kind(AssistantBlockKind::Reasoning)
            .filter_map(|block| match block {
                AssistantContentBlock::Reasoning { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Iterate over blocks of one semantic kind.
    pub fn blocks_of_kind(
        &self,
        kind: AssistantBlockKind,
    ) -> impl Iterator<Item = &AssistantContentBlock> + '_ {
        self.content
            .iter()
            .filter(move |block| block.kind() == kind)
    }

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
