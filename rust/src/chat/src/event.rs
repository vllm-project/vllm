// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use vllm_llm::TokenUsage;
use vllm_text::{DecodedLogprobs, DecodedPromptLogprobs};

use crate::FinishReason;

pub use vllm_chat_types::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall,
};

/// Streamed chat event emitted by [`crate::ChatEventStream`].
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    /// The request was accepted, streaming has started, and prompt metadata is
    /// ready.
    Start {
        /// The actual prompt token IDs for this request.
        prompt_token_ids: Arc<[u32]>,
        /// Once-only prompt logprobs metadata, when requested.
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    /// A new assistant output block has started.
    BlockStart {
        /// Stable block index within the assistant message.
        index: usize,
        /// Semantic kind of the opened block.
        kind: AssistantBlockKind,
    },
    /// A newly observed delta for one open assistant output block.
    BlockDelta {
        /// Stable block index within the assistant message.
        index: usize,
        /// Semantic kind of the open block.
        kind: AssistantBlockKind,
        /// Newly emitted text.
        delta: String,
    },
    /// Per-decoded-update sample metadata.
    LogprobsDelta {
        /// Decoded output logprobs, when requested.
        logprobs: Option<DecodedLogprobs>,
        /// Output token IDs emitted by this update.
        token_ids: Vec<u32>,
    },
    /// One assistant output block has ended.
    BlockEnd {
        /// Stable block index within the assistant message.
        index: usize,
        /// Finalized block.
        block: AssistantContentBlock,
    },
    /// One tool call has started.
    ToolCallStart {
        /// Stable tool-call index within the assistant message.
        index: usize,
        /// Stable tool-call identifier.
        id: String,
        /// Function name selected by the assistant.
        name: String,
    },
    /// One incremental tool-call arguments delta.
    ToolCallArgumentsDelta {
        /// Stable tool-call index within the assistant message.
        index: usize,
        /// Newly emitted arguments text.
        delta: String,
    },
    /// One tool call has ended.
    ToolCallEnd {
        /// Stable tool-call index within the assistant message.
        index: usize,
        /// Finalized tool call.
        call: AssistantToolCall,
    },
    /// Terminal event carrying the final assembled assistant message and finish
    /// metadata.
    Done {
        /// Final structured assistant message.
        message: AssistantMessage,
        /// Final token usage.
        usage: TokenUsage,
        /// Reason generation stopped.
        finish_reason: FinishReason,
        /// Connector-specific KV transfer parameters for disaggregated serving.
        kv_transfer_params: Option<serde_json::Value>,
        /// Connector-specific encoder cache transfer parameters for
        /// disaggregated serving.
        ec_transfer_params: Option<serde_json::Value>,
    },
}
