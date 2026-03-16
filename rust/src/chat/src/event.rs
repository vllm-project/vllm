use vllm_engine_core_client::protocol::{FinishReason, StopReason};

/// Streamed chat event emitted by [`crate::ChatEventStream`].
#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    /// The request was accepted and streaming has started.
    Start,
    /// A newly observed text suffix plus the full cumulative decoded text.
    TextDelta { delta: String, text: String },
    /// Terminal event carrying the final cumulative text and finish metadata.
    Done {
        text: String,
        /// Raw cumulative output token IDs, including a terminal stop token when
        /// the engine emitted one.
        token_ids: Vec<u32>,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    },
}
