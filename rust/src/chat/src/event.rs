use vllm_engine_core_client::protocol::{FinishReason, StopReason};

#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    Start,
    TextDelta {
        delta: String,
        text: String,
    },
    Done {
        text: String,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    },
}
