use vllm_engine_core_client::protocol::{FinishReason, StopReason};

#[derive(Debug, Clone, PartialEq)]
pub enum ChatEvent {
    Start {
        request_id: String,
    },
    TextDelta {
        request_id: String,
        delta: String,
        text: String,
    },
    Done {
        request_id: String,
        text: String,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    },
    Error {
        request_id: String,
        message: String,
    },
}
