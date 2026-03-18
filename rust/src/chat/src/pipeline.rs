use futures::Stream;
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::error::Result;
use crate::event::{AssistantBlockKind, AssistantToolCall};

/// Internal assistant-stream event after reasoning/tool parsing but before final assembly.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantStreamEvent {
    Start,
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        id: String,
        delta: String,
    },
    ToolCallEnd {
        call: AssistantToolCall,
    },
    Done {
        token_ids: Vec<u32>,
        finish_reason: Option<FinishReason>,
        stop_reason: Option<StopReason>,
    },
}

pub(crate) trait AssistantStreamEventStream =
    Stream<Item = Result<AssistantStreamEvent>> + Send + 'static;
