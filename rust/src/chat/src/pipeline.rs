use futures::Stream;
use vllm_engine_core_client::protocol::{FinishReason, StopReason};

use crate::decoded::DecodedTextEvent;
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

impl AssistantStreamEvent {
    /// Convert a [`DecodedTextEvent`] into an [`AssistantStreamEvent`] by treating all text as
    /// plain (non-reasoning) content and discarding  cumulative fields.
    pub(crate) fn from_decoded_plain_text(event: DecodedTextEvent) -> Self {
        match event {
            DecodedTextEvent::Start => Self::Start,
            DecodedTextEvent::TextDelta { delta, .. } => Self::TextDelta {
                kind: AssistantBlockKind::Text,
                delta,
            },
            DecodedTextEvent::Done {
                token_ids,
                finish_reason,
                stop_reason,
                ..
            } => Self::Done {
                token_ids,
                finish_reason,
                stop_reason,
            },
        }
    }
}

pub(crate) trait AssistantStreamEventStream =
    Stream<Item = Result<AssistantStreamEvent>> + Send + 'static;
