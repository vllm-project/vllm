mod default;

use std::sync::Arc;

use futures::Stream;
use subenum::subenum;
use vllm_text::output::{DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent};

use crate::FinishReason;
use crate::error::Result;
use crate::event::{AssistantBlockKind, AssistantToolCall, ChatEvent};

/// Internal assistant event before final assembly.
///
/// - [`ContentEvent`]: subenum after reasoning parsing, carries only text content.
/// - [`AssistantEvent`]: full event after tool parsing, adds tool-call variants.
#[subenum(ContentEvent)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantEvent {
    #[subenum(ContentEvent)]
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    #[subenum(ContentEvent)]
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    /// Per-decoded-update sample metadata: logprobs and/or output token IDs.
    #[subenum(ContentEvent)]
    LogprobsDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
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
    #[subenum(ContentEvent)]
    Done {
        prompt_token_count: usize,
        output_token_count: usize,
        finish_reason: FinishReason,
        /// Connector-specific KV transfer parameters for disaggregated serving.
        kv_transfer_params: Option<serde_json::Value>,
    },
}

impl ContentEvent {
    /// Convert a [`DecodedTextEvent`] into one or more [`ContentEvent`] values by treating all text
    /// as plain (non-reasoning) content.
    fn from_decoded_plain_text(event: DecodedTextEvent) -> Vec<Self> {
        match event {
            DecodedTextEvent::Start {
                prompt_token_ids,
                prompt_logprobs,
            } => vec![Self::Start {
                prompt_token_ids,
                prompt_logprobs,
            }],
            DecodedTextEvent::TextDelta {
                delta,
                token_ids,
                logprobs,
                finished,
            } => {
                let mut events = Vec::new();
                if !delta.is_empty() {
                    events.push(Self::TextDelta {
                        kind: AssistantBlockKind::Text,
                        delta,
                    });
                }
                if logprobs.is_some() || !token_ids.is_empty() {
                    events.push(Self::LogprobsDelta {
                        logprobs,
                        token_ids,
                    });
                }
                if let Some(finished) = finished {
                    events.push(Self::Done {
                        prompt_token_count: finished.prompt_token_count,
                        output_token_count: finished.output_token_count,
                        finish_reason: finished.finish_reason,
                        kv_transfer_params: finished.kv_transfer_params,
                    });
                }
                events
            }
        }
    }
}

pub(crate) trait DecodedTextEventStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
pub(crate) trait ChatEventStream = Stream<Item = Result<ChatEvent>> + Send + 'static;

pub(crate) use default::{OutputProcessors, output_stream};
