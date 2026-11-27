// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::convert::Infallible;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::response::sse::Event;
use futures::{Stream, StreamExt as _};
use serde::Serialize;
use vllm_chat::{AssistantBlockKind, ChatEvent, ChatEventStream};

use super::log_completion;
use super::response::{AnthropicError, convert_usage, stop_reason, thinking_signature};
use super::types::{MessagesResponse, MessagesStopReason, MessagesUsage, OutputContentBlock};
use crate::config::ApiServerOptions;
use crate::error::{ApiError, chat_submit_error, server_error};
use crate::routes::output_log::LoggedOutput;
use crate::utils::EventExt as _;

pub(super) fn messages_stream(
    stream: ChatEventStream,
    request_id: String,
    model: String,
    options: ApiServerOptions,
) -> impl Stream<Item = Result<Event, Infallible>> {
    stream_events(stream, request_id, model, options).map(|event| {
        Ok(match event {
            Ok(event) => event.into_sse(),
            Err(error) => {
                Event::default().event("error").json(&AnthropicError::from(error).into_body())
            }
        })
    })
}

#[try_stream]
async fn stream_events(
    mut stream: ChatEventStream,
    request_id: String,
    model: String,
    options: ApiServerOptions,
    mut y: TryYielder<MessageStreamEvent, ApiError>,
) -> Result<(), ApiError> {
    // Chat tool indexes and text/thinking indexes are separate. The Anthropic
    // content array has a single index across the ordered block stream.
    let mut block_index = 0;
    while let Some(event) = stream.next().await {
        match event.map_err(|error| chat_submit_error("Messages stream failed", error))? {
            ChatEvent::Start {
                prompt_token_ids, ..
            } => {
                // Report uncached input with the final cache breakdown when enabled.
                let input_tokens = if options.enable_prompt_tokens_details {
                    0
                } else {
                    prompt_token_ids.len()
                };
                y.yield_ok(MessageStreamEvent::MessageStart {
                    message: Box::new(MessagesResponse {
                        id: request_id.clone(),
                        role: "assistant",
                        content: Vec::new(),
                        model: model.clone(),
                        stop_reason: None,
                        stop_sequence: None,
                        usage: MessagesUsage {
                            input_tokens,
                            output_tokens: 0,
                            cache_read_input_tokens: None,
                            cache_creation_input_tokens: None,
                        },
                        kv_transfer_params: None,
                        ec_transfer_params: None,
                    }),
                })
                .await;
            }
            ChatEvent::BlockStart { kind, .. } => {
                let content_block = match kind {
                    AssistantBlockKind::Text => OutputContentBlock::Text {
                        text: String::new(),
                    },
                    AssistantBlockKind::Reasoning => OutputContentBlock::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    },
                    AssistantBlockKind::ToolCall => {
                        return Err(server_error!("Unexpected Messages tool block"));
                    }
                };
                y.yield_ok(MessageStreamEvent::ContentBlockStart {
                    index: block_index,
                    content_block,
                })
                .await;
            }
            ChatEvent::BlockDelta { kind, delta, .. } => {
                let delta = match kind {
                    AssistantBlockKind::Text => ContentDelta::TextDelta { text: delta },
                    AssistantBlockKind::Reasoning => {
                        ContentDelta::ThinkingDelta { thinking: delta }
                    }
                    AssistantBlockKind::ToolCall => {
                        return Err(server_error!("Unexpected Messages tool delta"));
                    }
                };
                y.yield_ok(MessageStreamEvent::ContentBlockDelta {
                    index: block_index,
                    delta,
                })
                .await;
            }
            ChatEvent::BlockEnd { block, .. } => {
                if block.kind() == AssistantBlockKind::Reasoning {
                    let delta = ContentDelta::SignatureDelta {
                        signature: thinking_signature(),
                    };
                    y.yield_ok(MessageStreamEvent::ContentBlockDelta {
                        index: block_index,
                        delta,
                    })
                    .await;
                }
                y.yield_ok(MessageStreamEvent::ContentBlockStop { index: block_index }).await;
                block_index += 1;
            }
            ChatEvent::ToolCallStart { id, name, .. } => {
                let content_block = OutputContentBlock::ToolUse {
                    id,
                    name,
                    input: Default::default(),
                };
                y.yield_ok(MessageStreamEvent::ContentBlockStart {
                    index: block_index,
                    content_block,
                })
                .await;
            }
            ChatEvent::ToolCallArgumentsDelta { delta, .. } => {
                let delta = ContentDelta::InputJsonDelta {
                    partial_json: delta,
                };
                y.yield_ok(MessageStreamEvent::ContentBlockDelta {
                    index: block_index,
                    delta,
                })
                .await;
            }
            ChatEvent::ToolCallEnd { .. } => {
                y.yield_ok(MessageStreamEvent::ContentBlockStop { index: block_index }).await;
                block_index += 1;
            }
            ChatEvent::LogprobsDelta { .. } => {}
            ChatEvent::Done {
                message,
                finish_reason,
                usage: final_usage,
                ..
            } => {
                let (reason, sequence) = stop_reason(&finish_reason, &message)?;
                log_completion(
                    options,
                    &model,
                    &message,
                    final_usage,
                    reason,
                    LoggedOutput::StreamComplete,
                    None,
                );
                if block_index == 0 {
                    let content_block = OutputContentBlock::Text {
                        text: String::new(),
                    };
                    y.yield_ok(MessageStreamEvent::ContentBlockStart {
                        index: block_index,
                        content_block,
                    })
                    .await;
                    y.yield_ok(MessageStreamEvent::ContentBlockStop { index: block_index }).await;
                }
                y.yield_ok(MessageStreamEvent::MessageDelta {
                    delta: StopDelta {
                        stop_reason: reason,
                        stop_sequence: sequence.map(str::to_owned),
                    },
                    usage: convert_usage(final_usage.engine, options.enable_prompt_tokens_details),
                })
                .await;
                y.yield_ok(MessageStreamEvent::MessageStop).await;
                return Ok(());
            }
        }
    }
    Err(server_error!(
        "Messages stream ended before generation finished"
    ))
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MessageStreamEvent {
    MessageStart {
        message: Box<MessagesResponse>,
    },
    ContentBlockStart {
        index: usize,
        content_block: OutputContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: StopDelta,
        usage: MessagesUsage,
    },
    MessageStop,
}

impl MessageStreamEvent {
    fn into_sse(self) -> Event {
        let name = match &self {
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop => "message_stop",
        };
        Event::default().event(name).json(&self)
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "Variant names match the wire values"
)]
enum ContentDelta {
    TextDelta { text: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Serialize)]
struct StopDelta {
    stop_reason: Option<MessagesStopReason>,
    stop_sequence: Option<String>,
}
