// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use trait_set::trait_set;
use uuid::Uuid;
use vllm_llm::TokenUsage;
use vllm_text::output::{DecodedLogprobs, DecodedPromptLogprobs, DecodedTextEvent};

use crate::FinishReason;
use crate::error::Result;
use crate::event::{AssistantBlockKind, ChatEvent};

mod default;
mod harmony;
mod structured;

pub use default::DefaultChatOutputProcessor;
pub use harmony::HarmonyChatOutputProcessor;
pub(crate) use harmony::validate_harmony_parser_overrides;

/// Internal assistant event before final assembly.
///
/// Unified parsing produces these events, and structured assembly consumes
/// them to build public chat events.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssistantEvent {
    Start {
        prompt_token_ids: Arc<[u32]>,
        prompt_logprobs: Option<DecodedPromptLogprobs>,
    },
    TextDelta {
        kind: AssistantBlockKind,
        delta: String,
    },
    /// Per-decoded-update sample metadata: logprobs and/or output token IDs.
    LogprobsDelta {
        logprobs: Option<DecodedLogprobs>,
        token_ids: Vec<u32>,
    },
    /// The start of a new tool call, with its declared name and generated ID.
    ToolCallStart { id: String, name: String },
    /// A delta for the arguments of the currently open tool call. Must follow a
    /// `ToolCallStart`.
    ToolCallArgumentsDelta { delta: String },
    Done {
        usage: TokenUsage,
        finish_reason: FinishReason,
        /// Connector-specific KV transfer parameters for disaggregated serving.
        kv_transfer_params: Option<serde_json::Value>,
        /// Connector-specific encoder cache transfer parameters for
        /// disaggregated serving.
        ec_transfer_params: Option<serde_json::Value>,
    },
}

/// Boxed stream of decoded text events coming from [`vllm_text`].
pub type DynDecodedTextEventStream = Pin<Box<dyn Stream<Item = Result<DecodedTextEvent>> + Send>>;
/// Boxed stream of structured chat events exposed by [`crate::ChatLlm`].
pub type DynChatEventStream = Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>;

/// Request-scoped output processor from decoded text events into structured
/// chat events.
pub trait ChatOutputProcessor: Send {
    /// Consume decoded text stream and return the structured chat-event stream.
    fn process(self: Box<Self>, decoded: DynDecodedTextEventStream) -> Result<DynChatEventStream>;
}

/// Trait-object form of [`ChatOutputProcessor`].
pub type DynChatOutputProcessor = Box<dyn ChatOutputProcessor>;

trait_set! {
    /// Boxed-stream constraint for decoded text updates.
    pub(crate) trait DecodedTextEventStream = Stream<Item = Result<DecodedTextEvent>> + Send + 'static;
    /// Boxed-stream constraint for internal assistant events.
    pub(crate) trait AssistantEventStream = Stream<Item = Result<AssistantEvent>> + Send + 'static;
    /// Boxed-stream constraint for public chat events.
    pub(crate) trait ChatEventStream = Stream<Item = Result<ChatEvent>> + Send + 'static;
}

/// Generate the northbound tool-call ID using the OpenAI-style `call_<id>`
/// format.
pub(crate) fn generate_tool_call_id() -> String {
    format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
}
