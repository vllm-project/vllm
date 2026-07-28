// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Engine-independent data types shared by chat renderers and output parsers.
//!
//! This crate defines chat history, rendering options, tool descriptions, and
//! structured assistant payloads. Serving requests, streamed events, renderer
//! implementations, parser state, and engine metadata live in their owning
//! crates.

mod assistant;
mod content;
mod message;
mod options;
#[cfg(test)]
mod tests;
mod tool;

pub use assistant::{
    AssistantBlockKind, AssistantContentBlock, AssistantMessage, AssistantMessageExt,
    AssistantToolCall,
};
pub use content::{ChatContent, ChatContentPart, ImageDetail};
pub use message::{ChatMessage, ChatRole};
pub use options::{ChatOptions, ChatToolChoice, GenerationPromptMode, ReasoningEffort};
pub use tool::Tool;
