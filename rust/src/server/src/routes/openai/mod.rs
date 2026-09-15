// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

pub mod chat_completions;
mod completions;
mod models;
pub(crate) mod utils;

pub use chat_completions::chat_completions;
pub(crate) use chat_completions::{ChatCompletionRequest, lower_chat_request};
pub use completions::completions;
// TODO: phase 3 of the derender endpoints re-exports CompletionStreamChoice
// and CompletionStreamResponse here as well.
pub(crate) use completions::{
    CompletionChoice, CompletionRequest, CompletionResponse, completion_echo_text,
    lower_completion_request,
};
pub use models::list_models;
