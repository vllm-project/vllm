// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Unified parser driven by a checkpoint's Hugging Face `response_template`.
//!
//! The renderer executes a checkpoint's `chat_template`; this parser executes its
//! `response_template`, the declarative output-protocol description introduced by
//! Transformers `chat_parsing` (`ResponseParser`).
//!
//! The template language and its value semantics (content parsers, transforms,
//! tool-schema coercion) follow Transformers at
//! <https://github.com/huggingface/transformers/tree/6d43ab4008/src/transformers/utils/chat_parsing>.
//! The streaming executor is native: it emits ordered [`UnifiedParserEvent`]s
//! instead of Transformers' region events and aggregated message dict.
//!
//! [`UnifiedParserEvent`]: super::UnifiedParserEvent

mod content;
mod parser;
mod pattern;
mod spec;
mod template;
mod transform;

#[cfg(test)]
mod tests;

pub use parser::HfUnifiedParser;
pub use template::ResponseTemplate;
use thiserror::Error;
use thiserror_ext::Macro;

/// Result alias for response-template operations.
pub type Result<T> = std::result::Result<T, HfTemplateError>;

/// Errors produced while loading a `response_template`.
///
/// Failures while parsing generated output are
/// [`UnifiedParserError::ParsingFailed`](super::UnifiedParserError::ParsingFailed), as for
/// the other unified parsers.
#[derive(Debug, Clone, Error, Macro)]
#[thiserror_ext(macro(path = "crate::unified::hf", mangle))]
pub enum HfTemplateError {
    /// The template violates the Transformers `response_template` schema.
    #[error("invalid response_template: {message}")]
    Invalid { message: String },
    /// The template is valid but uses a feature this implementation does not support.
    #[error("unsupported response_template: {message}")]
    Unsupported { message: String },
    /// The model provides no template.
    #[error("the model's tokenizer_config.json provides no response_template")]
    Missing,
}
