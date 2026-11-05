// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
pub enum TokenizerError {
    /// The tokenizer backend failed to load, encode, or decode.
    #[error("tokenizer error: {message}")]
    Backend { message: String },
    /// No tokenizer is loaded, so text cannot be encoded or decoded.
    #[error(
        "this request needs a tokenizer, which is not loaded when `--skip-tokenizer-init` is set"
    )]
    Unavailable,
}
