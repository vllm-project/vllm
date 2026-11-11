// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, Error>;

/// Encoding, decoding or validating a protocol message failed.
#[derive(Debug, Error, Macro)]
pub enum Error {
    #[error("messagepack encode failed for {target_type}: {message}")]
    Encode {
        target_type: &'static str,
        message: String,
    },
    #[error("messagepack decode failed for {target_type}: {message}")]
    Decode {
        target_type: &'static str,
        message: String,
    },
    #[error("messagepack value decode failed")]
    ValueDecode(#[from] rmpv::decode::Error),
    #[error("messagepack ext value decode failed: {message}")]
    ExtValueDecode { message: String },
    #[error("invalid structured outputs params: {message}")]
    InvalidStructuredOutputsParams { message: String },
    #[error("unsupported field `{field}` in {context}")]
    UnsupportedField {
        context: &'static str,
        field: &'static str,
    },
}
