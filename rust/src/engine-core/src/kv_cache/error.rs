// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/// Errors the KV cache manager reports to its caller.
///
/// Python `ValueError`s raised for invalid configuration or invalid caller
/// input map to typed errors; violated internal invariants (Python `assert`s
/// and errors that only a bookkeeping bug can trigger) panic.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvCacheError {
    /// The configuration is invalid (`ValueError` at construction).
    #[error("{message}")]
    InvalidConfig { message: String },
    /// The configuration uses a feature this port does not implement.
    #[error("{message}")]
    Unsupported { message: String },
    /// A method was called with invalid arguments (`ValueError`).
    #[error("{message}")]
    InvalidArgument { message: String },
}
