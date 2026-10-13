// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::Macro;

/// Result alias for tool parser operations.
pub type Result<T> = std::result::Result<T, ToolParserError>;

/// Errors produced while creating or running tool parsers.
#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::tool::error"))]
pub enum ToolParserError {
    #[error("tool parser parsing failed: {message}")]
    ParsingFailed { message: String },
    #[error(
        "`{name}` only provides a unified parser; the same reasoning parser and tool parser should be specified together"
    )]
    DummyUnifiedParser { name: String },
}
