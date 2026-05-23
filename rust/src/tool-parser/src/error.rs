use thiserror::Error;
use thiserror_ext::Macro;

/// Result alias for tool parser operations.
pub type Result<T> = std::result::Result<T, ToolParserError>;

/// Errors produced while creating or running tool parsers.
#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
pub enum ToolParserError {
    #[error("tool parser parsing failed: {message}")]
    ParsingFailed { message: String },
}
