use thiserror::Error as ThisError;

/// Error returned while resolving or reading model files.
#[derive(Debug, ThisError)]
#[error("model file error: {0}")]
pub struct Error(String);

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Result type used by model-file discovery helpers.
pub type Result<T> = std::result::Result<T, Error>;
