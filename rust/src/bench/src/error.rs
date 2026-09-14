// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;

#[derive(Error, Debug)]
pub enum BenchError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// The server's /tokenize//detokenize endpoint is not usable (4xx status:
    /// not exposed, or rejected by a gateway such as LLM-d/EPP that returns
    /// 400 instead of 404). Callers treat this as "skip verification", unlike
    /// `Tokenizer` errors which are genuine failures.
    #[error("tokenize endpoint unavailable: {0}")]
    TokenizeUnavailable(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Endpoint not ready after {0}s: {1}")]
    EndpointTimeout(u64, String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, BenchError>;
