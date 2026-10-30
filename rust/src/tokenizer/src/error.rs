// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use thiserror_ext::Macro;

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
#[error("tokenizer error: {0}")]
pub struct TokenizerError(#[message] pub String);
