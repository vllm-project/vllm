// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::ffi::OsString;

use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub(crate) enum TemplateError {
    #[error("failed to render jinja template")]
    Jinja(#[from] minijinja::Error),
    #[error("chat template exceeded its fuel budget; adjust VLLM_CHAT_TEMPLATE_FUEL to override")]
    OutOfFuel(#[source] minijinja::Error),
    #[error("VLLM_CHAT_TEMPLATE_FUEL must be a non-negative integer, got {0:?}")]
    InvalidFuel(OsString),
    #[error("failed to read chat template file")]
    ReadTemplateFile(#[source] std::io::Error),
    #[error("chat template looks like a file path but does not exist")]
    MissingTemplatePath,
    #[error("failed to parse chat_template.json")]
    ParseTemplateJson(#[source] serde_json::Error),
    #[error("chat_template.json does not contain a valid template")]
    InvalidTemplateJson,
}
