// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use vllm_tracing::LogLevel;

/// Python's `LoggingConfig`, from `--logging-config` and its dotted forms
/// (`--logging-config.log_level DEBUG`).
///
/// The Rust frontend's own logs honor `log_level` and `formatter`. The other
/// fields configure Python logging only: they are kept as given and forwarded
/// to the managed engine, which validates them.
#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct LoggingConfig {
    #[serde(default)]
    pub log_level: Option<LogLevel>,
    #[serde(default)]
    pub formatter: Option<LogFormatter>,
    #[serde(flatten)]
    pub python_only: Map<String, Value>,
}

/// Python's `LoggingConfig` formatters that the Rust frontend implements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormatter {
    #[default]
    Text,
}
