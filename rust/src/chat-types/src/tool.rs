// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    /// Function name exposed to the model.
    pub name: String,
    /// Optional human-readable function description.
    pub description: Option<String>,
    /// JSON Schema describing the function parameters.
    pub parameters: Value,
    /// Optional strict-schema enforcement request.
    pub strict: Option<bool>,
}
