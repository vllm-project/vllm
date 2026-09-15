// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Number, Value, json};

use crate::error::{Result, invalid_reasoning_control};

/// A model-specific reasoning effort, before or after renderer lowering.
///
/// Strings preserve model-specific names; JSON numbers preserve integer and
/// floating-point representations for model-local range validation. Missing/null
/// effort is represented by `Option::None` in [`crate::ChatOptions`].
/// Supported names and ranges are validated by each renderer or HF template.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EffortValue {
    String(String),
    Number(Number),
}

impl TryFrom<&Value> for EffortValue {
    type Error = crate::error::Error;

    fn try_from(value: &Value) -> Result<Self> {
        match value {
            Value::String(value) => Ok(Self::String(value.clone())),
            Value::Number(value) => Ok(Self::Number(value.clone())),
            _ => Err(invalid_reasoning_control!(
                "reasoning effort must be a string or number, got {value}"
            )),
        }
    }
}

impl From<&str> for EffortValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl fmt::Display for EffortValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(value) => json!(value).fmt(f),
            Self::Number(value) => value.fmt(f),
        }
    }
}

impl EffortValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            Self::Number(_) => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => value.as_f64(),
            Self::String(_) => None,
        }
    }
}
