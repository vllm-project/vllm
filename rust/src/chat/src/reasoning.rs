// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Shared reasoning-control resolution for native renderers.
//!
//! Each input source is parsed independently into [`ReasoningControl`]. Sources
//! then fall back in priority order: request, deployment defaults, model default.
//! Within a source, an explicit toggle decides the mode and effort supplies its
//! intensity. Keeping sources separate lets request effort override a deployment
//! toggle, and lets an enabled request inherit effort from enabled defaults.
//!
//! This module owns input types, precedence, and the standard `"none"` sentinel.
//! Model adapters own supported effort names, numeric ranges, and prompt encoding.
//! After model lowering, [`ReasoningControl::template_kwargs`] projects the prompt's
//! decision for downstream consumers. HF prepares raw template inputs separately.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Number, Value, json};

use crate::error::{Result, invalid_reasoning_control};
use crate::request::ChatRequest;

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

impl TryFrom<Value> for EffortValue {
    type Error = crate::error::Error;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::String(value) => Ok(Self::String(value)),
            Value::Number(value) => Ok(Self::Number(value)),
            _ => Err(invalid_reasoning_control!(
                "reasoning effort must be a string or number, got {value}"
            )),
        }
    }
}

impl TryFrom<f64> for EffortValue {
    type Error = crate::error::Error;

    /// Construct a numeric effort, rejecting NaN and infinities.
    fn try_from(value: f64) -> Result<Self> {
        Number::from_f64(value).map(Self::Number).ok_or_else(|| {
            invalid_reasoning_control!("reasoning effort must be finite, got {value}")
        })
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

/// Reasoning intent for one source, or the result of falling back across sources.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ReasoningControl {
    /// This source leaves the mode and effort to a lower-priority source.
    Default,
    /// Reasoning is explicitly disabled; lower-priority defaults preserve this mode.
    Disabled,
    /// Reasoning is enabled. Missing effort inherits from an enabled default.
    Enabled { effort: Option<EffortValue> },
}

impl ReasoningControl {
    /// Parse request controls; the typed effort field takes precedence over kwargs effort.
    pub(crate) fn from_request(request: &ChatRequest) -> Result<Self> {
        Self::from_kwargs(
            &request.chat_options.template_kwargs,
            request.chat_options.reasoning_effort.as_ref(),
        )
    }

    /// Parse a deployment source, whose controls all come from template kwargs.
    pub(crate) fn from_template_kwargs(kwargs: &HashMap<String, Value>) -> Result<Self> {
        Self::from_kwargs(kwargs, None)
    }

    /// Select controls within one source, then derive its mode and optional effort.
    fn from_kwargs(
        kwargs: &HashMap<String, Value>,
        typed_effort: Option<&EffortValue>,
    ) -> Result<Self> {
        // Presence sets alias priority: `thinking` wins, including when its value
        // is invalid. Validate the selected spelling as a boolean.
        let toggle = ["thinking", "enable_thinking"]
            .into_iter()
            .find_map(|key| kwargs.get(key).map(|value| (key, value)))
            .map(|(key, value)| {
                value.as_bool().ok_or_else(|| {
                    invalid_reasoning_control!(
                        "template kwarg `{key}` must be a boolean, got {value}"
                    )
                })
            })
            .transpose()?;

        // The typed request field wins over kwargs. Parse kwargs only when used;
        // disabled mode discards effort, and JSON null means omitted effort.
        let effort = match (toggle, typed_effort) {
            (Some(false), _) => None,
            (_, Some(effort)) => Some(effort.clone()),
            (_, None) => kwargs
                .get("reasoning_effort")
                .filter(|effort| !effort.is_null())
                .cloned()
                .map(EffortValue::try_from)
                .transpose()?,
        };

        match (toggle, effort) {
            // 1. Explicit false disables reasoning and discards any effort,
            //    including values that would fail validation in an enabled mode.
            (Some(false), _) => Ok(Self::Disabled),
            // 2. `none` disables reasoning when the toggle is absent. Explicit true
            //    enables reasoning and leaves effort to enabled deployment/model defaults.
            (toggle, Some(EffortValue::String(effort))) if effort == "none" => {
                Ok(if toggle == Some(true) {
                    Self::Enabled { effort: None }
                } else {
                    Self::Disabled
                })
            }
            // 3. Absent toggle and absent/null effort leave the entire decision
            //    to the next source in the fallback chain.
            (None, None) => Ok(Self::Default),
            // 4. Explicit true or a concrete effort enables reasoning;
            //    missing effort inherits later.
            (_, effort) => Ok(Self::Enabled { effort }),
        }
    }

    /// Parse request and deployment independently, then apply request precedence.
    ///
    /// Each source validates its selected input types. Model adapters subsequently
    /// apply their model default and validate/map the resulting effective effort.
    pub(crate) fn resolve(
        request: &ChatRequest,
        defaults: &HashMap<String, Value>,
    ) -> Result<Self> {
        Ok(Self::from_request(request)?.fallback(Self::from_template_kwargs(defaults)?))
    }

    /// Fill this higher-priority source's missing decision or effort from `defaults`.
    ///
    /// `Default` inherits the whole lower-priority state. `Enabled { effort: None }`
    /// inherits effort only from enabled defaults and retains its enabled mode.
    /// Explicit disabling and concrete effort are final at this priority level.
    pub(crate) fn fallback(self, defaults: Self) -> Self {
        match (self, defaults) {
            (Self::Default, defaults) => defaults,
            (Self::Enabled { effort: None }, Self::Enabled { effort }) => Self::Enabled { effort },
            (request, _) => request,
        }
    }

    /// Construct an enabled model default or a model-lowered concrete effort.
    pub(crate) fn enabled(effort: impl Into<EffortValue>) -> Self {
        Self::Enabled {
            effort: Some(effort.into()),
        }
    }

    /// Report the resolved mode; callers apply model defaults before rendering.
    pub(crate) fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }

    /// Return the active effort, which may still be missing before model fallback.
    pub(crate) fn effort(&self) -> Option<&EffortValue> {
        match self {
            Self::Enabled { effort } => effort.as_ref(),
            Self::Default | Self::Disabled => None,
        }
    }

    /// Project the same model-lowered decision used to render the native prompt.
    ///
    /// Preserve request extras and write both toggle aliases plus the effective
    /// effort. Disabled mode emits `"none"`; enabled mode with no effective effort
    /// clears the original effort key. `Default` preserves request kwargs.
    pub(crate) fn template_kwargs(&self, request: &ChatRequest) -> HashMap<String, Value> {
        let mut kwargs = request.chat_options.template_kwargs.clone();
        if matches!(self, Self::Default) {
            return kwargs;
        }
        kwargs.insert("thinking".to_string(), json!(self.is_enabled()));
        kwargs.insert("enable_thinking".to_string(), json!(self.is_enabled()));
        match self {
            Self::Disabled => {
                kwargs.insert("reasoning_effort".to_string(), json!("none"));
            }
            Self::Enabled {
                effort: Some(effort),
            } => {
                kwargs.insert("reasoning_effort".to_string(), json!(effort));
            }
            Self::Enabled { effort: None } => {
                kwargs.remove("reasoning_effort");
            }
            Self::Default => unreachable!(),
        }
        kwargs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kwargs(value: Value) -> HashMap<String, Value> {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn numeric_effort_rejects_non_finite_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(EffortValue::try_from(value).unwrap_err().is_request_validation_error());
        }
    }

    #[test]
    fn explicit_toggle_decides_mode_and_none_inherits_enabled_effort() {
        use ReasoningControl::{Default, Disabled, Enabled};

        for (source, expected) in [
            (json!({}), Default),
            (json!({"reasoning_effort": "none"}), Disabled),
            (
                json!({"reasoning_effort": "high"}),
                ReasoningControl::enabled("high"),
            ),
            (json!({"enable_thinking": true}), Enabled { effort: None }),
            (
                json!({"enable_thinking": true, "reasoning_effort": "none"}),
                Enabled { effort: None },
            ),
            (
                json!({"enable_thinking": true, "reasoning_effort": "low"}),
                ReasoningControl::enabled("low"),
            ),
            (
                json!({"enable_thinking": false, "reasoning_effort": "high"}),
                Disabled,
            ),
            (
                json!({"thinking": true, "enable_thinking": false}),
                Enabled { effort: None },
            ),
            (
                json!({"thinking": false, "enable_thinking": true}),
                Disabled,
            ),
            (
                json!({"reasoning_effort": 37}),
                ReasoningControl::enabled(EffortValue::Number(37.into())),
            ),
            (json!({"reasoning_effort": null}), Default),
        ] {
            let source = kwargs(source);
            assert_eq!(
                ReasoningControl::from_template_kwargs(&source).unwrap(),
                expected,
                "{source:?}"
            );
        }
    }

    #[test]
    fn request_and_deployment_states_fall_back_before_model_defaults() {
        for (request, defaults, expected) in [
            (
                json!({}),
                json!({"enable_thinking": false}),
                ReasoningControl::Disabled,
            ),
            (
                json!({"reasoning_effort": "high"}),
                json!({"thinking": false}),
                ReasoningControl::enabled("high"),
            ),
            (
                json!({"enable_thinking": false}),
                json!({"reasoning_effort": "high"}),
                ReasoningControl::Disabled,
            ),
            (
                json!({"thinking": true, "reasoning_effort": "none"}),
                json!({"reasoning_effort": "low"}),
                ReasoningControl::enabled("low"),
            ),
            (
                json!({"thinking": true}),
                json!({"thinking": false, "reasoning_effort": "low"}),
                ReasoningControl::enabled("max"),
            ),
            (json!({}), json!({}), ReasoningControl::enabled("max")),
        ] {
            let request = kwargs(request);
            let defaults = kwargs(defaults);
            let actual = ReasoningControl::from_template_kwargs(&request)
                .unwrap()
                .fallback(ReasoningControl::from_template_kwargs(&defaults).unwrap())
                .fallback(ReasoningControl::enabled("max"));
            assert_eq!(
                actual, expected,
                "request={request:?}, defaults={defaults:?}"
            );
        }
    }

    #[test]
    fn selected_toggle_requires_a_boolean() {
        for value in [
            json!("true"),
            json!(1),
            json!(null),
            json!({"type": "enabled"}),
        ] {
            assert!(
                ReasoningControl::from_template_kwargs(&kwargs(json!({
                    "thinking": value, "enable_thinking": true
                })))
                .unwrap_err()
                .is_request_validation_error()
            );
        }
    }

    #[test]
    fn invalid_effort_types_are_rejected_only_when_the_source_uses_them() {
        for value in [json!(true), json!(false), json!([]), json!({})] {
            for toggle in [None, Some(true), Some(false)] {
                let mut source = kwargs(json!({"reasoning_effort": value}));
                if let Some(toggle) = toggle {
                    source.insert("thinking".to_owned(), json!(toggle));
                }
                let control = ReasoningControl::from_template_kwargs(&source);
                if toggle == Some(false) {
                    assert_eq!(control.unwrap(), ReasoningControl::Disabled);
                } else {
                    assert!(control.unwrap_err().is_request_validation_error());
                }
            }

            // A typed request effort shadows malformed kwargs effort.
            let mut request = ChatRequest::for_test();
            request.chat_options.reasoning_effort = Some(EffortValue::from("high"));
            request.chat_options.template_kwargs = kwargs(json!({"reasoning_effort": value}));
            assert_eq!(
                ReasoningControl::from_request(&request).unwrap(),
                ReasoningControl::enabled("high")
            );
        }
    }
}
