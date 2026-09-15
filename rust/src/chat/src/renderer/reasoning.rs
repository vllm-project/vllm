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

use serde::Serialize;
use serde_json::{Number, Value, json};

use crate::error::{Result, invalid_reasoning_control};
use crate::request::{ChatRequest, ReasoningEffort};

/// A native effort value, before or after model-specific mapping.
///
/// Strings preserve model-specific names; JSON numbers preserve integer and
/// floating-point representations for model-local range validation. Missing/null
/// standard effort is represented by `Option::None` in [`ReasoningControl`].
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub(super) enum EffortValue {
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
    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            Self::Number(_) => None,
        }
    }

    pub(super) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => value.as_f64(),
            Self::String(_) => None,
        }
    }
}

/// Reasoning intent for one source, or the result of falling back across sources.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum ReasoningControl {
    /// This source leaves the mode and effort to a lower-priority source.
    Default,
    /// Reasoning is explicitly disabled; lower-priority defaults preserve this mode.
    Disabled,
    /// Reasoning is enabled. Missing effort inherits from an enabled default.
    Enabled { effort: Option<EffortValue> },
}

impl ReasoningControl {
    /// Parse request kwargs and the typed OpenAI effort, which wins over kwargs effort.
    pub(super) fn from_request(request: &ChatRequest) -> Result<Self> {
        Self::from_kwargs(
            &request.chat_options.template_kwargs,
            request.chat_options.reasoning_effort,
        )
    }

    /// Parse a deployment source, whose controls all come from template kwargs.
    pub(super) fn from_template_kwargs(kwargs: &HashMap<String, Value>) -> Result<Self> {
        Self::from_kwargs(kwargs, None)
    }

    /// Select controls within one source, then derive its mode and optional effort.
    fn from_kwargs(
        kwargs: &HashMap<String, Value>,
        typed_effort: Option<ReasoningEffort>,
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
        // The typed OpenAI field wins over kwargs. A selected JSON null means
        // omitted effort; keep raw JSON until the toggle decides whether it is used.
        let effort = typed_effort
            .map(|effort| json!(effort.as_str()))
            .or_else(|| kwargs.get("reasoning_effort").cloned())
            .filter(|effort| !effort.is_null());

        match (toggle, effort) {
            // 1. Explicit false disables reasoning and discards any effort,
            //    including values that would fail validation in an enabled mode.
            (Some(false), _) => Ok(Self::Disabled),
            // 2. `none` disables reasoning when the toggle is absent. Explicit true
            //    enables reasoning and leaves effort to enabled deployment/model defaults.
            (toggle, Some(Value::String(effort))) if effort == "none" => {
                Ok(if toggle == Some(true) {
                    Self::Enabled { effort: None }
                } else {
                    Self::Disabled
                })
            }
            // 3. Absent toggle and absent/null effort leave the entire decision
            //    to the next source in the fallback chain.
            (None, None) => Ok(Self::Default),
            // 4. Explicit true or a concrete effort enables reasoning. Validate
            //    supplied effort as string/number; missing effort inherits later.
            (_, effort) => Ok(Self::Enabled {
                effort: effort.as_ref().map(EffortValue::try_from).transpose()?,
            }),
        }
    }

    /// Parse request and deployment independently, then apply request precedence.
    ///
    /// Each source validates its selected input types. Model adapters subsequently
    /// apply their model default and validate/map the resulting effective effort.
    pub(super) fn resolve(
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
    pub(super) fn fallback(self, defaults: Self) -> Self {
        match (self, defaults) {
            (Self::Default, defaults) => defaults,
            (Self::Enabled { effort: None }, Self::Enabled { effort }) => Self::Enabled { effort },
            (request, _) => request,
        }
    }

    /// Apply a model-native effort override within one source, before fallback.
    ///
    /// Disabled mode discards the override. Otherwise a supplied override enables
    /// reasoning and replaces this source's effort after string/number validation.
    /// The model adapter owns the override's names and numeric range; for example,
    /// K3 validates `thinking_effort` against its supported string grades.
    pub(super) fn with_effort(self, effort: Option<&Value>) -> Result<Self> {
        Ok(match (self, effort) {
            (Self::Disabled, _) => Self::Disabled,
            (_, Some(effort)) => Self::Enabled {
                effort: Some(EffortValue::try_from(effort)?),
            },
            (control, None) => control,
        })
    }

    /// Construct an enabled model default or a model-lowered concrete effort.
    pub(super) fn enabled(effort: impl Into<EffortValue>) -> Self {
        Self::Enabled {
            effort: Some(effort.into()),
        }
    }

    /// Report the resolved mode; callers apply model defaults before rendering.
    pub(super) fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }

    /// Return the active effort, which may still be missing before model fallback.
    pub(super) fn effort(&self) -> Option<&EffortValue> {
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
    pub(super) fn template_kwargs(&self, request: &ChatRequest) -> HashMap<String, Value> {
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
    fn explicit_toggle_decides_mode_and_none_inherits_enabled_effort() {
        let actual = [
            json!({}),
            json!({"reasoning_effort": "none"}),
            json!({"reasoning_effort": "high"}),
            json!({"enable_thinking": true}),
            json!({"enable_thinking": true, "reasoning_effort": "none"}),
            json!({"enable_thinking": true, "reasoning_effort": "low"}),
            json!({"enable_thinking": false, "reasoning_effort": "high"}),
            json!({"thinking": true, "enable_thinking": false}),
            json!({"thinking": false, "enable_thinking": true}),
            json!({"reasoning_effort": 37}),
            json!({"reasoning_effort": null}),
        ]
        .map(|value| ReasoningControl::from_template_kwargs(&kwargs(value)).unwrap());
        expect_test::expect![[r#"
            [
                Default,
                Disabled,
                Enabled {
                    effort: Some(
                        String(
                            "high",
                        ),
                    ),
                },
                Enabled {
                    effort: None,
                },
                Enabled {
                    effort: None,
                },
                Enabled {
                    effort: Some(
                        String(
                            "low",
                        ),
                    ),
                },
                Disabled,
                Enabled {
                    effort: None,
                },
                Disabled,
                Enabled {
                    effort: Some(
                        Number(
                            Number(37),
                        ),
                    ),
                },
                Default,
            ]
        "#]]
        .assert_debug_eq(&actual);
    }

    #[test]
    fn request_and_deployment_states_fall_back_before_model_defaults() {
        let sources = [
            (json!({}), json!({"enable_thinking": false})),
            (
                json!({"reasoning_effort": "high"}),
                json!({"thinking": false}),
            ),
            (
                json!({"enable_thinking": false}),
                json!({"reasoning_effort": "high"}),
            ),
            (
                json!({"thinking": true, "reasoning_effort": "none"}),
                json!({"reasoning_effort": "low"}),
            ),
            (
                json!({"thinking": true}),
                json!({"thinking": false, "reasoning_effort": "low"}),
            ),
            (json!({}), json!({})),
        ];
        let actual = sources.map(|(request, defaults)| {
            ReasoningControl::from_template_kwargs(&kwargs(request))
                .unwrap()
                .fallback(ReasoningControl::from_template_kwargs(&kwargs(defaults)).unwrap())
                .fallback(ReasoningControl::enabled("max"))
        });
        expect_test::expect![[r#"
            [
                Disabled,
                Enabled {
                    effort: Some(
                        String(
                            "high",
                        ),
                    ),
                },
                Disabled,
                Enabled {
                    effort: Some(
                        String(
                            "low",
                        ),
                    ),
                },
                Enabled {
                    effort: Some(
                        String(
                            "max",
                        ),
                    ),
                },
                Enabled {
                    effort: Some(
                        String(
                            "max",
                        ),
                    ),
                },
            ]
        "#]]
        .assert_debug_eq(&actual);
    }

    #[test]
    fn typed_effort_wins_over_kwargs_without_mutating_the_request() {
        let mut request = ChatRequest::for_test();
        request.chat_options.reasoning_effort = Some(ReasoningEffort::High);
        request.chat_options.template_kwargs = kwargs(json!({"reasoning_effort": 37}));
        let original = request.clone();
        let control =
            ReasoningControl::resolve(&request, &kwargs(json!({"thinking": false}))).unwrap();
        assert_eq!(control, ReasoningControl::enabled("high"));
        assert_eq!(request, original);
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
            request.chat_options.reasoning_effort = Some(ReasoningEffort::High);
            request.chat_options.template_kwargs = kwargs(json!({"reasoning_effort": value}));
            assert_eq!(
                ReasoningControl::from_request(&request).unwrap(),
                ReasoningControl::enabled("high")
            );

            // Model-native overrides follow the same disabled-mode short circuit.
            assert!(
                ReasoningControl::Default
                    .with_effort(Some(&value))
                    .unwrap_err()
                    .is_request_validation_error()
            );
            assert_eq!(
                ReasoningControl::Disabled.with_effort(Some(&value)).unwrap(),
                ReasoningControl::Disabled
            );
        }
    }
}
