// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use indexmap::IndexMap;
use minijinja::value::{Enumerator, Object, ObjectExt, ObjectRepr};
use minijinja::{Error as TemplateError, ErrorKind as TemplateErrorKind, State};
use serde::Serialize;
use serde_json::Value as JsonValue;

/// A wrapper around `minijinja::Value` that can be constructed with `to_template_value` and used
/// as a value in the chat template.
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub(super) struct TemplateValue(minijinja::Value);

pub(super) fn to_template_value(value: JsonValue) -> TemplateValue {
    TemplateValue(match value {
        JsonValue::Array(values) => values
            .into_iter()
            .map(to_template_value)
            .map(|value| value.0)
            .collect::<minijinja::Value>(),
        JsonValue::Object(values) => minijinja::Value::from_object(TemplateMap(
            values
                .into_iter()
                .map(|(key, value)| (key, to_template_value(value).0))
                .collect(),
        )),
        // For primitive values, directly convert them to `minijinja::Value` using `from_serialize`.
        value => minijinja::Value::from_serialize(value),
    })
}

/// A custom map type that always returns `UnknownMethod` for method calls, so that pycompat can
/// always handle dict methods through the unknown-method callback.
///
/// Use `IndexMap` to preserve the original key order when iterating.
///
/// MiniJinja's default map can resolve a same-named field before Python dict methods. HF templates
/// commonly call `dict.items()`, which would fail if the map had an `items` field.
/// See issue: https://github.com/mitsuhiko/minijinja/issues/903
#[derive(Debug)]
struct TemplateMap(IndexMap<String, minijinja::Value>);

impl Object for TemplateMap {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value(self: &Arc<Self>, key: &minijinja::Value) -> Option<minijinja::Value> {
        self.0.get(key.as_str()?).cloned()
    }

    fn get_value_by_str(self: &Arc<Self>, key: &str) -> Option<minijinja::Value> {
        self.0.get(key).cloned()
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        self.mapped_rev_enumerator(|this| {
            Box::new(this.0.keys().map(|key| minijinja::Value::from(key.as_str())))
        })
    }

    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        Some(self.0.len())
    }

    fn call_method(
        self: &Arc<Self>,
        _state: &State<'_, '_>,
        _method: &str,
        _args: &[minijinja::Value],
    ) -> std::result::Result<minijinja::Value, TemplateError> {
        // Always return `UnknownMethod` for method calls,
        // so that pycompat can handle dict methods through the unknown-method callback.
        Err(TemplateError::from(TemplateErrorKind::UnknownMethod))
    }
}
