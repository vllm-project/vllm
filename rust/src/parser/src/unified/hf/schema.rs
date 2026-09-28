// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Serde mirror of the `response_template` JSON.
//!
//! Key sets and defaults follow `response_templates.py`; semantic validation
//! happens when compiling into [`super::ResponseTemplate`].

use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror_ext::AsReport as _;

use super::{Result, invalid};

/// Top-level `response_template` object.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TemplateSpec {
    #[serde(default = "default_version")]
    pub version: u64,
    /// Constant keys of the Transformers output dict (e.g. `role`). Unused by the
    /// event-based executor, but validated as part of the schema.
    #[serde(default, rename = "defaults")]
    _defaults: Map<String, Value>,
    /// Field specs in declaration order (`serde_json` preserves object order).
    pub fields: Map<String, Value>,
    pub start_anchor: Option<Literals>,
    pub start_anchor_pattern: Option<String>,
}

/// One entry of `fields`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FieldSpec {
    pub open: Option<Literals>,
    pub open_pattern: Option<String>,
    pub close: Option<Literals>,
    pub close_pattern: Option<String>,
    #[serde(default = "default_content")]
    pub content: String,
    #[serde(default)]
    pub content_args: Map<String, Value>,
    #[serde(default)]
    pub repeats: bool,
    pub join: Option<String>,
    #[serde(default = "default_optional")]
    pub optional: bool,
    pub transform: Option<Value>,
    #[serde(default)]
    pub transform_each: bool,
}

/// A literal anchor: one string, or a list of alternatives.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum Literals {
    One(String),
    Many(Vec<String>),
}

/// An anchor after resolving its literal-or-pattern form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum AnchorSpec<'a> {
    /// Deduplicated literals in first-seen order.
    Literals(Vec<String>),
    /// A regex in Python `regex` syntax.
    Pattern(&'a str),
}

impl TemplateSpec {
    /// Parse the top-level template object.
    pub fn from_value(value: &Value) -> Result<Self> {
        let spec: Self = serde_json::from_value(value.clone())
            .map_err(|error| invalid!("{}", error.as_report()))?;
        if spec.version != 1 {
            return Err(invalid!(
                "unsupported response_template version: {}",
                spec.version
            ));
        }
        if spec.fields.is_empty() {
            return Err(invalid!(
                "response_template.fields must be a non-empty dict"
            ));
        }
        Ok(spec)
    }

    /// Parse the field specs in declaration order.
    pub fn field_specs(&self) -> Result<Vec<(&str, FieldSpec)>> {
        self.fields
            .iter()
            .map(|(name, value)| {
                let spec = serde_json::from_value(value.clone())
                    .map_err(|error| invalid!("Field '{name}': {}", error.as_report()))?;
                Ok((name.as_str(), spec))
            })
            .collect()
    }
}

/// Resolve an anchor given in literal form (`literal_key`) or regex form (`pattern_key`).
///
/// Mirrors `_compile_anchor`: both forms at once is an error, literal lists must be
/// non-empty and contain no empty strings, and duplicates are dropped in first-seen
/// order.
pub(super) fn anchor_spec<'a>(
    scope: &str,
    literals: Option<&Literals>,
    pattern: Option<&'a str>,
    literal_key: &str,
    pattern_key: &str,
) -> Result<Option<AnchorSpec<'a>>> {
    match (literals, pattern) {
        (Some(_), Some(_)) => Err(invalid!(
            "{scope}: cannot specify both '{literal_key}' and '{pattern_key}'"
        )),
        (Some(literals), None) => {
            let raw = match literals {
                Literals::One(literal) => std::slice::from_ref(literal),
                Literals::Many(literals) => literals.as_slice(),
            };
            if raw.is_empty() {
                return Err(invalid!(
                    "{scope}: '{literal_key}' list must contain at least one literal"
                ));
            }
            if raw.iter().any(String::is_empty) {
                return Err(invalid!(
                    "{scope}: '{literal_key}' literals cannot be empty strings"
                ));
            }
            let mut deduped: Vec<String> = Vec::with_capacity(raw.len());
            for literal in raw {
                if !deduped.contains(literal) {
                    deduped.push(literal.clone());
                }
            }
            Ok(Some(AnchorSpec::Literals(deduped)))
        }
        (None, Some(pattern)) => Ok(Some(AnchorSpec::Pattern(pattern))),
        (None, None) => Ok(None),
    }
}

fn default_version() -> u64 {
    1
}

fn default_content() -> String {
    "text".to_string()
}

fn default_optional() -> bool {
    true
}
