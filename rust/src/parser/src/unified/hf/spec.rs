// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Serde mirror of the `response_template` JSON.
//!
//! Key sets and defaults follow `response_templates.py`; semantic validation
//! happens when compiling into [`super::ResponseTemplate`].

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use serde_with::serde_as;
use thiserror_ext::AsReport as _;

use super::{Result, invalid};

/// Top-level `response_template` object.
#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TemplateSpec {
    #[serde(default = "default_version")]
    version: u64,
    /// Constant keys of the Transformers output dict (e.g. `role`). Unused by the
    /// event-based executor, but validated as part of the schema.
    #[serde(default, rename = "defaults")]
    _defaults: Map<String, Value>,
    /// Field specs in declaration order.
    #[serde_as(as = "serde_with::Map<_, _>")]
    pub fields: Vec<(String, FieldSpec)>,
    start_anchor: Option<Literals>,
    start_anchor_pattern: Option<String>,
}

/// One entry of `fields`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FieldSpec {
    open: Option<Literals>,
    open_pattern: Option<String>,
    close: Option<Literals>,
    close_pattern: Option<String>,
    #[serde(default)]
    pub content: ContentKind,
    /// Arguments of the `content` parser; their keys depend on the parser.
    #[serde(default)]
    pub content_args: Map<String, Value>,
    #[serde(default)]
    pub repeats: bool,
    pub join: Option<String>,
    #[serde(default = "default_optional")]
    pub optional: bool,
    /// A free-form JSON template, compiled into a [`super::transform::Transform`].
    pub transform: Option<Value>,
    #[serde(default)]
    pub transform_each: bool,
}

/// Name of a content parser.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ContentKind {
    #[default]
    Text,
    Int,
    Float,
    Bool,
    Json,
    XmlInline,
    KvLines,
}

/// A literal anchor: one string, or a list of alternatives.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Literals {
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
        let spec: Self = deserialize(value, "response_template")?;
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

    /// The required start anchor.
    pub fn start_anchor(&self) -> Result<AnchorSpec<'_>> {
        anchor_spec(
            "response_template",
            (self.start_anchor.as_ref(), "start_anchor"),
            (self.start_anchor_pattern.as_deref(), "start_anchor_pattern"),
        )?
        .ok_or_else(|| {
            invalid!("response_template must define 'start_anchor' or 'start_anchor_pattern'.")
        })
    }
}

impl FieldSpec {
    /// The open anchor; `None` for the implicit field.
    pub fn open(&self, scope: &str) -> Result<Option<AnchorSpec<'_>>> {
        anchor_spec(
            scope,
            (self.open.as_ref(), "open"),
            (self.open_pattern.as_deref(), "open_pattern"),
        )
    }

    /// The close anchor; `None` when the field runs to the end of the stream.
    pub fn close(&self, scope: &str) -> Result<Option<AnchorSpec<'_>>> {
        anchor_spec(
            scope,
            (self.close.as_ref(), "close"),
            (self.close_pattern.as_deref(), "close_pattern"),
        )
    }
}

/// Deserialize `value`, reporting `context` and the JSON path of any error.
pub(super) fn deserialize<T: DeserializeOwned>(value: &Value, context: &str) -> Result<T> {
    serde_path_to_error::deserialize(value).map_err(|error| {
        let path = error.path().to_string();
        let error = error.into_inner();
        if path == "." {
            invalid!("{context}: {}", error.as_report())
        } else {
            invalid!("{context}.{path}: {}", error.as_report())
        }
    })
}

/// Resolve an anchor given in literal form or regex form.
///
/// Mirrors `_compile_anchor`: both forms at once is an error, literal lists must be
/// non-empty and contain no empty strings, and duplicates are dropped in first-seen
/// order.
fn anchor_spec<'a>(
    scope: &str,
    (literals, literal_key): (Option<&Literals>, &str),
    (pattern, pattern_key): (Option<&'a str>, &str),
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

fn default_optional() -> bool {
    true
}
