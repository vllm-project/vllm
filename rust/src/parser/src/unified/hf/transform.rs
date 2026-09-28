// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Transform templates: restructure a parsed region value into the output shape.
//!
//! Port of `_apply_transform`, `validate_transform_strings`, and `process_field`
//! from `content_parsers.py`.

use std::sync::LazyLock;

use regex_automata::meta::Regex;
use serde_json::{Map, Value};

use super::{Result, invalid, value};

/// `\{(\w+(?:\.\w+)*)\}`: a whole-string placeholder such as `{content}` or
/// `{content.args}`.
static PLACEHOLDER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\w+(?:\.\w+)*\}").expect("valid regex"));

/// A compiled transform template.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Transform {
    Object(Vec<(String, Transform)>),
    Array(Vec<Transform>),
    /// A dotted placeholder path, e.g. `["content", "args"]`.
    Placeholder(Vec<String>),
    /// Any other JSON value, copied verbatim.
    Literal(Value),
}

impl Transform {
    /// Compile a transform template, rejecting any string that mixes a
    /// `{name}` placeholder with literal text.
    pub fn compile(scope: &str, template: &Value) -> Result<Self> {
        Ok(match template {
            Value::Object(object) => Self::Object(
                object
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), Self::compile(scope, value)?)))
                    .collect::<Result<_>>()?,
            ),
            Value::Array(values) => Self::Array(
                values.iter().map(|value| Self::compile(scope, value)).collect::<Result<_>>()?,
            ),
            Value::String(string) => match PLACEHOLDER.find(string) {
                Some(found) if found.range() == (0..string.len()) => Self::Placeholder(
                    string[1..string.len() - 1].split('.').map(str::to_string).collect(),
                ),
                Some(_) => {
                    return Err(invalid!(
                        "{scope}: transform string {string:?} mixes a {{placeholder}} with literal text. \
                         Use either a whole-string placeholder (e.g. \"{{content}}\") or a plain literal; \
                         string interpolation is not supported."
                    ));
                }
                None => Self::Literal(template.clone()),
            },
            other => Self::Literal(other.clone()),
        })
    }

    /// Return the placeholder path found at `key_path` inside this template, if any.
    pub fn placeholder_at(&self, key_path: &[&str]) -> Option<&[String]> {
        match (self, key_path) {
            (Self::Placeholder(path), []) => Some(path),
            (Self::Object(entries), [key, rest @ ..]) => entries
                .iter()
                .find(|(entry, _)| entry == key)
                .and_then(|(_, transform)| transform.placeholder_at(rest)),
            _ => None,
        }
    }

    /// Root names of every placeholder in this template.
    pub fn placeholder_roots(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        match self {
            Self::Object(entries) => {
                Box::new(entries.iter().flat_map(|(_, transform)| transform.placeholder_roots()))
            }
            Self::Array(transforms) => {
                Box::new(transforms.iter().flat_map(Self::placeholder_roots))
            }
            Self::Placeholder(path) => Box::new(path.first().map(String::as_str).into_iter()),
            Self::Literal(_) => Box::new(std::iter::empty()),
        }
    }

    /// Recursively instantiate this template against `scope`.
    pub fn apply(&self, scope: &Map<String, Value>) -> Result<Value> {
        Ok(match self {
            Self::Object(entries) => Value::Object(
                entries
                    .iter()
                    .map(|(key, transform)| Ok((key.clone(), transform.apply(scope)?)))
                    .collect::<Result<_>>()?,
            ),
            Self::Array(transforms) => Value::Array(
                transforms
                    .iter()
                    .map(|transform| transform.apply(scope))
                    .collect::<Result<_>>()?,
            ),
            Self::Literal(value) => value.clone(),
            Self::Placeholder(path) => {
                let dotted = path.join(".");
                let (root, keys) = path.split_first().expect("placeholder path is non-empty");
                let Some(mut value) = scope.get(root) else {
                    let available: Vec<_> = scope.keys().collect();
                    return Err(value!(
                        "transform placeholder '{{{dotted}}}' is not defined. Available: {available:?}"
                    ));
                };
                for key in keys {
                    let Some(object) = value.as_object() else {
                        return Err(value!(
                            "transform placeholder '{{{dotted}}}' cannot index into {} at '{key}'",
                            python_type_name(value)
                        ));
                    };
                    let Some(next) = object.get(key) else {
                        let available: Vec<_> = object.keys().collect();
                        return Err(value!(
                            "transform placeholder '{{{dotted}}}' is missing key '{key}'. Available: {available:?}"
                        ));
                    };
                    value = next;
                }
                value.clone()
            }
        })
    }
}

/// A field's transform: applied to the parsed value, or to each of its elements.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct FieldTransform {
    pub template: Transform,
    /// `transform_each`: the parsed content must be a list of dicts, and the
    /// template is applied to each element (with the element's keys unpacked into
    /// the template scope, alongside any regex captures).
    pub each: bool,
}

impl FieldTransform {
    /// Apply the transform to a parsed region value.
    pub fn apply(
        &self,
        field_name: &str,
        parsed: Value,
        captures: &[(String, String)],
    ) -> Result<Value> {
        let captures: Map<String, Value> = captures
            .iter()
            .map(|(name, text)| (name.clone(), Value::String(text.clone())))
            .collect();

        if !self.each {
            let mut scope = captures;
            scope.insert("content".to_string(), parsed);
            return self.template.apply(&scope);
        }

        let Value::Array(items) = parsed else {
            return Err(value!(
                "Field '{field_name}': transform_each requires the parsed content to be a list, got {}.",
                python_type_name(&parsed)
            ));
        };
        items
            .into_iter()
            .map(|item| {
                let Value::Object(item) = item else {
                    return Err(value!(
                        "Field '{field_name}': transform_each requires each list element to be a dict, got {}.",
                        python_type_name(&item)
                    ));
                };
                let mut scope = captures.clone();
                scope.extend(item);
                self.template.apply(&scope)
            })
            .collect::<Result<_>>()
            .map(Value::Array)
    }
}

/// Python type name of a JSON value, for error messages.
fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn scope(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn dotted_placeholders_descend_into_objects() {
        let transform = Transform::compile(
            "test",
            &json!({"type": "function", "function": {"name": "{content.name}", "arguments": "{content.args}"}}),
        )
        .unwrap();
        let value = transform
            .apply(&scope(json!({"content": {"name": "f", "args": {"a": 1}}})))
            .unwrap();
        assert_eq!(
            value,
            json!({"type": "function", "function": {"name": "f", "arguments": {"a": 1}}})
        );
        assert_eq!(
            transform.placeholder_at(&["function", "name"]),
            Some(["content".to_string(), "name".to_string()].as_slice())
        );
    }

    #[test]
    fn placeholder_errors_match_transformers() {
        let apply = |template: Value, scope_value: Value| {
            Transform::compile("test", &template)
                .unwrap()
                .apply(&scope(scope_value))
                .unwrap_err()
        };
        expect_test::expect![[r#"
            Value {
                message: "transform placeholder '{name}' is not defined. Available: [\"content\"]",
            }
        "#]]
        .assert_debug_eq(&apply(json!("{name}"), json!({"content": 1})));
        expect_test::expect![[r#"
            Value {
                message: "transform placeholder '{content.a}' cannot index into int at 'a'",
            }
        "#]]
        .assert_debug_eq(&apply(json!("{content.a}"), json!({"content": 1})));
        expect_test::expect![[r#"
            Value {
                message: "transform placeholder '{content.b}' is missing key 'b'. Available: [\"a\"]",
            }
        "#]]
        .assert_debug_eq(&apply(json!("{content.b}"), json!({"content": {"a": 1}})));
    }

    #[test]
    fn interpolation_is_rejected_at_load() {
        assert!(Transform::compile("test", &json!({"name": "fn_{name}"})).is_err());
        assert_eq!(
            Transform::compile("test", &json!("no placeholder")).unwrap(),
            Transform::Literal(json!("no placeholder"))
        );
    }
}
