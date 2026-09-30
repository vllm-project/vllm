// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Views of tool parameter JSON schemas.
//!
//! Tool parsers coerce generated parameter values by their schema, and output
//! grammars constrain the same values by it. Both read a parameter's schema
//! through [`SchemaView`], so they agree on which JSON types the value may take.

use std::borrow::Cow;
use std::collections::HashSet;

use serde_json::{Map, Value, json};

/// Bound on `$ref` and combinator nesting within one value, which keeps a long
/// chain of references from overflowing the stack.
const MAX_SCHEMA_DEPTH: usize = 32;

/// The schema accepting any value.
static ANY_SCHEMA: Value = Value::Bool(true);

/// A tool's parameters schema, which local `$ref`s resolve against.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SchemaView<'a> {
    root: &'a Value,
}

/// Reference targets expanded while collecting one value's options.
#[derive(Default)]
struct Expansion {
    expanded: HashSet<*const Value>,
    /// Whether a reference was skipped because its target already expanded.
    skipped: bool,
}

/// Views are equal when they read the same schema, which a pointer compare
/// decides without walking it.
impl PartialEq for SchemaView<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.root, other.root)
    }
}

impl Eq for SchemaView<'_> {}

/// One option of a value: a single JSON type under a narrowed schema, a
/// constant, or any value of the type.
#[derive(Debug, Clone)]
pub(crate) struct SchemaOption<'a> {
    pub(crate) ty: JsonType,
    pub(crate) source: OptionSource<'a>,
}

/// Where a [`SchemaOption`] comes from.
#[derive(Debug, Clone)]
pub(crate) enum OptionSource<'a> {
    /// A schema, of which the option takes the values of its type; [`narrow`]
    /// sets that type on it.
    Schema(&'a Map<String, Value>),
    /// A `const` or `enum` value.
    Literal(&'a Value),
    /// Any value of the option's type.
    Any,
}

/// JSON Schema type of a value option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonType {
    String,
    Integer,
    Number,
    Boolean,
    Null,
    Object,
    Array,
}

impl<'a> SchemaView<'a> {
    pub(crate) fn new(root: &'a Value) -> Self {
        Self { root }
    }

    /// The parameters schema itself.
    pub(crate) fn root(&self) -> &'a Value {
        self.root
    }

    /// Follow `$ref`s and single-schema `allOf`s. A reference that does not
    /// resolve, or a chain deeper than [`MAX_SCHEMA_DEPTH`], accepts any value.
    pub(crate) fn resolve(&self, schema: &'a Value) -> &'a Value {
        let mut schema = schema;
        for _ in 0..=MAX_SCHEMA_DEPTH {
            if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
                match self.resolve_ref(reference) {
                    Some(target) => schema = target,
                    None => return &ANY_SCHEMA,
                }
                continue;
            }
            match schema.get("allOf").and_then(Value::as_array).map(Vec::as_slice) {
                Some([inner]) => schema = inner,
                _ => return schema,
            }
        }
        &ANY_SCHEMA
    }

    /// Every option of a value under `schema`.
    ///
    /// `const` and `enum` values become literal options, and `anyOf`, `oneOf`,
    /// and type arrays one option per alternative, in schema order. An object
    /// or array schema without `type` is typed by its `properties`,
    /// `additionalProperties`, or `items`. `"nullable": true`, the OpenAPI
    /// spelling of an optional value, adds a `null` option. Type names accept
    /// the aliases that real tool schemas use, such as `int` or `dict`.
    ///
    /// Each reference target expands at most once. A union gains no values
    /// from a repeated alternative, and a definition shared by several
    /// alternatives would otherwise expand once per path to it, exponentially
    /// in the nesting. A schema that constrains nothing, or whose references
    /// only lead back into it, yields [`Self::any`].
    pub(crate) fn options(&self, schema: &'a Value) -> Vec<SchemaOption<'a>> {
        let mut expansion = Expansion::default();
        let mut options = self.options_at(schema, 0, &mut expansion);
        if options.is_empty() && expansion.skipped {
            options = Self::any();
        }
        let nullable = schema.get("nullable").and_then(Value::as_bool) == Some(true);
        if nullable && !options.iter().any(|option| option.ty == JsonType::Null) {
            options.push(SchemaOption {
                ty: JsonType::Null,
                source: OptionSource::Any,
            });
        }
        options
    }

    /// One option per JSON value type, each accepting any value of the type.
    pub(crate) fn any() -> Vec<SchemaOption<'a>> {
        JsonType::VALUE_TYPES
            .into_iter()
            .map(|ty| SchemaOption {
                ty,
                source: OptionSource::Any,
            })
            .collect()
    }

    fn resolve_ref(&self, reference: &str) -> Option<&'a Value> {
        self.root.pointer(reference.strip_prefix('#')?)
    }

    fn options_at(
        &self,
        schema: &'a Value,
        depth: usize,
        expansion: &mut Expansion,
    ) -> Vec<SchemaOption<'a>> {
        let schema = match schema {
            _ if depth > MAX_SCHEMA_DEPTH => return Self::any(),
            Value::Bool(false) => return vec![],
            Value::Object(schema) => schema,
            _ => return Self::any(),
        };
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            let Some(target) = self.resolve_ref(reference) else {
                return Self::any();
            };
            if !expansion.expanded.insert(std::ptr::from_ref(target)) {
                expansion.skipped = true;
                return vec![];
            }
            return self.options_at(target, depth + 1, expansion);
        }
        if let Some(value) = schema.get("const") {
            return vec![SchemaOption::literal(value)];
        }
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            return values.iter().map(SchemaOption::literal).collect();
        }
        if let Some(options) =
            schema.get("anyOf").or_else(|| schema.get("oneOf")).and_then(Value::as_array)
        {
            return options
                .iter()
                .flat_map(|option| self.options_at(option, depth + 1, expansion))
                .collect();
        }
        if let Some(schemas) = schema.get("allOf").and_then(Value::as_array) {
            return match schemas.as_slice() {
                [schema] => self.options_at(schema, depth + 1, expansion),
                _ => Self::any(),
            };
        }
        match schema.get("type") {
            Some(Value::String(name)) => SchemaOption::typed(schema, name).into_iter().collect(),
            Some(Value::Array(names)) => names
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|name| SchemaOption::typed(schema, name))
                .collect(),
            Some(_) => Self::any(),
            None if schema.contains_key("properties")
                || schema.contains_key("additionalProperties") =>
            {
                SchemaOption::typed(schema, JsonType::Object.name()).into_iter().collect()
            }
            None if schema.contains_key("items") => {
                SchemaOption::typed(schema, JsonType::Array.name()).into_iter().collect()
            }
            None => Self::any(),
        }
    }
}

impl<'a> SchemaOption<'a> {
    /// The schema of a [`OptionSource::Schema`] option.
    pub(crate) fn schema(&self) -> Option<&'a Map<String, Value>> {
        match self.source {
            OptionSource::Schema(schema) => Some(schema),
            OptionSource::Literal(_) | OptionSource::Any => None,
        }
    }

    /// The option of `schema` whose type is named `type_name`, or `None` for an
    /// unknown name.
    fn typed(schema: &'a Map<String, Value>, type_name: &str) -> Option<Self> {
        Some(Self {
            ty: JsonType::parse(type_name)?,
            source: OptionSource::Schema(schema),
        })
    }

    fn literal(value: &'a Value) -> Self {
        Self {
            ty: JsonType::of_value(value),
            source: OptionSource::Literal(value),
        }
    }
}

/// `schema` with its `type` set to the canonical name of `ty`, copied only
/// when that changes it.
pub(crate) fn narrow(schema: &Map<String, Value>, ty: JsonType) -> Cow<'_, Map<String, Value>> {
    if schema.get("type").and_then(Value::as_str) == Some(ty.name()) {
        Cow::Borrowed(schema)
    } else {
        let mut schema = schema.clone();
        schema.insert("type".to_string(), Value::String(ty.name().to_string()));
        Cow::Owned(schema)
    }
}

impl JsonType {
    /// The types of JSON values, which do not distinguish integers.
    const VALUE_TYPES: [Self; 6] = [
        Self::String,
        Self::Number,
        Self::Boolean,
        Self::Null,
        Self::Object,
        Self::Array,
    ];

    /// The JSON Schema type name.
    pub fn name(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Number => "number",
            Self::Boolean => "boolean",
            Self::Null => "null",
            Self::Object => "object",
            Self::Array => "array",
        }
    }

    /// Schema accepting every JSON value of this type.
    pub(crate) fn any_schema(self) -> Value {
        match self {
            Self::Object => json!({ "type": "object", "additionalProperties": true }),
            Self::Array => json!({ "type": "array", "items": true }),
            ty => json!({ "type": ty.name() }),
        }
    }

    /// Parse a schema type name, including aliases used by real tool schemas.
    fn parse(name: &str) -> Option<Self> {
        let name = name.trim().to_ascii_lowercase();
        Some(match name.as_str() {
            "string" | "str" | "text" | "varchar" | "char" | "enum" => Self::String,
            "integer" | "int" => Self::Integer,
            "number" | "float" | "double" => Self::Number,
            "boolean" | "bool" | "binary" => Self::Boolean,
            "object" | "dict" | "map" => Self::Object,
            "array" | "arr" | "list" | "sequence" => Self::Array,
            "null" => Self::Null,
            _ if name.starts_with("int")
                || name.starts_with("uint")
                || name.starts_with("long")
                || name.starts_with("short")
                || name.starts_with("unsigned") =>
            {
                Self::Integer
            }
            _ if name.starts_with("num") || name.starts_with("float") => Self::Number,
            _ if name.starts_with("dict") => Self::Object,
            _ if name.starts_with("list") => Self::Array,
            _ => return None,
        })
    }

    fn of_value(value: &Value) -> Self {
        match value {
            Value::String(_) => Self::String,
            Value::Number(number) if number.is_f64() => Self::Number,
            Value::Number(_) => Self::Integer,
            Value::Bool(_) => Self::Boolean,
            Value::Null => Self::Null,
            Value::Object(_) => Self::Object,
            Value::Array(_) => Self::Array,
        }
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use serde_json::json;

    use super::*;

    /// A Pydantic v2 `model_json_schema()` for a nested model, an enum, and an
    /// optional nested model.
    fn pydantic_schema() -> Value {
        json!({
            "$defs": {
                "Place": {
                    "properties": { "city": { "title": "City", "type": "string" } },
                    "required": ["city"],
                    "title": "Place",
                    "type": "object"
                },
                "Unit": { "enum": ["celsius", "fahrenheit"], "title": "Unit", "type": "string" }
            },
            "properties": {
                "place": { "$ref": "#/$defs/Place" },
                "unit": { "$ref": "#/$defs/Unit", "default": "celsius" },
                "backup": { "anyOf": [{ "$ref": "#/$defs/Place" }, { "type": "null" }], "default": null }
            },
            "required": ["place"],
            "title": "Args",
            "type": "object"
        })
    }

    fn check_options(root: &Value, schema: &Value, expected: Expect) {
        let options = SchemaView::new(root)
            .options(schema)
            .into_iter()
            .map(|option| match option.source {
                OptionSource::Schema(schema) => format!(
                    "{} {}",
                    option.ty.name(),
                    Value::Object(narrow(schema, option.ty).into_owned())
                ),
                OptionSource::Literal(value) => format!("{} = {value}", option.ty.name()),
                OptionSource::Any => format!("{} any", option.ty.name()),
            })
            .collect::<Vec<_>>();
        expected.assert_debug_eq(&options);
    }

    #[test]
    fn references_resolve_to_their_definitions() {
        let root = pydantic_schema();
        let properties = &root["properties"];
        check_options(
            &root,
            &properties["place"],
            expect![[r#"
            [
                "object {\"properties\":{\"city\":{\"title\":\"City\",\"type\":\"string\"}},\"required\":[\"city\"],\"title\":\"Place\",\"type\":\"object\"}",
            ]
        "#]],
        );
        check_options(
            &root,
            &properties["unit"],
            expect![[r#"
            [
                "string = \"celsius\"",
                "string = \"fahrenheit\"",
            ]
        "#]],
        );
        check_options(
            &root,
            &properties["backup"],
            expect![[r#"
            [
                "object {\"properties\":{\"city\":{\"title\":\"City\",\"type\":\"string\"}},\"required\":[\"city\"],\"title\":\"Place\",\"type\":\"object\"}",
                "null {\"type\":\"null\"}",
            ]
        "#]],
        );
    }

    #[test]
    fn unresolvable_and_cyclic_references_accept_any_value() {
        let root = json!({
            "$defs": { "a": { "$ref": "#/$defs/b" }, "b": { "$ref": "#/$defs/a" } },
            "properties": {
                "missing": { "$ref": "#/$defs/missing" },
                "cycle": { "$ref": "#/$defs/a" },
                "remote": { "$ref": "https://example.com/schema.json" }
            }
        });
        let view = SchemaView::new(&root);
        for key in ["missing", "cycle", "remote"] {
            assert_eq!(view.resolve(&root["properties"][key]), &ANY_SCHEMA, "{key}");
            assert!(
                view.options(&root["properties"][key])
                    .iter()
                    .all(|option| matches!(option.source, OptionSource::Any)),
                "{key}"
            );
        }
    }

    #[test]
    fn options_normalize_common_schema_spellings() {
        let root = json!({});
        check_options(
            &root,
            &json!({ "type": ["int", "null"], "minimum": 0 }),
            expect![[r#"
                [
                    "integer {\"type\":\"integer\",\"minimum\":0}",
                    "null {\"type\":\"null\",\"minimum\":0}",
                ]
            "#]],
        );
        check_options(
            &root,
            &json!({ "type": "string", "nullable": true }),
            expect![[r#"
                [
                    "string {\"type\":\"string\",\"nullable\":true}",
                    "null any",
                ]
            "#]],
        );
        check_options(
            &root,
            &json!({ "enum": ["a", 1, 1.5, null] }),
            expect![[r#"
                [
                    "string = \"a\"",
                    "integer = 1",
                    "number = 1.5",
                    "null = null",
                ]
            "#]],
        );
        check_options(
            &root,
            &json!({ "allOf": [{ "type": "boolean" }] }),
            expect![[r#"
                [
                    "boolean {\"type\":\"boolean\"}",
                ]
            "#]],
        );
        check_options(
            &root,
            &json!({ "items": { "type": "string" } }),
            expect![[r#"
                [
                    "array {\"items\":{\"type\":\"string\"},\"type\":\"array\"}",
                ]
            "#]],
        );
    }
}
