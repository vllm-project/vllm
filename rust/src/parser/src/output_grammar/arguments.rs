// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Tool-call argument grammars for keyed-parameter protocols.
//!
//! Many tool-call protocols render the root arguments object as a sequence of
//! keyed parameters. [`arguments`] owns what follows from the JSON schema
//! alone: which parameters exist, their order and presence, and every option
//! a parameter value can take, with `$ref`s resolved and unions, type arrays,
//! and enums split per option. An [`ArgumentSyntax`], implemented next to each
//! parser, decides how one parameter and its value options are rendered.
//! Values nested below a parameter stay with XGrammar's JSON schema converter
//! through [`ValueOption::json`], unless the syntax renders nested objects as
//! keyed parameters too, through [`ValueOption::fields`] and
//! [`ValueOption::items`].
//!
//! This is for protocols whose argument rendering XGrammar's per-model schema
//! styles do not cover, or cover without matching the parser. Parameter lists
//! follow those styles' fixed-order property semantics, with these deliberate
//! differences:
//!
//! - Unions, type arrays, and mixed enums split into one option per
//!   alternative, so a syntax can keep a value's type label consistent with
//!   the value.
//! - Additional-property keys are not required to differ from declared keys.
//! - String `format` and length constraints are not enforced. `pattern` is.
//! - `minProperties` and `maxProperties` are not enforced.
//! - `allOf` with more than one schema accepts any value.

use serde_json::{Map, Value, json};
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat};

pub use crate::schema::JsonType;
use crate::schema::{self, OptionSource, SchemaOption, SchemaRoot};

/// The schema accepting any value.
static ANY_SCHEMA: Value = Value::Bool(true);

/// Bound on the object and array levels a syntax renders through
/// [`ValueOption::fields`] and [`ValueOption::items`], which keeps a deep schema
/// from overflowing the stack.
const MAX_NESTING_DEPTH: usize = 32;

/// A schema object.
type SchemaMap<'a> = &'a Map<String, Value>;

/// How one protocol renders the parameters of a call.
pub trait ArgumentSyntax {
    /// Text around and between parameters, or `None` when parameters are
    /// adjacent.
    fn separator(&self) -> Option<Format>;

    /// The grammar of one parameter whose value takes one of `options`, or
    /// `None` when none of them can be rendered. A syntax that renders no
    /// [`ParameterKey::Free`] parameter cannot render objects that admit
    /// undeclared keys; see [`arguments`].
    fn parameter(&self, key: ParameterKey<'_>, options: &[ValueOption<'_>]) -> Option<Format>;
}

/// The key of one parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterKey<'a> {
    /// A property declared by the schema.
    Declared(&'a str),
    /// Any key allowed by `additionalProperties` or an unconstrained schema.
    Free,
}

impl ParameterKey<'_> {
    /// Pattern of free keys, as in XGrammar's XML styles.
    pub const FREE_PATTERN: &'static str = "[a-zA-Z_][a-zA-Z0-9_]*";

    /// `prefix KEY suffix content end`: a tag for a declared key, or a
    /// sequence matching [`Self::FREE_PATTERN`] for a free key.
    pub fn tag(&self, prefix: &str, suffix: &str, content: Format, end: &str) -> Format {
        match self {
            Self::Declared(key) => Format::tag(format!("{prefix}{key}{suffix}"), content, end),
            Self::Free => Format::sequence(vec![
                Format::const_string(prefix),
                Format::regex(Self::FREE_PATTERN),
                Format::const_string(suffix),
                content,
                Format::const_string(end),
            ]),
        }
    }
}

/// One option of a parameter value: a single JSON type under a narrowed
/// schema, a constant, or any value of the type.
pub struct ValueOption<'a> {
    /// The option's JSON Schema type.
    pub ty: JsonType,
    source: OptionSource<'a>,
    cx: &'a ArgumentContext<'a>,
    /// Object and array schemas rendered around this value, outermost first.
    enclosing: Vec<SchemaMap<'a>>,
}

impl<'a> ValueOption<'a> {
    /// The option as JSON text.
    pub fn json(&self) -> Format {
        match &self.source {
            OptionSource::Schema(schema) => {
                self.cx.json(Value::Object(schema::narrow(schema, self.ty).into_owned()))
            }
            OptionSource::Literal(value) => self.cx.json(json!({ "const": value })),
            OptionSource::Any => self.cx.json(self.ty.any_schema()),
        }
    }

    /// The option as raw text that ends before any of `terminators`, or `None`
    /// for a non-string option or a constant containing a terminator or an
    /// excluded substring.
    pub fn raw_string(&self, terminators: &[&str]) -> Option<Format> {
        if self.ty != JsonType::String {
            return None;
        }
        let excludes = self
            .cx
            .options
            .excludes
            .iter()
            .map(String::as_str)
            .chain(terminators.iter().copied())
            .collect::<Vec<_>>();
        let text = || Format::any_text_excluding(&excludes);
        Some(match &self.source {
            OptionSource::Literal(Value::String(value)) => {
                if excludes.iter().any(|exclude| value.contains(exclude)) {
                    return None;
                }
                Format::const_string(value)
            }
            OptionSource::Literal(_) => unreachable!("literal options are typed by their value"),
            OptionSource::Schema(schema) => match schema.get("pattern").and_then(Value::as_str) {
                Some(pattern) => Format::regex(pattern),
                None => text(),
            },
            OptionSource::Any => text(),
        })
    }

    /// The fields of an object option as keyed parameters rendered by
    /// `syntax`, like the root arguments.
    ///
    /// `None` for a non-object option, a constant, an object of any shape, an
    /// object that admits keys `syntax` cannot render, or one that
    /// [`Self::recurses`]; a syntax then needs a fallback for the value.
    // TODO: render constant objects as fixed fields.
    pub fn fields(&self, syntax: &dyn ArgumentSyntax) -> Option<Format> {
        if self.ty != JsonType::Object {
            return None;
        }
        let (schema, enclosing) = self.nested()?;
        self.cx.parameters(schema, syntax, &enclosing)
    }

    /// The options of an array option's items.
    ///
    /// `None` for a non-array option, a constant, an array of any items, or one
    /// that [`Self::recurses`]. Every item takes the same options.
    // TODO: honor `prefixItems`.
    pub fn items(&self) -> Option<Vec<ValueOption<'a>>> {
        if self.ty != JsonType::Array {
            return None;
        }
        let (schema, enclosing) = self.nested()?;
        let items = schema.get("items").unwrap_or(&ANY_SCHEMA);
        Some(self.cx.value_options(items, &enclosing))
    }

    /// Whether the option's children cannot be rendered by nesting: its
    /// schema encloses itself through references, or sits
    /// [`MAX_NESTING_DEPTH`] levels deep.
    pub fn recurses(&self) -> bool {
        let OptionSource::Schema(schema) = self.source else {
            return false;
        };
        self.enclosing.len() >= MAX_NESTING_DEPTH
            || self.enclosing.iter().any(|outer| std::ptr::eq(*outer, schema))
    }

    /// The option's schema and the enclosing schemas of its children, or
    /// `None` when the option has no schema or [`Self::recurses`].
    fn nested(&self) -> Option<(SchemaMap<'a>, Vec<SchemaMap<'a>>)> {
        let OptionSource::Schema(schema) = self.source else {
            return None;
        };
        if self.recurses() {
            return None;
        }
        let mut enclosing = self.enclosing.clone();
        enclosing.push(schema);
        Some((schema, enclosing))
    }
}

/// Request options that also apply to nested JSON schema regions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArgumentOptions {
    /// Whether object properties may appear in any order, at the parameter
    /// level and in nested objects.
    pub any_order: bool,
    /// Maximum consecutive whitespace characters, or no limit when unset.
    pub max_whitespace_cnt: Option<i32>,
    /// Substrings forbidden in string values, nested JSON strings included.
    pub excludes: Vec<String>,
}

/// Request state shared by the value options of one call.
struct ArgumentContext<'a> {
    options: &'a ArgumentOptions,
    root: SchemaRoot<'a>,
}

impl<'a> ArgumentContext<'a> {
    /// A JSON value under `schema`, which keeps the root's definitions so its
    /// local `$ref`s still resolve.
    fn json(&self, mut schema: Value) -> Format {
        if let (Some(schema), Some(root)) = (schema.as_object_mut(), self.root.schema().as_object())
        {
            for key in ["$defs", "definitions"] {
                if let Some(definitions) = root.get(key) {
                    schema.entry(key).or_insert_with(|| definitions.clone());
                }
            }
        }
        Format::JsonSchema(JsonSchemaFormat {
            excludes: self.options.excludes.clone(),
            ..JsonSchemaFormat::new(schema)
                .with_any_order(self.options.any_order)
                .with_max_whitespace_cnt(self.options.max_whitespace_cnt)
        })
    }

    /// Every option of a value under `schema`, nested in `enclosing`.
    fn value_options(
        &'a self,
        schema: &'a Value,
        enclosing: &[SchemaMap<'a>],
    ) -> Vec<ValueOption<'a>> {
        self.root
            .options(schema)
            .into_iter()
            .map(|SchemaOption { ty, source }| ValueOption {
                ty,
                source,
                cx: self,
                enclosing: enclosing.to_vec(),
            })
            .collect()
    }

    /// The keyed parameters of an object under `schema`, nested in
    /// `enclosing`, or `None` when it admits undeclared keys that `syntax`
    /// cannot render.
    fn parameters(
        &'a self,
        schema: SchemaMap<'a>,
        syntax: &dyn ArgumentSyntax,
        enclosing: &[SchemaMap<'a>],
    ) -> Option<Format> {
        let parameter = |key: ParameterKey<'_>, schema: &'a Value| {
            syntax.parameter(key, &self.value_options(schema, enclosing))
        };
        let undeclared = match schema.get("additionalProperties") {
            Some(Value::Bool(false)) => None,
            Some(additional) => Some(additional),
            // Strict schemas allow no undeclared properties, and a schema
            // without declared properties allows any.
            None if schema.contains_key("properties") => None,
            None => Some(&ANY_SCHEMA),
        };
        let additional = match undeclared {
            Some(additional) => Some(parameter(ParameterKey::Free, additional)?),
            None => None,
        };
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|required| required.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .filter_map(|(key, schema)| {
                let parameter = parameter(ParameterKey::Declared(key), schema)?;
                Some((parameter, required.contains(&key.as_str())))
            })
            .collect();
        Some(parameter_list(
            self.options,
            syntax.separator(),
            properties,
            additional,
        ))
    }
}

/// Build the grammar of one call's arguments under `schema`, the tool's
/// parameters schema, rendering each parameter with `syntax`, or `None` when
/// the arguments admit undeclared keys that `syntax` cannot render.
pub fn arguments(
    schema: &Value,
    syntax: &dyn ArgumentSyntax,
    options: &ArgumentOptions,
) -> Option<Format> {
    let cx = ArgumentContext {
        options,
        root: SchemaRoot::new(schema),
    };
    match cx.root.resolved() {
        Value::Object(schema) => cx.parameters(schema, syntax, &[]),
        Value::Bool(false) => Some(parameter_list(options, syntax.separator(), vec![], None)),
        _ => {
            let options = cx.value_options(&ANY_SCHEMA, &[]);
            let additional = syntax.parameter(ParameterKey::Free, &options)?;
            Some(parameter_list(
                cx.options,
                syntax.separator(),
                vec![],
                Some(additional),
            ))
        }
    }
}

/// Parameters in declared order, required ones mandatory, followed by any
/// number of additional parameters; or, under `any_order`, any sequence of
/// them with at least one when some are required.
fn parameter_list(
    options: &ArgumentOptions,
    separator: Option<Format>,
    properties: Vec<(Format, bool)>,
    additional: Option<Format>,
) -> Format {
    let separated = |parameter: Format| match &separator {
        Some(separator) => Format::sequence(vec![parameter, separator.clone()]),
        None => parameter,
    };
    let mut elements = Vec::from_iter(separator.clone());
    if options.any_order {
        let any_required = properties.iter().any(|(_, required)| *required);
        let parameters = properties
            .into_iter()
            .map(|(parameter, _)| parameter)
            .chain(additional)
            .collect::<Vec<_>>();
        if !parameters.is_empty() {
            let parameter = separated(one_of(parameters));
            elements.push(if any_required {
                Format::plus(parameter)
            } else {
                Format::star(parameter)
            });
        }
    } else {
        for (parameter, required) in properties {
            let parameter = separated(parameter);
            elements.push(if required {
                parameter
            } else {
                Format::optional(parameter)
            });
        }
        if let Some(additional) = additional {
            elements.push(Format::star(separated(additional)));
        }
    }
    match elements.len() {
        0 => Format::const_string(""),
        1 => elements.pop().expect("one element"),
        _ => Format::sequence(elements),
    }
}

/// One of `formats`, without an `or` around a single format.
pub fn one_of(mut formats: Vec<Format>) -> Format {
    if formats.len() == 1 {
        formats.pop().expect("one format")
    } else {
        Format::or(formats)
    }
}

/// `options` grouped by `key`, in order of first appearance.
pub fn group_by<'o, 'a, K: PartialEq>(
    options: &'o [ValueOption<'a>],
    key: impl Fn(&ValueOption<'a>) -> K,
) -> Vec<(K, Vec<&'o ValueOption<'a>>)> {
    let mut groups: Vec<(K, Vec<&ValueOption<'a>>)> = Vec::new();
    for option in options {
        let key = key(option);
        match groups.iter_mut().find(|(existing, _)| *existing == key) {
            Some((_, group)) => group.push(option),
            None => groups.push((key, vec![option])),
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::*;
    use crate::output_grammar::test_utils::outline;
    use expect_test::{Expect, expect};
    use serde_json::json;

    /// Qwen-like: `<parameter=KEY>VALUE</parameter>` on separate lines, with
    /// raw strings.
    struct Untyped;

    impl ArgumentSyntax for Untyped {
        fn separator(&self) -> Option<Format> {
            Some(Format::const_string("\n"))
        }

        fn parameter(&self, key: ParameterKey<'_>, options: &[ValueOption<'_>]) -> Option<Format> {
            let values = options
                .iter()
                .filter_map(|option| match option.ty {
                    JsonType::String => option.raw_string(&["</parameter>"]),
                    _ => Some(option.json()),
                })
                .collect::<Vec<_>>();
            (!values.is_empty())
                .then(|| key.tag("<parameter=", ">", one_of(values), "</parameter>"))
        }
    }

    /// K3-like: `<arg key="KEY" type="TYPE">VALUE</arg>` with adjacent
    /// parameters and one header per JSON type.
    struct Typed;

    impl ArgumentSyntax for Typed {
        fn separator(&self) -> Option<Format> {
            None
        }

        fn parameter(&self, key: ParameterKey<'_>, options: &[ValueOption<'_>]) -> Option<Format> {
            let tags = group_by(options, |option| option.ty)
                .into_iter()
                .filter_map(|(ty, options)| {
                    let values = options
                        .into_iter()
                        .filter_map(|option| match ty {
                            JsonType::String => option.raw_string(&["</arg>"]),
                            _ => Some(option.json()),
                        })
                        .collect::<Vec<_>>();
                    (!values.is_empty()).then(|| {
                        let suffix = format!("\" type=\"{}\">", ty.name());
                        key.tag("<arg key=\"", &suffix, one_of(values), "</arg>")
                    })
                })
                .collect::<Vec<_>>();
            (!tags.is_empty()).then(|| one_of(tags))
        }
    }

    /// M3-like: `<KEY>VALUE</KEY>` with objects as nested fields and arrays as
    /// `<item>` elements, and JSON where a value cannot nest.
    struct Nested;

    impl ArgumentSyntax for Nested {
        fn separator(&self) -> Option<Format> {
            None
        }

        fn parameter(&self, key: ParameterKey<'_>, options: &[ValueOption<'_>]) -> Option<Format> {
            let ParameterKey::Declared(key) = key else {
                return None;
            };
            let values = options.iter().map(nested_value).collect();
            Some(Format::tag(
                format!("<{key}>"),
                one_of(values),
                format!("</{key}>"),
            ))
        }
    }

    fn nested_value(option: &ValueOption<'_>) -> Format {
        let nested = match option.ty {
            JsonType::Object => option.fields(&Nested),
            JsonType::Array => option.items().map(|items| {
                let values = items.iter().map(nested_value).collect();
                Format::star(Format::tag("<item>", one_of(values), "</item>"))
            }),
            _ => None,
        };
        nested.unwrap_or_else(|| option.json())
    }

    fn check(schema: Value, syntax: &dyn ArgumentSyntax, expected: Expect) {
        check_with(schema, syntax, &ArgumentOptions::default(), expected);
    }

    fn check_with(
        schema: Value,
        syntax: &dyn ArgumentSyntax,
        options: &ArgumentOptions,
        expected: Expect,
    ) {
        let outline = match arguments(&schema, syntax, options) {
            Some(arguments) => outline(&arguments),
            None => "none\n".to_string(),
        };
        expected.assert_eq(&outline);
    }

    fn weather_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" },
                "unit": { "enum": ["celsius", "fahrenheit"] },
                "days": { "type": "integer", "minimum": 1 }
            },
            "required": ["location"]
        })
    }

    #[test]
    fn parameters_follow_declared_order() {
        check(
            weather_schema(),
            &Untyped,
            expect![[r#"
                sequence
                  `\n`
                  sequence
                    tag `<parameter=location>` text excluding [`</parameter>`] `</parameter>`
                    `\n`
                  optional
                    sequence
                      tag `<parameter=unit>` .. `</parameter>`
                        or
                          `celsius`
                          `fahrenheit`
                      `\n`
                  optional
                    sequence
                      tag `<parameter=days>` json(integer(minimum=1)) `</parameter>`
                      `\n`
            "#]],
        );
    }

    #[test]
    fn any_order_repeats_any_declared_parameter() {
        check_with(
            weather_schema(),
            &Typed,
            &ArgumentOptions {
                any_order: true,
                ..Default::default()
            },
            expect![[r#"
                plus
                  or
                    tag `<arg key="location" type="string">` text excluding [`</arg>`] `</arg>`
                    tag `<arg key="unit" type="string">` .. `</arg>`
                      or
                        `celsius`
                        `fahrenheit`
                    tag `<arg key="days" type="integer">` json(integer(minimum=1)) any_order `</arg>`
            "#]],
        );
    }

    #[test]
    fn unions_and_mixed_enums_split_per_option() {
        check(
            json!({
                "type": "object",
                "properties": {
                    "id": { "type": ["integer", "string", "null"] },
                    "mode": { "enum": ["fast", 1] },
                    "q": { "anyOf": [{ "type": "string", "pattern": "[a-z]+" }, { "type": "array", "items": { "type": "string" } }] }
                },
                "required": ["q"]
            }),
            &Typed,
            expect![[r#"
                sequence
                  optional
                    or
                      tag `<arg key="id" type="integer">` json(integer) `</arg>`
                      tag `<arg key="id" type="string">` text excluding [`</arg>`] `</arg>`
                      tag `<arg key="id" type="null">` json(null) `</arg>`
                  optional
                    or
                      tag `<arg key="mode" type="string">` `fast` `</arg>`
                      tag `<arg key="mode" type="integer">` json(1) `</arg>`
                  or
                    tag `<arg key="q" type="string">` /[a-z]+/ `</arg>`
                    tag `<arg key="q" type="array">` json(string[]) `</arg>`
            "#]],
        );
    }

    #[test]
    fn references_resolve_against_the_root_definitions() {
        let schema = json!({
            "$defs": {
                "place": { "type": "object", "properties": { "city": { "type": "string" } } },
                "name": { "type": "string" }
            },
            "type": "object",
            "properties": {
                "place": { "$ref": "#/$defs/place" },
                "name": { "$ref": "#/$defs/name" }
            },
            "required": ["place", "name"]
        });
        let arguments = arguments(&schema, &Typed, &ArgumentOptions::default()).unwrap();
        expect![[r#"
            sequence
              tag `<arg key="place" type="object">` json({ city?: string } where place = { city?: string }, name = string) `</arg>`
              tag `<arg key="name" type="string">` text excluding [`</arg>`] `</arg>`
        "#]].assert_eq(&outline(&arguments));

        // Nested JSON regions keep the definitions for their own references.
        let serialized = serde_json::to_value(&arguments).unwrap();
        assert_eq!(
            serialized["elements"][0]["content"]["json_schema"]["$defs"],
            schema["$defs"]
        );
    }

    #[test]
    fn shared_definitions_expand_once() {
        // Each level unions four references to the next, so expanding every
        // path would yield 4^12 options.
        let mut definitions = Map::new();
        for level in 0..12 {
            let next = json!({ "$ref": format!("#/$defs/level{}", level + 1) });
            definitions.insert(format!("level{level}"), json!({ "anyOf": vec![next; 4] }));
        }
        definitions.insert("level12".to_string(), json!({ "type": "string" }));
        check(
            json!({
                "$defs": definitions,
                "type": "object",
                "properties": { "p": { "$ref": "#/$defs/level0" } },
                "required": ["p"]
            }),
            &Untyped,
            expect![[r#"
                sequence
                  `\n`
                  sequence
                    tag `<parameter=p>` text excluding [`</parameter>`] `</parameter>`
                    `\n`
            "#]],
        );
    }

    #[test]
    fn references_that_only_cycle_accept_any_value() {
        check(
            json!({
                "$defs": {
                    "a": { "anyOf": [{ "$ref": "#/$defs/b" }] },
                    "b": { "$ref": "#/$defs/a" }
                },
                "type": "object",
                "properties": { "p": { "$ref": "#/$defs/a" } },
                "required": ["p"]
            }),
            &Untyped,
            expect![[r#"
                sequence
                  `\n`
                  sequence
                    tag `<parameter=p>` .. `</parameter>`
                      or
                        text excluding [`</parameter>`]
                        json(number where a = b, b = a)
                        json(boolean where a = b, b = a)
                        json(null where a = b, b = a)
                        json({ ... } where a = b, b = a)
                        json(any[] where a = b, b = a)
                    `\n`
            "#]],
        );
    }

    #[test]
    fn unconstrained_arguments_accept_any_parameter() {
        check(
            json!(true),
            &Typed,
            expect![[r#"
                star
                  or
                    sequence
                      `<arg key="`
                      /[a-zA-Z_][a-zA-Z0-9_]*/
                      `" type="string">`
                      text excluding [`</arg>`]
                      `</arg>`
                    sequence
                      `<arg key="`
                      /[a-zA-Z_][a-zA-Z0-9_]*/
                      `" type="number">`
                      json(number)
                      `</arg>`
                    sequence
                      `<arg key="`
                      /[a-zA-Z_][a-zA-Z0-9_]*/
                      `" type="boolean">`
                      json(boolean)
                      `</arg>`
                    sequence
                      `<arg key="`
                      /[a-zA-Z_][a-zA-Z0-9_]*/
                      `" type="null">`
                      json(null)
                      `</arg>`
                    sequence
                      `<arg key="`
                      /[a-zA-Z_][a-zA-Z0-9_]*/
                      `" type="object">`
                      json({ ... })
                      `</arg>`
                    sequence
                      `<arg key="`
                      /[a-zA-Z_][a-zA-Z0-9_]*/
                      `" type="array">`
                      json(any[])
                      `</arg>`
            "#]],
        );
        check(
            json!(false),
            &Untyped,
            expect![[r#"
                `\n`
            "#]],
        );
    }

    #[test]
    fn additional_properties_follow_declared_ones() {
        check(
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
                "additionalProperties": { "type": "boolean" }
            }),
            &Untyped,
            expect![[r#"
                sequence
                  `\n`
                  sequence
                    tag `<parameter=path>` text excluding [`</parameter>`] `</parameter>`
                    `\n`
                  star
                    sequence
                      sequence
                        `<parameter=`
                        /[a-zA-Z_][a-zA-Z0-9_]*/
                        `>`
                        json(boolean)
                        `</parameter>`
                      `\n`
            "#]],
        );
    }

    #[test]
    fn raw_literals_exclude_terminators_and_request_exclusions() {
        check_with(
            json!({
                "type": "object",
                "properties": { "tag": { "enum": ["ok", "a</call>b", "c</arg>d"] } },
                "required": ["tag"]
            }),
            &Typed,
            &ArgumentOptions {
                excludes: vec!["</call>".to_string()],
                ..Default::default()
            },
            expect![[r#"
                tag `<arg key="tag" type="string">` `ok` `</arg>`
            "#]],
        );
    }

    #[test]
    fn nested_objects_and_arrays_render_through_the_syntax() {
        check(
            json!({
                "type": "object",
                "properties": {
                    "place": {
                        "type": "object",
                        "properties": { "city": { "type": "string" }, "zip": { "type": "integer" } },
                        "required": ["city"]
                    },
                    "stops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": { "name": { "type": "string" } },
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["place"]
            }),
            &Nested,
            expect![[r#"
                sequence
                  tag `<place>` .. `</place>`
                    sequence
                      tag `<city>` json(string) `</city>`
                      optional tag `<zip>` json(integer) `</zip>`
                  optional
                    tag `<stops>` .. `</stops>`
                      star
                        tag `<item>` .. `</item>`
                          optional tag `<name>` json(string) `</name>`
            "#]],
        );
    }

    #[test]
    fn nesting_stops_at_recursion_and_undeclared_keys() {
        check(
            json!({
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "label": { "type": "string" },
                            "children": { "type": "array", "items": { "$ref": "#/$defs/node" } }
                        },
                        "additionalProperties": false
                    }
                },
                "type": "object",
                "properties": {
                    "tree": { "$ref": "#/$defs/node" },
                    "labels": { "type": "object", "additionalProperties": { "type": "string" } },
                    "extra": { "type": "object" }
                },
                "additionalProperties": false
            }),
            &Nested,
            expect![[r#"
                sequence
                  optional
                    tag `<tree>` .. `</tree>`
                      sequence
                        optional tag `<label>` json(string where node = { label?: string, children?: node[] }) `</label>`
                        optional
                          tag `<children>` .. `</children>`
                            star tag `<item>` json({ label?: string, children?: node[] } where node = { label?: string, children?: node[] }) `</item>`
                  optional tag `<labels>` json({ ...: string } where node = { label?: string, children?: node[] }) `</labels>`
                  optional tag `<extra>` json(object where node = { label?: string, children?: node[] }) `</extra>`
            "#]],
        );
    }

    #[test]
    fn undeclared_keys_without_free_parameters_cannot_render() {
        check(
            json!(true),
            &Nested,
            expect![[r#"
            none
        "#]],
        );
        check(
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "additionalProperties": { "type": "boolean" }
            }),
            &Nested,
            expect![[r#"
                none
            "#]],
        );
    }
}
