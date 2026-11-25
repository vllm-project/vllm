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
//! through [`ValueOption::json`].
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
use xgrammar_structural_tag::format::{EndBoundary, Format, JsonSchemaFormat, TokenFormat};

use super::{OutputGrammarError, Result};
pub use crate::schema::JsonType;
use crate::schema::{self, OptionSource, SchemaOption, SchemaRoot};

/// The schema accepting any value.
static ANY_SCHEMA: Value = Value::Bool(true);

const MAX_TOOL_GRAMMAR_WORK: usize = 8192;
const MAX_TOOL_GRAMMAR_SCHEMA_BYTES: usize = 8 * 1024 * 1024;

/// Shared by all tool arguments rendered into one output grammar.
pub(crate) struct ToolGrammarBudget {
    work_left: usize,
    schema_bytes_left: usize,
}

impl Default for ToolGrammarBudget {
    fn default() -> Self {
        Self {
            work_left: MAX_TOOL_GRAMMAR_WORK,
            schema_bytes_left: MAX_TOOL_GRAMMAR_SCHEMA_BYTES,
        }
    }
}

impl ToolGrammarBudget {
    fn charge_work(&mut self, work: usize) -> Result<()> {
        self.work_left = self
            .work_left
            .checked_sub(work)
            .ok_or(OutputGrammarError::ToolSchemaTooComplex)?;
        Ok(())
    }

    fn charge_bytes(&mut self, bytes: usize) -> Result<()> {
        self.schema_bytes_left = self
            .schema_bytes_left
            .checked_sub(bytes)
            .ok_or(OutputGrammarError::ToolSchemaTooComplex)?;
        Ok(())
    }

    /// Estimate the cost of copying JSON into the grammar before any cloning.
    pub(crate) fn charge_schema(&mut self, value: &Value) -> Result<()> {
        // Account for container and value overhead as well as text.
        self.charge_bytes(64)?;
        match value {
            Value::Object(object) => self.charge_object(object),
            Value::Array(values) => values.iter().try_for_each(|value| self.charge_schema(value)),
            Value::String(text) => self.charge_bytes(text.len()),
            _ => Ok(()),
        }
    }

    fn charge_object(&mut self, object: &Map<String, Value>) -> Result<()> {
        for (key, value) in object {
            self.charge_bytes(key.len())?;
            self.charge_schema(value)?;
        }
        Ok(())
    }
}

/// How one protocol renders the parameters of a call.
pub trait ArgumentSyntax {
    /// Text around and between parameters, or `None` when parameters are
    /// adjacent.
    fn separator(&self) -> Option<Format>;

    /// The grammar of one parameter whose value takes one of `options`, or
    /// `None` when none of them can be rendered.
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
    pub fn tag(
        &self,
        prefix: &str,
        suffix: &str,
        content: Format,
        end: impl Into<EndBoundary>,
    ) -> Format {
        match self {
            Self::Declared(key) => Format::tag(format!("{prefix}{key}{suffix}"), content, end),
            Self::Free => Format::sequence(vec![
                Format::const_string(prefix),
                Format::regex(Self::FREE_PATTERN),
                Format::const_string(suffix),
                content,
                match end.into() {
                    EndBoundary::Text(end) => Format::const_string(end),
                    EndBoundary::Texts(ends) => {
                        one_of(ends.into_iter().map(Format::const_string).collect())
                    }
                    EndBoundary::Token(end) => Format::Token(TokenFormat { token: end.token }),
                },
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
}

impl ValueOption<'_> {
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

impl ArgumentContext<'_> {
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

    /// Every option of a value under `schema`.
    fn value_options<'s>(
        &'s self,
        schema: &'s Value,
        budget: &mut ToolGrammarBudget,
    ) -> Result<Vec<ValueOption<'s>>> {
        let root: SchemaRoot<'s> = self.root;
        let options = root
            .options_with_budget(schema, &mut budget.work_left)
            .ok_or(OutputGrammarError::ToolSchemaTooComplex)?;
        options
            .into_iter()
            .map(|SchemaOption { ty, source }| {
                match source {
                    OptionSource::Schema(schema) => budget.charge_object(schema)?,
                    OptionSource::Literal(value) => budget.charge_schema(value)?,
                    OptionSource::Any => (),
                }
                if ty != JsonType::String {
                    for key in ["$defs", "definitions"] {
                        if let Some(definitions) = root.schema().get(key) {
                            budget.charge_schema(definitions)?;
                        }
                    }
                }
                Ok(ValueOption {
                    ty,
                    source,
                    cx: self,
                })
            })
            .collect()
    }
}

/// Build the grammar of one call's arguments under `schema`, the tool's
/// parameters schema, rendering each parameter with `syntax`.
pub(crate) fn arguments(
    schema: &Value,
    syntax: &dyn ArgumentSyntax,
    options: &ArgumentOptions,
    budget: &mut ToolGrammarBudget,
) -> Result<Format> {
    Ok(parameter_list(
        options,
        syntax.separator(),
        parameters(schema, syntax, options, budget)?,
    ))
}

/// The parameters of one call, each rendered by an [`ArgumentSyntax`], for a
/// syntax that lists them itself rather than through [`arguments`].
pub struct Parameters {
    /// The declared parameters, in declared order.
    pub declared: Vec<Parameter>,
    /// Any one additional parameter, if the schema allows them.
    pub additional: Option<Format>,
}

/// One declared parameter.
pub struct Parameter {
    pub key: String,
    pub format: Format,
    pub required: bool,
}

/// The parameters of one call under `schema`, the tool's parameters schema,
/// rendering each parameter with `syntax`.
pub(crate) fn parameters(
    schema: &Value,
    syntax: &dyn ArgumentSyntax,
    options: &ArgumentOptions,
    budget: &mut ToolGrammarBudget,
) -> Result<Parameters> {
    let cx = ArgumentContext {
        options,
        root: SchemaRoot::new(schema),
    };
    let resolved = cx.root.resolved();
    let required = resolved.get("required").and_then(Value::as_array);
    budget.charge_work(1 + required.map_or(0, Vec::len))?;
    let mut parameter = |key: ParameterKey<'_>, schema: &Value| -> Result<Option<Format>> {
        if let ParameterKey::Declared(key) = key {
            budget.charge_bytes(key.len())?;
        }
        Ok(syntax.parameter(key, &cx.value_options(schema, budget)?))
    };

    Ok(match resolved {
        Value::Object(schema) => {
            let additional = match schema.get("additionalProperties") {
                Some(Value::Bool(false)) => None,
                Some(additional) => parameter(ParameterKey::Free, additional)?,
                // Strict schemas allow no undeclared properties, and a schema
                // without declared properties allows any.
                None if schema.contains_key("properties") => None,
                None => parameter(ParameterKey::Free, &ANY_SCHEMA)?,
            };
            let required =
                required.into_iter().flatten().filter_map(Value::as_str).collect::<Vec<_>>();
            let mut declared = Vec::new();
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for (key, schema) in properties {
                    let Some(format) = parameter(ParameterKey::Declared(key), schema)? else {
                        continue;
                    };
                    declared.push(Parameter {
                        key: key.clone(),
                        format,
                        required: required.contains(&key.as_str()),
                    });
                }
            }
            Parameters {
                declared,
                additional,
            }
        }
        Value::Bool(false) => Parameters {
            declared: vec![],
            additional: None,
        },
        _ => Parameters {
            declared: vec![],
            additional: parameter(ParameterKey::Free, &ANY_SCHEMA)?,
        },
    })
}

/// Parameters in declared order, required ones mandatory, followed by any
/// number of additional parameters; or, under `any_order`, any sequence of
/// them with at least one when some are required.
fn parameter_list(
    options: &ArgumentOptions,
    separator: Option<Format>,
    Parameters {
        declared,
        additional,
    }: Parameters,
) -> Format {
    let separated = |parameter: Format| match &separator {
        Some(separator) => Format::sequence(vec![parameter, separator.clone()]),
        None => parameter,
    };
    let mut elements = Vec::from_iter(separator.clone());
    if options.any_order {
        let any_required = declared.iter().any(|parameter| parameter.required);
        let parameters = declared
            .into_iter()
            .map(|parameter| parameter.format)
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
        for Parameter {
            format, required, ..
        } in declared
        {
            let parameter = separated(format);
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

    fn check(schema: Value, syntax: &dyn ArgumentSyntax, expected: Expect) {
        check_with(schema, syntax, &ArgumentOptions::default(), expected);
    }

    fn check_with(
        schema: Value,
        syntax: &dyn ArgumentSyntax,
        options: &ArgumentOptions,
        expected: Expect,
    ) {
        let grammar =
            arguments(&schema, syntax, options, &mut ToolGrammarBudget::default()).unwrap();
        expected.assert_eq(&outline(&grammar));
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
        let arguments = arguments(
            &schema,
            &Typed,
            &ArgumentOptions::default(),
            &mut ToolGrammarBudget::default(),
        )
        .unwrap();
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
    fn argument_building_charges_copied_schema_data_before_rendering() {
        let schema = json!({
            "$defs": { "item": { "type": "integer" } },
            "properties": { "item": { "$ref": "#/$defs/item" } }
        });
        let mut budget = ToolGrammarBudget {
            work_left: 16,
            schema_bytes_left: 128,
        };
        let grammar = arguments(&schema, &Typed, &ArgumentOptions::default(), &mut budget);
        assert!(matches!(
            grammar,
            Err(OutputGrammarError::ToolSchemaTooComplex)
        ));
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
}
