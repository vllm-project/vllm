// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Kimi K3 XTML tool-call grammar: call tags and their typed arguments.

use serde_json::Value;
use xgrammar_structural_tag::format::{Format, JsonSchemaFormat, TagFormat};
use xgrammar_structural_tag::tool::{FunctionToolParam, function_parameters};

use super::super::{ARG_CLOSE, ARG_OPEN, CALL_CLOSE, JSON_CLOSE, JSON_OPEN, OPEN, SEP};
use crate::output_grammar::arguments::{
    self, ArgumentOptions, ArgumentSyntax, JsonType, ParameterKey, ValueOption, group_by, one_of,
};

/// XTML arguments:
/// `<|open|>argument key="KEY" type="TYPE"<|sep|>VALUE<|close|>argument<|sep|>`,
/// adjacent to each other.
///
/// The parser decodes each value by its `type` attribute, so every value option
/// sits under the attribute of its own JSON type. String values are raw text
/// and every other value is JSON.
struct XtmlArguments;

impl ArgumentSyntax for XtmlArguments {
    fn separator(&self) -> Option<Format> {
        None
    }

    fn parameter(&self, key: ParameterKey<'_>, options: &[ValueOption<'_>]) -> Option<Format> {
        let escaped;
        let key = match key {
            ParameterKey::Declared(key) => {
                escaped = escape_attr_value(key);
                ParameterKey::Declared(&escaped)
            }
            ParameterKey::Free => ParameterKey::Free,
        };
        let arguments = group_by(options, |option| xtml_type(option.ty))
            .into_iter()
            .filter_map(|(type_name, options)| {
                let values = options
                    .into_iter()
                    .filter_map(|option| match option.ty {
                        // The parser ends the call at its close marker, so a
                        // string value may contain neither close marker.
                        JsonType::String => option.raw_string(&[ARG_CLOSE, CALL_CLOSE]),
                        _ => Some(option.json()),
                    })
                    .collect::<Vec<_>>();
                (!values.is_empty()).then(|| {
                    let suffix = format!("\" type=\"{type_name}\"{SEP}");
                    key.tag(
                        &format!("{ARG_OPEN} key=\""),
                        &suffix,
                        one_of(values),
                        ARG_CLOSE,
                    )
                })
            })
            .collect::<Vec<_>>();
        (!arguments.is_empty()).then(|| one_of(arguments))
    }
}

/// The `type` attribute K3 renders for a value of type `ty`, as
/// `_xtml_type` in the checkpoint's `encoding_k3.py`: integers and floats are
/// both `number`.
fn xtml_type(ty: JsonType) -> &'static str {
    match ty {
        JsonType::String => "string",
        JsonType::Integer | JsonType::Number => "number",
        JsonType::Boolean => "boolean",
        JsonType::Null => "null",
        JsonType::Object => "object",
        JsonType::Array => "array",
    }
}

/// Build the tag for one call to `tool`.
pub(super) fn call_tag(tool: &FunctionToolParam) -> TagFormat {
    let parameters = function_parameters(&tool.function);
    let typed_arguments =
        arguments::arguments(&parameters, &XtmlArguments, &ArgumentOptions::default());
    let call_body = Format::or(vec![typed_arguments, raw_json_arguments(&parameters)]);

    TagFormat::new(
        format!(
            "{OPEN}call tool=\"{}\" index=\"",
            escape_attr_value(&tool.function.name)
        ),
        Format::sequence(vec![
            Format::regex("[1-9][0-9]*"),
            Format::const_string(format!("\"{SEP}")),
            call_body,
        ]),
        CALL_CLOSE,
    )
}

fn raw_json_arguments(parameters: &Value) -> Format {
    Format::tag(
        format!("{JSON_OPEN} type=\"object\"{SEP}"),
        Format::JsonSchema(JsonSchemaFormat::new(parameters.clone())),
        JSON_CLOSE,
    )
}

fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use serde_json::json;
    use xgrammar_structural_tag::FunctionDefinition;
    use xgrammar_structural_tag::format::Format;
    use xgrammar_structural_tag::tool::FunctionToolParam;

    use crate::output_grammar::test_utils::outline;

    fn check(parameters: serde_json::Value, expected: Expect) {
        let tool = FunctionToolParam::new(
            FunctionDefinition::new("get_weather").with_parameters(parameters),
        );
        expected.assert_eq(&outline(&Format::Tag(super::call_tag(&tool))));
    }

    #[test]
    fn call_tag_matches_xtml_arguments() {
        check(
            json!({
                "$defs": {
                    "place": { "type": "object", "properties": { "city": { "type": "string" } } }
                },
                "type": "object",
                "properties": {
                    "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] },
                    "place": { "$ref": "#/$defs/place", "type": "object" }
                },
                "required": ["place"]
            }),
            expect![[r#"
                tag `<|open|>call tool="get_weather" index="` .. `<|close|>call<|sep|>`
                  sequence
                    /[1-9][0-9]*/
                    `"<|sep|>`
                    or
                      sequence
                        optional
                          tag `<|open|>argument key="unit" type="string"<|sep|>` .. `<|close|>argument<|sep|>`
                            or
                              `celsius`
                              `fahrenheit`
                        tag `<|open|>argument key="place" type="object"<|sep|>` json({ city?: string } where place = { city?: string }) `<|close|>argument<|sep|>`
                      tag `<|open|>json type="object"<|sep|>` json({ unit?: "celsius" | "fahrenheit", place: place & object } where place = { city?: string }) `<|close|>json<|sep|>`
            "#]],
        );
    }

    #[test]
    fn union_argument_content_matches_its_xtml_type() {
        check(
            json!({
                "type": "object",
                "properties": {
                    "count": { "type": ["integer", "null"] }
                }
            }),
            expect![[r#"
                tag `<|open|>call tool="get_weather" index="` .. `<|close|>call<|sep|>`
                  sequence
                    /[1-9][0-9]*/
                    `"<|sep|>`
                    or
                      optional
                        or
                          tag `<|open|>argument key="count" type="number"<|sep|>` json(integer) `<|close|>argument<|sep|>`
                          tag `<|open|>argument key="count" type="null"<|sep|>` json(null) `<|close|>argument<|sep|>`
                      tag `<|open|>json type="object"<|sep|>` json({ count?: integer | null }) `<|close|>json<|sep|>`
            "#]],
        );
    }
}
