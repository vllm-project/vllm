// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use indexmap::IndexMap;
use minijinja::value::{Kwargs, ValueKind, ViaDeserialize};
use minijinja::{Error as MinijinjaError, ErrorKind, Value};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::value::RawValue;
use serde_json_fmt::{JsonFormat, JsonSyntaxError};
use thiserror_ext::AsReport;

/// Hugging Face-compatible `tojson` filter for chat templates.
///
/// We cannot use MiniJinja's built-in filter directly because HF relies on
/// Python `json.dumps` semantics:
/// - no HTML escaping
/// - extra kwargs such as `ensure_ascii`, `separators`, and `sort_keys`
/// - Python-style `indent` handling
pub(super) fn hf_tojson_filter(
    value: Value,
    kwargs: Kwargs,
) -> std::result::Result<Value, MinijinjaError> {
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let indent = parse_indent(
        kwargs.get::<Option<ViaDeserialize<IndentArg>>>("indent")?.map(|value| value.0),
    );
    let separators = parse_separators(
        kwargs
            .get::<Option<ViaDeserialize<SeparatorsArg>>>("separators")?
            .map(|value| value.0),
        indent.is_some(),
    );
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);

    kwargs.assert_all_used()?;

    let json_str = build_json_format(indent, separators.0, separators.1, ensure_ascii)?
        .format_to_string(&PythonJson(&value, sort_keys))
        .map_err(|e| {
            MinijinjaError::new(
                ErrorKind::InvalidOperation,
                format!("Failed to serialize JSON: {}", e.as_report()),
            )
        })?;

    Ok(Value::from_safe_string(json_str))
}

pub(super) fn raw_json_to_minijinja_value(raw: &str) -> Option<Value> {
    let raw = raw.trim();
    match raw.as_bytes().first()? {
        b'[' => {
            let values: Vec<Box<RawValue>> = serde_json::from_str(raw).ok()?;
            values
                .iter()
                .map(|value| raw_json_to_minijinja_value(value.get()))
                .collect::<Option<Value>>()
        }
        b'{' => {
            let values: IndexMap<String, Box<RawValue>> = serde_json::from_str(raw).ok()?;
            values
                .into_iter()
                .map(|(key, value)| Some((key, raw_json_to_minijinja_value(value.get())?)))
                .collect::<Option<Value>>()
        }
        _ => raw
            .parse::<i128>()
            .map(Value::from)
            .ok()
            .or_else(|| raw.parse::<u128>().map(Value::from).ok())
            .or_else(|| serde_json::from_str(raw).ok()),
    }
}

struct PythonJson<'a>(&'a Value, bool);

impl Serialize for PythonJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error as _;

        match self.0.kind() {
            ValueKind::Map => {
                let mut keys: Vec<_> = self.0.try_iter().map_err(S::Error::custom)?.collect();
                if self.1 {
                    keys.sort_by_key(Value::to_string);
                }
                let mut map = serializer.serialize_map(Some(keys.len()))?;
                for key in keys {
                    let value = self.0.get_item(&key).map_err(S::Error::custom)?;
                    map.serialize_entry(&key, &PythonJson(&value, self.1))?;
                }
                map.end()
            }
            ValueKind::Seq | ValueKind::Iterable => {
                let mut seq = serializer.serialize_seq(self.0.len())?;
                for value in self.0.try_iter().map_err(S::Error::custom)? {
                    seq.serialize_element(&PythonJson(&value, self.1))?;
                }
                seq.end()
            }
            ValueKind::Number if !self.0.is_integer() => {
                let mut number = serde_json::to_string(self.0).map_err(S::Error::custom)?;
                if let Some(index) = number.find('e')
                    && number.len() - index == 3
                {
                    number.insert(index + 2, '0');
                }
                let raw: &RawValue = serde_json::from_str(&number).map_err(S::Error::custom)?;
                raw.serialize(serializer)
            }
            _ => self.0.serialize(serializer),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum IndentArg {
    // Python `json.dumps` accepts bool, int, and string indentation styles.
    Bool(bool),
    Integer(i64),
    String(String),
}

fn parse_indent(value: Option<IndentArg>) -> Option<String> {
    match value? {
        IndentArg::Bool(indent) => Some(if indent {
            " ".to_owned()
        } else {
            String::new()
        }),
        IndentArg::Integer(indent) => Some(if indent > 0 {
            " ".repeat(indent as usize)
        } else {
            String::new()
        }),
        IndentArg::String(indent) => Some(indent),
    }
}

#[derive(Deserialize)]
struct SeparatorsArg((String, String));

fn parse_separators(value: Option<SeparatorsArg>, pretty: bool) -> (String, String) {
    let Some(SeparatorsArg((item_separator, key_separator))) = value else {
        let default_item_separator = if pretty { "," } else { ", " };
        let default_key_separator = ": ";

        return (
            default_item_separator.to_owned(),
            default_key_separator.to_owned(),
        );
    };

    (item_separator, key_separator)
}

fn build_json_format(
    indent: Option<String>,
    item_separator: String,
    key_separator: String,
    ensure_ascii: bool,
) -> std::result::Result<JsonFormat, MinijinjaError> {
    JsonFormat::new()
        .indent(indent)
        .map_err(map_json_syntax_error("indent"))?
        .comma(item_separator)
        .map_err(map_json_syntax_error("separators (item)"))?
        .colon(key_separator)
        .map_err(map_json_syntax_error("separators (key)"))
        .map(|format| format.ascii(ensure_ascii))
}

fn map_json_syntax_error(
    field: &'static str,
) -> impl FnOnce(JsonSyntaxError) -> MinijinjaError + Copy {
    move |error| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("invalid {field} value for tojson: {error}"),
        )
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use minijinja::Environment;
    use serde_json::json;
    use thiserror_ext::AsReport;

    use super::{hf_tojson_filter, raw_json_to_minijinja_value};

    fn render(template: &str, payload: impl serde::Serialize) -> String {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson_filter);
        env.render_str(template, minijinja::context! { payload }).unwrap()
    }

    fn render_error(template: &str, payload: serde_json::Value) -> minijinja::Error {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson_filter);
        env.render_str(template, json!({ "payload": payload })).unwrap_err()
    }

    #[test]
    fn tojson_does_not_html_escape_like_minijinja_builtin() {
        let rendered = render("{{ payload|tojson }}", json!("<tag>&'"));
        assert_eq!(rendered, "\"<tag>&'\"");
    }

    #[test]
    fn tojson_supports_sort_keys_recursively() {
        let rendered = render(
            "{{ payload|tojson(sort_keys=true) }}",
            json!({
                "z": {"b": 1, "a": 2},
                "a": 0
            }),
        );

        assert_eq!(rendered, "{\"a\": 0, \"z\": {\"a\": 2, \"b\": 1}}");
    }

    #[test]
    fn tojson_supports_indent() {
        let rendered = render("{{ payload|tojson(indent=2) }}", json!([1, 2]));

        assert_eq!(rendered, "[\n  1,\n  2\n]");
    }

    #[test]
    fn tojson_supports_ensure_ascii_false() {
        let rendered = render("{{ payload|tojson(ensure_ascii=false) }}", json!("中文"));
        assert_eq!(rendered, "\"中文\"");
    }

    #[test]
    fn tojson_supports_ensure_ascii_true() {
        let rendered = render("{{ payload|tojson(ensure_ascii=true) }}", json!("中文"));
        assert_eq!(rendered, "\"\\u4e2d\\u6587\"");
    }

    #[test]
    fn tojson_supports_separators() {
        let rendered = render(
            "{{ payload|tojson(separators=[',', ':']) }}",
            json!({
                "x": [1, 2]
            }),
        );

        assert_eq!(rendered, "{\"x\":[1,2]}");
    }

    #[test]
    fn tojson_matches_python_number_spelling() {
        let payload = raw_json_to_minijinja_value(
            r#"{"x":2,"y":1.00,"small":1e-7,"big":123456789012345678901234567890}"#,
        )
        .unwrap();
        let rendered = render("{{ payload|tojson }}", payload);

        // TODO: we cannot preserve the original number precision by enabling `serde_json`'s
        // `arbitrary_precision` feature, otherwise the following test
        // `serialized_json_numbers_do_not_leak_serde_private_representation` will fail.
        // See issue: https://github.com/mitsuhiko/minijinja/issues/641
        assert_eq!(
            rendered,
            "{\"x\": 2, \"y\": 1.0, \"small\": 1e-07, \"big\": 123456789012345678901234567890}"
        );
    }

    #[test]
    fn tojson_preserves_nested_numbers_with_formatting_options() {
        let payload = raw_json_to_minijinja_value(
            r#"{"z":[-123456789012345678901234567890,1e-7],"a":340282366920938463463374607431768211455}"#,
        ).unwrap();
        assert_eq!(
            render(
                "{{ payload|tojson(sort_keys=true, separators=[',', ':']) }}",
                payload
            ),
            r#"{"a":340282366920938463463374607431768211455,"z":[-123456789012345678901234567890,1e-07]}"#
        );
    }

    #[test]
    fn exponent_normalization_skips_json_strings() {
        assert_eq!(
            render(
                "{{ payload|tojson }}",
                json!([1e-7, "1e-7", "\"1e-7\\", "中文"])
            ),
            r#"[1e-07, "1e-7", "\"1e-7\\", "中文"]"#
        );
    }

    #[test]
    fn tojson_formats_numbers_in_lazy_iterators() {
        assert_eq!(
            render(
                "{{ payload|map(attribute='x')|tojson }}",
                json!([{"x": 1e-7}])
            ),
            "[1e-07]"
        );
    }

    #[test]
    fn serialized_json_numbers_do_not_leak_serde_private_representation() {
        let payload: serde_json::Value = serde_json::from_str(r#"{"x":2,"y":1.00}"#).unwrap();
        let rendered = render("{{ payload }}", payload);

        // TODO: we cannot preserve the original number precision by enabling `serde_json`'s
        // `arbitrary_precision` feature, otherwise this will fail.
        // See issue: https://github.com/mitsuhiko/minijinja/issues/641
        assert!(!rendered.contains("$serde_json::private::Number"));
        assert_eq!(rendered, r#"{"x": 2, "y": 1.0}"#);
    }

    #[test]
    fn tojson_supports_negative_indent_as_newline_only() {
        let rendered = render("{{ payload|tojson(indent=-1) }}", json!([1, 2]));
        assert_eq!(rendered, "[\n1,\n2\n]");
    }

    #[test]
    fn tojson_supports_string_indent() {
        let rendered = render("{{ payload|tojson(indent='  ') }}", json!([1, 2]));
        assert_eq!(rendered, "[\n  1,\n  2\n]");
    }

    #[test]
    fn tojson_supports_boolean_indent() {
        let rendered_true = render("{{ payload|tojson(indent=true) }}", json!([1, 2]));
        assert_eq!(rendered_true, "[\n 1,\n 2\n]");

        let rendered_false = render("{{ payload|tojson(indent=false) }}", json!([1, 2]));
        assert_eq!(rendered_false, "[\n1,\n2\n]");
    }

    #[test]
    fn tojson_combines_indent_sort_keys_separators_and_ensure_ascii() {
        let rendered = render(
            "{{ payload|tojson(ensure_ascii=true, sort_keys=true, separators=[',', ':'], indent='  ') }}",
            json!({
                "b": "<中>",
                "a": [1, 2]
            }),
        );

        assert_eq!(
            rendered,
            "{\n  \"a\":[\n    1,\n    2\n  ],\n  \"b\":\"<\\u4e2d>\"\n}"
        );
    }

    #[test]
    fn tojson_rejects_invalid_indent() {
        let error = render_error("{{ payload|tojson(indent='-->') }}", json!({"a": 1}));
        expect!["invalid operation: invalid indent value for tojson: string contains unexpected character '-' (in <string>:1)"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn tojson_rejects_invalid_separator_shape() {
        let error = render_error("{{ payload|tojson(separators=':,') }}", json!({"a": 1}));
        expect!["cannot deserialize: invalid type: string \":,\", expected a tuple of size 2 (in <string>:1)"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn tojson_rejects_invalid_key_separator() {
        let error = render_error(
            "{{ payload|tojson(separators=[',', '=>']) }}",
            json!({"a": 1}),
        );
        expect!["invalid operation: invalid separators (key) value for tojson: string contains unexpected character '=' (in <string>:1)"]
            .assert_eq(&error.to_report_string());
    }
}
