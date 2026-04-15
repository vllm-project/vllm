use minijinja::value::Kwargs;
use minijinja::{Error as MinijinjaError, ErrorKind, Value};
use serde::Serialize;
use serde_json::ser::PrettyFormatter;
use serde_json::{self, Value as JsonValue};

/// Custom `tojson` filter compatible with HuggingFace transformers' implementation.
pub(super) fn hf_tojson_filter(
    value: Value,
    kwargs: Kwargs,
) -> std::result::Result<Value, MinijinjaError> {
    let _ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<i64> = kwargs.get("indent")?;
    let _separators: Option<Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;

    kwargs.assert_all_used()?;

    let json_value: serde_json::Value = serde_json::to_value(&value).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Failed to convert to JSON value: {e}"),
        )
    })?;

    fn serialize_with_indent<T: Serialize>(
        value: &T,
        spaces: usize,
    ) -> std::result::Result<String, MinijinjaError> {
        let indent_str = vec![b' '; spaces];
        let formatter = PrettyFormatter::with_indent(&indent_str);
        let mut buf = Vec::new();
        let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut serializer).map_err(|e| {
            MinijinjaError::new(
                ErrorKind::InvalidOperation,
                format!("Failed to serialize JSON: {e}"),
            )
        })?;
        String::from_utf8(buf).map_err(|e| {
            MinijinjaError::new(
                ErrorKind::InvalidOperation,
                format!("Invalid UTF-8 in JSON output: {e}"),
            )
        })
    }

    let json_str: std::result::Result<String, MinijinjaError> = {
        let sorted_json;
        let value_to_serialize = if sort_keys.unwrap_or(false) {
            sorted_json = sort_json_keys(&json_value);
            &sorted_json
        } else {
            &json_value
        };

        if let Some(spaces) = indent {
            if spaces < 0 {
                return Err(MinijinjaError::new(
                    ErrorKind::InvalidOperation,
                    "indent cannot be negative",
                ));
            }
            serialize_with_indent(value_to_serialize, spaces as usize)
        } else {
            serde_json::to_string(value_to_serialize).map_err(|e| {
                MinijinjaError::new(
                    ErrorKind::InvalidOperation,
                    format!("Failed to serialize JSON: {e}"),
                )
            })
        }
    };

    json_str.map(Value::from_safe_string)
}

/// Recursively sort all object keys in a JSON value.
fn sort_json_keys(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(map) => {
            let mut sorted: serde_json::Map<String, JsonValue> = serde_json::Map::new();
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), sort_json_keys(&map[key]));
            }
            JsonValue::Object(sorted)
        }
        JsonValue::Array(arr) => JsonValue::Array(arr.iter().map(sort_json_keys).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use minijinja::Environment;
    use serde_json::json;

    use super::hf_tojson_filter;

    fn render(template: &str, payload: serde_json::Value) -> String {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson_filter);
        env.render_str(template, json!({ "payload": payload }))
            .unwrap()
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

        assert_eq!(rendered, "{\"a\":0,\"z\":{\"a\":2,\"b\":1}}");
    }

    #[test]
    fn tojson_supports_indent() {
        let rendered = render("{{ payload|tojson(indent=2) }}", json!([1, 2]));

        assert_eq!(rendered, "[\n  1,\n  2\n]");
    }

    #[test]
    fn tojson_accepts_hf_ensure_ascii_kwarg() {
        // TODO: implement full ensure_ascii semantics instead of only accepting the kwarg.
        let rendered = render("{{ payload|tojson(ensure_ascii=false) }}", json!("中文"));
        assert_eq!(rendered, "\"中文\"");
    }

    #[test]
    fn tojson_accepts_hf_separators_kwarg() {
        // TODO: implement separators semantics instead of only accepting the kwarg.
        let rendered = render(
            "{{ payload|tojson(separators=[',', ':']) }}",
            json!({
                "b": 1,
                "a": 2
            }),
        );

        let reparsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(reparsed, json!({ "b": 1, "a": 2 }));
    }

    #[test]
    fn tojson_rejects_negative_indent() {
        // TODO: align with Python json.dumps, which accepts negative indent values.
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson_filter);

        let error = env
            .render_str(
                "{{ payload|tojson(indent=-1) }}",
                json!({ "payload": { "a": 1 } }),
            )
            .unwrap_err();

        assert!(error.to_string().contains("indent cannot be negative"));
    }
}
