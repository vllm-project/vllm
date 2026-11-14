// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Content parsers: turn one region's captured text into a JSON value.
//!
//! Port of `content_parsers.py`. Each parser takes a chunk of captured text and
//! parses it into a single key in the output message dictionary.

use std::sync::Arc;

use regex::Regex;
use serde::Deserialize;
use serde_json::{Map, Value};
use thiserror_ext::AsReport as _;

use super::pcre;
use super::spec::{ContentKind, deserialize};
use super::{Result, invalid};
use crate::tool::{convert_integer_text, convert_number_text};
use crate::unified::{self, parsing_failed};

// Sentinel characters for lax-JSON string pre-extraction — ASCII control chars
// that should never appear in real LLM output.
const LAX_OPEN: char = '\u{1}';
const LAX_CLOSE: char = '\u{2}';

/// A configured content parser (`content` plus `content_args` of one field).
#[derive(Debug, Clone)]
pub(super) enum ContentParser {
    Text(TextArgs),
    /// Converted like an `integer` tool parameter.
    Int,
    /// Converted like a `number` tool parameter.
    Float,
    Bool(TextArgs),
    Json(JsonArgs),
    XmlInline(XmlInlineArgs),
    KvLines(KvLinesArgs),
}

/// The `strip` argument shared by parsers that go through `_text`.
#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct TextArgs {
    #[serde(default = "default_strip")]
    pub strip: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct JsonArgs {
    #[serde(flatten)]
    text: TextArgs,
    /// Quote bare-identifier keys before parsing.
    #[serde(default)]
    unquoted_keys: bool,
    /// Strings delimited by these custom markers are pre-extracted, then restored
    /// as standard JSON strings.
    #[serde(default)]
    string_delims: Vec<(String, String)>,
    /// Return stripped text if parsing fails.
    #[serde(default)]
    allow_non_json: bool,
}

#[derive(Debug, Clone)]
pub(super) struct XmlInlineArgs {
    /// The pattern's source and compiled form.
    tag_pattern: (String, Arc<pcre::Regex>),
    /// Index of the required `key` group.
    key_group: usize,
    /// Index of the `value` group, if the pattern declares one.
    value_group: Option<usize>,
    value_parser: Option<Box<ContentParser>>,
    /// Collect duplicate keys into a list.
    merge_duplicates: bool,
}

#[derive(Debug, Clone)]
pub(super) struct KvLinesArgs {
    text: TextArgs,
    line_sep: String,
    kv_sep: String,
    value_parser: Option<Box<ContentParser>>,
}

/// `content_args` of `xml-inline`.
#[derive(Deserialize)]
struct XmlInlineSpec {
    /// Must have named groups `key` and `value`.
    tag_pattern: String,
    value_parser: Option<ValueParserSpec>,
    #[serde(default)]
    merge_duplicates: bool,
}

/// `content_args` of `kv-lines`.
#[derive(Deserialize)]
struct KvLinesSpec {
    #[serde(flatten)]
    text: TextArgs,
    #[serde(default = "default_line_sep")]
    line_sep: String,
    #[serde(default = "default_kv_sep")]
    kv_sep: String,
    value_parser: Option<ValueParserSpec>,
}

/// A nested `value_parser`: `{"name": ..., "args": {...}}`.
#[derive(Deserialize)]
struct ValueParserSpec {
    #[serde(default)]
    name: ContentKind,
    #[serde(default)]
    args: Map<String, Value>,
}

impl ContentParser {
    /// Build a parser from its kind and `content_args`.
    ///
    /// Unknown argument keys are ignored, as in Transformers. Argument types and the
    /// `xml-inline` tag pattern are checked here rather than at parse time.
    pub fn new(scope: &str, kind: ContentKind, args: &Map<String, Value>) -> Result<Self> {
        let context = format!("{scope}: content_args");
        let args = Value::Object(args.clone());
        Ok(match kind {
            ContentKind::Text => Self::Text(deserialize(&args, &context)?),
            // Number conversion ignores `strip`, but a malformed value is still rejected.
            ContentKind::Int => {
                deserialize::<TextArgs>(&args, &context)?;
                Self::Int
            }
            ContentKind::Float => {
                deserialize::<TextArgs>(&args, &context)?;
                Self::Float
            }
            ContentKind::Bool => Self::Bool(deserialize(&args, &context)?),
            ContentKind::Json => {
                let args: JsonArgs = deserialize(&args, &context)?;
                if args
                    .string_delims
                    .iter()
                    .any(|(open, close)| open.is_empty() || close.is_empty())
                {
                    return Err(invalid!("{context}: string_delims cannot be empty strings"));
                }
                Self::Json(args)
            }
            ContentKind::XmlInline => {
                let spec: XmlInlineSpec = deserialize(&args, &context)?;
                let regex = pcre::Regex::new(scope, &spec.tag_pattern)?;
                let group = |group: &str| {
                    regex.group_names().find_map(|(index, name)| (name == group).then_some(index))
                };
                let Some(key_group) = group("key") else {
                    return Err(invalid!(
                        "{scope}: xml-inline: tag_pattern must have a named group 'key'. Pattern: {}",
                        spec.tag_pattern
                    ));
                };
                let value_group = group("value");
                Self::XmlInline(XmlInlineArgs {
                    tag_pattern: (spec.tag_pattern, Arc::new(regex)),
                    key_group,
                    value_group,
                    value_parser: ValueParserSpec::compile(scope, spec.value_parser)?,
                    merge_duplicates: spec.merge_duplicates,
                })
            }
            ContentKind::KvLines => {
                let spec: KvLinesSpec = deserialize(&args, &context)?;
                if spec.line_sep.is_empty() {
                    return Err(invalid!("{context}: line_sep cannot be empty"));
                }
                Self::KvLines(KvLinesArgs {
                    text: spec.text,
                    line_sep: spec.line_sep,
                    kv_sep: spec.kv_sep,
                    value_parser: ValueParserSpec::compile(scope, spec.value_parser)?,
                })
            }
        })
    }

    /// Parse one region body.
    pub fn parse(&self, text: &str) -> unified::Result<Value> {
        match self {
            Self::Text(args) => Ok(Value::String(args.apply(text).to_string())),
            // Numbers follow the tool-parameter conversion.
            Self::Int => convert_integer_text(text)
                .ok_or_else(|| parsing_failed!("int: invalid literal {text:?}")),
            Self::Float => convert_number_text(text)
                .ok_or_else(|| parsing_failed!("float: invalid or non-finite literal {text:?}")),
            Self::Bool(args) => {
                let text = args.apply(text).to_lowercase();
                Ok(Value::Bool(text == "true" || text == "1"))
            }
            Self::Json(args) => args.parse(text),
            Self::XmlInline(args) => args.parse(text),
            Self::KvLines(args) => args.parse(text),
        }
    }

    /// The source of an `xml-inline` tag pattern.
    pub fn tag_pattern(&self) -> Option<&str> {
        match self {
            Self::XmlInline(args) => Some(&args.tag_pattern.0),
            _ => None,
        }
    }
}

impl ValueParserSpec {
    /// Build the nested parser, if any.
    fn compile(scope: &str, spec: Option<Self>) -> Result<Option<Box<ContentParser>>> {
        spec.map(|spec| {
            ContentParser::new(&format!("{scope}: value_parser"), spec.name, &spec.args)
                .map(Box::new)
        })
        .transpose()
    }
}

impl TextArgs {
    /// Apply `_text`: strip surrounding whitespace unless disabled.
    pub fn apply(self, text: &str) -> &str {
        if self.strip { text.trim() } else { text }
    }
}

impl JsonArgs {
    /// JSON parser with optional dialect knobs for LLM-emitted quirks.
    fn parse(&self, text: &str) -> unified::Result<Value> {
        if !self.string_delims.is_empty() && text.contains([LAX_OPEN, LAX_CLOSE]) {
            return Err(parsing_failed!(
                "json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely."
            ));
        }

        let mut working = text.to_string();
        let mut captured: Vec<String> = Vec::new();
        for (open, close) in &self.string_delims {
            working = extract_delimited(&working, open, close, &mut captured);
        }

        if self.unquoted_keys {
            working = quote_bare_keys(&working);
        }

        for (index, string) in captured.iter().enumerate() {
            let quoted = serde_json::to_string(string).map_err(|error| {
                parsing_failed!("json: failed to quote string: {}", error.as_report())
            })?;
            working = working.replace(&format!("{LAX_OPEN}{index}{LAX_CLOSE}"), &quoted);
        }

        match serde_json::from_str(&working) {
            Ok(value) => Ok(value),
            Err(_) if self.allow_non_json => Ok(Value::String(self.text.apply(text).to_string())),
            Err(error) if working == text => Err(parsing_failed!(
                "json parser could not parse region as JSON.\nContent: {text:?}\nError: {}",
                error.as_report()
            )),
            Err(error) => Err(parsing_failed!(
                "json: could not parse after dialect transforms.\nOriginal: {text:?}\nTransformed: {working:?}\nError: {}",
                error.as_report()
            )),
        }
    }
}

impl XmlInlineArgs {
    /// Parse shallow XML-ish tags into a dict.
    fn parse(&self, text: &str) -> unified::Result<Value> {
        let mut out = Map::new();
        let tags =
            self.tag_pattern.1.find_iter(text).map_err(|error| parsing_failed!("{error}"))?;
        for tag in tags {
            let group = |index: usize| tag.groups[index].clone().map(|span| &text[span]);
            let key = group(self.key_group).ok_or_else(|| {
                parsing_failed!("xml-inline: named group 'key' did not participate")
            })?;
            let raw = match self.value_group {
                // The pattern has no `value` group.
                None => Some(""),
                Some(index) => group(index),
            };
            let value = match raw {
                Some(raw) => sub_parse(raw, self.value_parser.as_deref())?,
                None if self.value_parser.is_none() => Value::Null,
                None => {
                    return Err(parsing_failed!(
                        "xml-inline: named group 'value' did not participate"
                    ));
                }
            };
            match out.get_mut(key) {
                Some(existing) if self.merge_duplicates => match existing {
                    Value::Array(values) => values.push(value),
                    _ => {
                        let first = existing.take();
                        *existing = Value::Array(vec![first, value]);
                    }
                },
                _ => {
                    out.insert(key.to_string(), value);
                }
            }
        }
        Ok(Value::Object(out))
    }
}

impl KvLinesArgs {
    /// Parse line-delimited `key<sep>value` pairs into a dict.
    fn parse(&self, text: &str) -> unified::Result<Value> {
        let mut out = Map::new();
        for line in text.split(self.line_sep.as_str()) {
            let line = self.text.apply(line);
            let Some((key, value)) = line.split_once(self.kv_sep.as_str()) else {
                continue;
            };
            let (key, value) = (self.text.apply(key), self.text.apply(value));
            out.insert(
                key.to_string(),
                sub_parse(value, self.value_parser.as_deref())?,
            );
        }
        Ok(Value::Object(out))
    }
}

/// Parse `raw` with an optional nested value parser.
fn sub_parse(raw: &str, value_parser: Option<&ContentParser>) -> unified::Result<Value> {
    match value_parser {
        None => Ok(Value::String(raw.to_string())),
        Some(parser) => parser.parse(raw),
    }
}

/// Replace every `open(.*?)close` span with an indexed sentinel, collecting the
/// inner strings. Equivalent to the non-overlapping, left-to-right `re.sub` with
/// `re.DOTALL` used by Transformers.
fn extract_delimited(text: &str, open: &str, close: &str, captured: &mut Vec<String>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        let inner_start = start + open.len();
        // No close after this open means no close after any later open either.
        let Some(inner_len) = rest[inner_start..].find(close) else {
            break;
        };
        out.push_str(&rest[..start]);
        captured.push(rest[inner_start..inner_start + inner_len].to_string());
        out.push(LAX_OPEN);
        out.push_str(&(captured.len() - 1).to_string());
        out.push(LAX_CLOSE);
        rest = &rest[inner_start + inner_len + close.len()..];
    }
    out.push_str(rest);
    out
}

/// Quote bare-identifier keys: Transformers' `(?<=[{,])(\w+):` → `"\1":`.
///
/// The Rust regex engine has no lookbehind, so the preceding `{` or `,` is
/// matched and written back. Matches cannot overlap either way, so the rewrite is
/// equivalent.
fn quote_bare_keys(text: &str) -> String {
    static BARE_KEY: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"([{,])(\w+):").expect("valid regex"));

    BARE_KEY.replace_all(text, r#"$1"$2":"#).into_owned()
}

fn default_strip() -> bool {
    true
}

fn default_line_sep() -> String {
    "\n".to_string()
}

fn default_kv_sep() -> String {
    ":".to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn kind(name: &str) -> ContentKind {
        serde_json::from_value(json!(name)).unwrap()
    }

    fn parser(name: &str, args: Value) -> ContentParser {
        ContentParser::new("test", kind(name), args.as_object().unwrap()).unwrap()
    }

    #[test]
    fn scalar_parsers_convert_like_tool_parameters() {
        let int = parser("int", json!({}));
        assert_eq!(int.parse(" 42\n").unwrap(), json!(42));
        assert_eq!(int.parse("-1_000").unwrap(), json!(-1000));
        assert!(int.parse("1__0").is_err());
        assert!(int.parse("4.2").is_err());

        let float = parser("float", json!({}));
        assert_eq!(float.parse("1.5").unwrap(), json!(1.5));
        assert_eq!(float.parse(" 1e3 ").unwrap(), json!(1000.0));
        assert!(float.parse("inf").is_err(), "JSON cannot represent inf");

        let boolean = parser("bool", json!({}));
        assert_eq!(boolean.parse(" TRUE ").unwrap(), json!(true));
        assert_eq!(boolean.parse("1").unwrap(), json!(true));
        assert_eq!(boolean.parse("yes").unwrap(), json!(false));

        let text = parser("text", json!({"strip": false}));
        assert_eq!(text.parse(" a \n").unwrap(), json!(" a \n"));
        assert_eq!(
            parser("text", json!({})).parse(" a \n").unwrap(),
            json!("a")
        );
    }

    #[test]
    fn json_dialects_match_gemma4_syntax() {
        let json = parser(
            "json",
            json!({"unquoted_keys": true, "string_delims": [["<|\"|>", "<|\"|>"]]}),
        );
        let value = json
            .parse(
                r#"{bool_value:true,list_value:[<|"|>foo<|"|>,<|"|>bar<|"|>],null_value:null,number_value:1,string_value:<|"|>a, b: {c}<|"|>,struct_value:{foo:<|"|>bar<|"|>}}"#,
            )
            .unwrap();
        assert_eq!(
            value,
            json!({
                "bool_value": true,
                "list_value": ["foo", "bar"],
                "null_value": null,
                "number_value": 1,
                "string_value": "a, b: {c}",
                "struct_value": {"foo": "bar"},
            })
        );
        assert!(json.parse("{a:\u{1}}").is_err());
    }

    #[test]
    fn json_allow_non_json_returns_stripped_text() {
        let json = parser("json", json!({"allow_non_json": true}));
        assert_eq!(json.parse(" celsius \n").unwrap(), json!("celsius"));
        assert_eq!(json.parse(" [1, 2] ").unwrap(), json!([1, 2]));
        assert!(parser("json", json!({})).parse("celsius").is_err());
    }

    #[test]
    fn xml_inline_parses_tags_with_value_parser_and_merging() {
        let xml = parser(
            "xml-inline",
            json!({
                "tag_pattern": r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>",
                "value_parser": {"name": "json", "args": {"allow_non_json": true}},
                "merge_duplicates": true,
            }),
        );
        let value = xml
            .parse(
                "<parameter=locations>\n[{\"city\": \"Paris\"}]\n</parameter>\n\
                 <parameter=unit>\ncelsius\n</parameter><parameter=unit>2</parameter>",
            )
            .unwrap();
        assert_eq!(
            value,
            json!({"locations": [{"city": "Paris"}], "unit": ["celsius", 2]})
        );
    }

    #[test]
    fn xml_inline_tags_may_end_where_the_pattern_looks_ahead() {
        let xml = parser(
            "xml-inline",
            json!({
                "tag_pattern": r#"<arg name="(?P<key>\w+)">(?P<value>.*?)\n(?:</arg>|(?=<arg name="))"#,
            }),
        );
        let value = xml
            .parse("<arg name=\"a\">1\n<arg name=\"b\">x <arg name=\"c\"> y\n</arg>")
            .unwrap();
        assert_eq!(value, json!({"a": "1", "b": "x <arg name=\"c\"> y"}));
        assert_eq!(
            xml.tag_pattern(),
            Some(r#"<arg name="(?P<key>\w+)">(?P<value>.*?)\n(?:</arg>|(?=<arg name="))"#)
        );

        // Look-around anywhere, as in Python.
        let middle = parser(
            "xml-inline",
            json!({"tag_pattern": r"(?P<key>\w+)(?=:)(?P<value>.*)"}),
        );
        assert_eq!(middle.parse("a:b").unwrap(), json!({"a": ":b"}));
        let behind = parser(
            "xml-inline",
            json!({"tag_pattern": r"(?<=\n)(?P<key>\w+)=(?P<value>\w*)(?!\w)"}),
        );
        assert_eq!(
            behind.parse("x=1\ny=2\nz=").unwrap(),
            json!({"y": "2", "z": ""})
        );
    }

    #[test]
    fn kv_lines_split_and_strip() {
        let kv = parser("kv-lines", json!({"value_parser": {"name": "int"}}));
        assert_eq!(
            kv.parse(" a : 1 \n\nnot a pair\nb:2").unwrap(),
            json!({"a": 1, "b": 2})
        );
    }

    #[test]
    fn invalid_arguments_are_rejected_at_load() {
        let error = |name: &str, args: Value| {
            ContentParser::new("Field 'x'", kind(name), args.as_object().unwrap()).unwrap_err()
        };
        expect_test::expect![[r#"
            [
                Invalid {
                    message: "Field 'x': content_args.strip: invalid type: string \"no\", expected a boolean",
                },
                Invalid {
                    message: "Field 'x': content_args.string_delims[0]: invalid length 1, expected a tuple of size 2",
                },
                Invalid {
                    message: "Field 'x': content_args: missing field `tag_pattern`",
                },
                Invalid {
                    message: "Field 'x': xml-inline: tag_pattern must have a named group 'key'. Pattern: <(?P<k>\\w+)>",
                },
                Invalid {
                    message: "Field 'x': content_args.value_parser.name: unknown variant `yaml`, expected one of `text`, `int`, `float`, `bool`, `json`, `xml-inline`, `kv-lines`",
                },
            ]
        "#]].assert_debug_eq(&[
            error("text", json!({"strip": "no"})),
            error("json", json!({"string_delims": [["<s>"]]})),
            error("xml-inline", json!({})),
            error("xml-inline", json!({"tag_pattern": r"<(?P<k>\w+)>"})),
            error("kv-lines", json!({"value_parser": {"name": "yaml"}})),
        ]);
        assert!(matches!(
            // Python's `regex` accepts unbounded lookbehinds; PCRE2 does not.
            error("xml-inline", json!({"tag_pattern": r"(?<=a+)(?P<key>b)"})),
            super::super::HfTemplateError::Unsupported { .. }
        ));
    }
}
