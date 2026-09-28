// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Content parsers: turn one region's captured text into a JSON value.
//!
//! Port of `content_parsers.py`. Each parser takes a chunk of captured text and
//! parses it into a single key in the output message dictionary.

use regex_automata::meta::Regex;
use regex_automata::util::syntax;
use serde::Deserialize;
use serde_json::{Map, Number, Value};
use thiserror_ext::AsReport as _;

use super::spec::{ContentKind, deserialize};
use super::{Result, invalid, unsupported, value};

// Sentinel characters for lax-JSON string pre-extraction — ASCII control chars
// that should never appear in real LLM output.
const LAX_OPEN: char = '\u{1}';
const LAX_CLOSE: char = '\u{2}';

/// A configured content parser (`content` plus `content_args` of one field).
#[derive(Debug, Clone)]
pub(super) enum ContentParser {
    Text(TextArgs),
    Int(TextArgs),
    Float(TextArgs),
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
    tag_pattern: Regex,
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
            ContentKind::Int => Self::Int(deserialize(&args, &context)?),
            ContentKind::Float => Self::Float(deserialize(&args, &context)?),
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
                let tag_pattern = compile_regex(scope, &spec.tag_pattern)?;
                let group_info = tag_pattern.group_info();
                let Some(key_group) = group_info.to_index(Default::default(), "key") else {
                    return Err(invalid!(
                        "{scope}: xml-inline: tag_pattern must have a named group 'key'. Pattern: {}",
                        spec.tag_pattern
                    ));
                };
                let value_group = group_info.to_index(Default::default(), "value");
                Self::XmlInline(XmlInlineArgs {
                    tag_pattern,
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
    pub fn parse(&self, text: &str) -> Result<Value> {
        match self {
            Self::Text(args) => Ok(Value::String(args.apply(text).to_string())),
            Self::Int(args) => {
                python_int(args.apply(text)).ok_or_else(|| value!("int: invalid literal {text:?}"))
            }
            Self::Float(args) => python_float(args.apply(text))
                .and_then(Number::from_f64)
                .map(Value::Number)
                .ok_or_else(|| value!("float: invalid or non-finite literal {text:?}")),
            Self::Bool(args) => {
                let text = args.apply(text).to_lowercase();
                Ok(Value::Bool(text == "true" || text == "1"))
            }
            Self::Json(args) => args.parse(text),
            Self::XmlInline(args) => args.parse(text),
            Self::KvLines(args) => args.parse(text),
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
    /// Apply `_text`: strip Python whitespace unless disabled.
    pub fn apply(self, text: &str) -> &str {
        if self.strip { python_strip(text) } else { text }
    }
}

impl JsonArgs {
    /// JSON parser with optional dialect knobs for LLM-emitted quirks.
    fn parse(&self, text: &str) -> Result<Value> {
        if !self.string_delims.is_empty() && text.contains([LAX_OPEN, LAX_CLOSE]) {
            return Err(value!(
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
            let quoted = serde_json::to_string(string)
                .map_err(|error| value!("json: failed to quote string: {}", error.as_report()))?;
            working = working.replace(&format!("{LAX_OPEN}{index}{LAX_CLOSE}"), &quoted);
        }

        match serde_json::from_str(&working) {
            Ok(value) => Ok(value),
            Err(_) if self.allow_non_json => Ok(Value::String(self.text.apply(text).to_string())),
            Err(error) if working == text => Err(value!(
                "json parser could not parse region as JSON.\nContent: {text:?}\nError: {}",
                error.as_report()
            )),
            Err(error) => Err(value!(
                "json: could not parse after dialect transforms.\nOriginal: {text:?}\nTransformed: {working:?}\nError: {}",
                error.as_report()
            )),
        }
    }
}

impl XmlInlineArgs {
    /// Parse shallow XML-ish tags into a dict.
    fn parse(&self, text: &str) -> Result<Value> {
        let mut out = Map::new();
        for captures in self.tag_pattern.captures_iter(text) {
            let key = captures
                .get_group(self.key_group)
                .map(|span| &text[span.range()])
                .ok_or_else(|| value!("xml-inline: named group 'key' did not participate"))?;
            let raw = match self.value_group {
                // The pattern has no `value` group.
                None => Some(""),
                Some(index) => captures.get_group(index).map(|span| &text[span.range()]),
            };
            let value = match raw {
                Some(raw) => sub_parse(raw, self.value_parser.as_deref())?,
                None if self.value_parser.is_none() => Value::Null,
                None => {
                    return Err(value!(
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
    fn parse(&self, text: &str) -> Result<Value> {
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
fn sub_parse(raw: &str, value_parser: Option<&ContentParser>) -> Result<Value> {
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

    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for captures in BARE_KEY.captures_iter(text) {
        let whole = captures.get_match().expect("match");
        let separator = captures.get_group(1).expect("group 1");
        let key = captures.get_group(2).expect("group 2");
        out.push_str(&text[last..whole.start()]);
        out.push_str(&text[separator.range()]);
        out.push('"');
        out.push_str(&text[key.range()]);
        out.push_str("\":");
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

/// Compile a Python-syntax regex with `re.DOTALL`, as Transformers does.
pub(super) fn compile_regex(scope: &str, pattern: &str) -> Result<Regex> {
    Regex::builder()
        .syntax(syntax::Config::new().dot_matches_new_line(true))
        .build(pattern)
        .map_err(|error| {
            unsupported!(
                "{scope}: regex {pattern:?} is not supported by the Rust regex engine: {}",
                error.as_report()
            )
        })
}

/// Python `str.isspace()` for one character.
///
/// Rust's `char::is_whitespace` omits the information separators U+001C..U+001F,
/// which Python treats as whitespace.
pub(super) fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub(super) fn python_strip(text: &str) -> &str {
    text.trim_matches(is_python_space)
}

/// Python `int(text)` for base-10 literals, as a JSON number.
///
/// Accepts surrounding whitespace, a sign, and single underscores between digits.
// TODO: Python also accepts non-ASCII decimal digits and integers wider than 64 bits.
pub(super) fn python_int(text: &str) -> Option<Value> {
    let digits = python_digits(python_strip(text), false)?;
    if let Ok(value) = digits.parse::<i64>() {
        return Some(Value::Number(value.into()));
    }
    digits.parse::<u64>().ok().map(|value| Value::Number(value.into()))
}

/// Python `float(text)`.
///
/// Accepts surrounding whitespace, `inf`/`infinity`/`nan` in any case, and single
/// underscores between digits. Non-finite results are returned as-is; callers
/// decide whether JSON can represent them.
pub(super) fn python_float(text: &str) -> Option<f64> {
    let text = python_strip(text);
    let unsigned = text.strip_prefix(['+', '-']).unwrap_or(text);
    if ["inf", "infinity", "nan"]
        .iter()
        .any(|special| unsigned.eq_ignore_ascii_case(special))
    {
        return text.to_ascii_lowercase().parse().ok();
    }
    python_digits(text, true)?.parse().ok()
}

/// Validate Python underscore placement and remove underscores.
///
/// With `float = false` only a sign and ASCII digits are allowed; otherwise the
/// decimal point and exponent characters are allowed as well. Rust's parser then
/// validates the remaining syntax.
fn python_digits(text: &str, float: bool) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let mut previous: Option<char> = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '_' {
            let between_digits = previous.is_some_and(|p| p.is_ascii_digit())
                && chars.peek().is_some_and(|n| n.is_ascii_digit());
            if !between_digits {
                return None;
            }
        } else if c.is_ascii_digit()
            || matches!(c, '+' | '-')
            || (float && matches!(c, '.' | 'e' | 'E'))
        {
            out.push(c);
        } else {
            return None;
        }
        previous = Some(c);
    }
    (!out.is_empty()).then_some(out)
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
    fn scalar_parsers_follow_python_conversions() {
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
            parser("text", json!({})).parse("\u{1c} a \n").unwrap(),
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
            error("xml-inline", json!({"tag_pattern": r"(?<=a)b"})),
            super::super::HfTemplateError::Unsupported { .. }
        ));
    }
}
