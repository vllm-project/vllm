// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The OpenAI `arguments` JSON of a tool call.
//!
//! Converted-format parsers recognize one argument at a time. Only a string
//! value streams as it arrives, since escaping maps every text into JSON the
//! same way; any other argument is written whole when it closes, once its
//! conversion is known. [`ArgumentsWriter`] owns the JSON syntax: separators,
//! keys, quotes, escaping, duplicate keys, and validating JSON text the model
//! wrote.

use std::collections::HashSet;
use std::io;

use serde::de::IgnoredAny;
use serde::{Serialize, Serializer as _};
use serde_json::Value;
use serde_json::ser::{Formatter, Serializer};

use super::ToolCallDelta;

/// JSON separators of the tool-call `arguments` a parser writes.
///
/// A parser that serializes `arguments` itself uses the style its format's
/// chat template renders JSON in, so parsing a rendered tool call gives back
/// the same text. A template that renders no JSON gets [`JsonStyle::Spaced`],
/// the default of HF `tojson` and Python `json.dumps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonStyle {
    /// `,` and `:`, serde_json's default.
    Compact,
    /// `, ` and `: `, Python `json.dumps`'s default, which HF `tojson` uses.
    Spaced,
}

impl JsonStyle {
    /// Separator between array elements and between object members.
    pub(crate) fn comma(self) -> &'static str {
        match self {
            Self::Compact => ",",
            Self::Spaced => ", ",
        }
    }

    /// Separator between an object key and its value.
    pub(crate) fn colon(self) -> &'static str {
        match self {
            Self::Compact => ":",
            Self::Spaced => ": ",
        }
    }

    /// Serialize `value` as JSON in this style, like [`serde_json::to_string`].
    pub(crate) fn to_string<T: ?Sized + Serialize>(self, value: &T) -> serde_json::Result<String> {
        let mut json = Vec::new();
        self.to_writer(&mut json, value)?;
        Ok(String::from_utf8(json).expect("serde_json writes UTF-8"))
    }

    /// Serialize `value` as JSON in this style into `writer`, like
    /// [`serde_json::to_writer`].
    pub(crate) fn to_writer<W: io::Write, T: ?Sized + Serialize>(
        self,
        writer: W,
        value: &T,
    ) -> serde_json::Result<()> {
        /// serde_json's compact formatter with a space after `,` and `:`.
        struct SpacedFormatter;

        impl Formatter for SpacedFormatter {
            fn begin_array_value<W: ?Sized + io::Write>(
                &mut self,
                writer: &mut W,
                first: bool,
            ) -> io::Result<()> {
                if first {
                    Ok(())
                } else {
                    writer.write_all(b", ")
                }
            }

            fn begin_object_key<W: ?Sized + io::Write>(
                &mut self,
                writer: &mut W,
                first: bool,
            ) -> io::Result<()> {
                if first {
                    Ok(())
                } else {
                    writer.write_all(b", ")
                }
            }

            fn begin_object_value<W: ?Sized + io::Write>(
                &mut self,
                writer: &mut W,
            ) -> io::Result<()> {
                writer.write_all(b": ")
            }
        }

        match self {
            Self::Compact => serde_json::to_writer(writer, value),
            Self::Spaced => {
                value.serialize(&mut Serializer::with_formatter(writer, SpacedFormatter))
            }
        }
    }
}

/// How a streamed argument's value text maps into the arguments JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueSyntax {
    /// Raw text of a JSON string: escaped as it arrives, quoted by the writer.
    String,
    /// JSON text: appended as it arrives.
    Json,
}

/// Incremental writer of one tool call's `arguments` JSON object, in
/// generation order.
///
/// It keeps what it wrote since the last [`take`](Self::take), so a parser
/// writes the arguments of one event and then emits them as one delta. A later
/// duplicate key is dropped whole: the first value may already be on the wire
/// and cannot be retracted.
#[derive(Debug)]
pub(crate) struct ArgumentsWriter {
    /// Separators of the object and of whole values.
    style: JsonStyle,
    /// Keys written so far.
    keys: HashSet<String>,
    /// The argument being streamed, if any.
    open: Option<OpenArgument>,
    /// Text written since the last `take`. serde_json serializes into it
    /// directly; it is UTF-8 because serde_json and every `&str` appended are.
    pending: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
enum OpenArgument {
    Writing(ValueSyntax),
    /// A duplicate key, whose value is dropped.
    Skipping,
}

impl ArgumentsWriter {
    /// Create the writer of one call's arguments in `style`.
    pub(crate) fn new(style: JsonStyle) -> Self {
        Self {
            style,
            keys: HashSet::new(),
            open: None,
            pending: Vec::new(),
        }
    }

    /// Start a streamed string argument, whose fragments are raw text escaped
    /// as they arrive.
    pub(crate) fn open_string(&mut self, key: &str) {
        self.open(key, ValueSyntax::String);
    }

    /// Start a streamed argument whose fragments are JSON text appended
    /// unchanged, which nothing checks. No parser uses it: JSON the model
    /// writes may be malformed, so parsers buffer it for
    /// [`write_json`](Self::write_json).
    #[allow(dead_code)]
    pub(crate) fn open_raw_json(&mut self, key: &str) {
        self.open(key, ValueSyntax::Json);
    }

    /// Start a streamed argument: `{` or a comma, the key, and a colon, plus
    /// `"` for [`ValueSyntax::String`].
    fn open(&mut self, key: &str, syntax: ValueSyntax) {
        debug_assert!(self.open.is_none(), "an argument is already open");
        let open = if self.write_key(key) {
            if syntax == ValueSyntax::String {
                self.pending.push(b'"');
            }
            OpenArgument::Writing(syntax)
        } else {
            OpenArgument::Skipping
        };
        self.open = Some(open);
    }

    /// Append one fragment of the open argument's value.
    pub(crate) fn push(&mut self, text: &str) {
        match self.open {
            Some(OpenArgument::Writing(ValueSyntax::String)) => {
                push_escaped(text, &mut self.pending);
            }
            Some(OpenArgument::Writing(ValueSyntax::Json)) => {
                self.pending.extend_from_slice(text.as_bytes());
            }
            Some(OpenArgument::Skipping) => {}
            None => debug_assert!(false, "no argument is open"),
        }
    }

    /// End the open argument: `"` for [`ValueSyntax::String`].
    pub(crate) fn close(&mut self) {
        match self.open.take() {
            Some(OpenArgument::Writing(ValueSyntax::String)) => self.pending.push(b'"'),
            Some(OpenArgument::Writing(ValueSyntax::Json) | OpenArgument::Skipping) => {}
            None => debug_assert!(false, "no argument is open"),
        }
    }

    /// Write a whole argument whose value the parser already converted.
    pub(crate) fn write_value(&mut self, key: &str, value: &Value) {
        debug_assert!(self.open.is_none(), "an argument is already open");
        if self.write_key(key) {
            self.style.to_writer(&mut self.pending, value).expect("a JSON value serializes");
        }
    }

    /// Write a whole argument from JSON text the model wrote: the text itself,
    /// without surrounding whitespace, when it is valid JSON, and otherwise a
    /// JSON string of it, so a malformed value cannot break the object.
    ///
    /// Valid text is kept as spelled rather than re-serialized, so numbers keep
    /// their spelling (`1.50`, `1e-05`) and a value rendered by the template
    /// parses back to the same text.
    pub(crate) fn write_json(&mut self, key: &str, text: &str) {
        debug_assert!(self.open.is_none(), "an argument is already open");
        if !self.write_key(key) {
            return;
        }
        let json = text.trim_matches([' ', '\t', '\n', '\r']);
        if serde_json::from_str::<IgnoredAny>(json).is_ok() {
            self.pending.extend_from_slice(json.as_bytes());
        } else {
            serde_json::to_writer(&mut self.pending, text).expect("a string serializes");
        }
    }

    /// End the object: `}`, or `{}` when no argument was written.
    pub(crate) fn finish(&mut self) {
        debug_assert!(self.open.is_none(), "an argument is still open");
        let end: &[u8] = if self.keys.is_empty() { b"{}" } else { b"}" };
        self.pending.extend_from_slice(end);
    }

    /// Take the text written since the last call.
    pub(crate) fn take(&mut self) -> String {
        String::from_utf8(std::mem::take(&mut self.pending)).expect("arguments text is UTF-8")
    }

    /// Write `{` or a comma, the key, and a colon, unless `key` was already
    /// written.
    fn write_key(&mut self, key: &str) -> bool {
        if self.keys.contains(key) {
            return false;
        }
        if self.keys.is_empty() {
            self.pending.push(b'{');
        } else {
            self.pending.extend_from_slice(self.style.comma().as_bytes());
        }
        serde_json::to_writer(&mut self.pending, key).expect("a string serializes");
        self.pending.extend_from_slice(self.style.colon().as_bytes());
        self.keys.insert(key.to_string());
        true
    }
}

/// A tool call whose arguments a parser streams: its function name, output
/// index, and arguments writer.
#[derive(Debug)]
pub(crate) struct OpenToolCall {
    /// The called function, for schema lookups.
    pub(crate) name: String,
    tool_index: usize,
    /// Whether a delta carried the name already.
    named: bool,
    /// The arguments the parser writes.
    pub(crate) arguments: ArgumentsWriter,
}

impl OpenToolCall {
    /// Open the call at `tool_index` with arguments in `style`.
    pub(crate) fn new(name: String, tool_index: usize, style: JsonStyle) -> Self {
        Self {
            name,
            tool_index,
            named: false,
            arguments: ArgumentsWriter::new(style),
        }
    }

    /// The delta of what was written since the last one: the name first, then
    /// new arguments text; `None` when nothing is new.
    pub(crate) fn take_delta(&mut self) -> Option<ToolCallDelta> {
        let arguments = self.arguments.take();
        if self.named && arguments.is_empty() {
            return None;
        }
        let name = (!self.named).then(|| self.name.clone());
        self.named = true;
        Some(ToolCallDelta {
            tool_index: self.tool_index,
            name,
            arguments,
        })
    }

    /// End the arguments object and return the call's last delta.
    pub(crate) fn finish(mut self) -> ToolCallDelta {
        self.arguments.finish();
        self.take_delta().expect("the object end is new text")
    }
}

/// Append `text` escaped as JSON string contents, without quotes.
///
/// serde_json escapes one byte at a time, so escaping fragments of a string
/// one by one yields the same bytes as escaping the whole string.
fn push_escaped(text: &str, out: &mut Vec<u8>) {
    /// A compact formatter that writes string contents without the surrounding
    /// quotes, so a string value can be escaped one fragment at a time.
    struct FragmentFormatter;

    impl Formatter for FragmentFormatter {
        fn begin_string<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
            Ok(())
        }

        fn end_string<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
            Ok(())
        }
    }

    Serializer::with_formatter(out, FragmentFormatter)
        .serialize_str(text)
        .expect("a string serializes");
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use serde_json::json;

    use super::{ArgumentsWriter, JsonStyle};

    /// Text exercising every class serde_json escapes, plus multi-byte chars.
    const TRICKY: &str = "a\"b\\c\nd\te\u{1}f/g\u{7f}h\u{2028}i\u{e9}j\u{1f600}k";

    #[test]
    fn string_fragments_escape_like_the_whole_string() {
        let whole = serde_json::to_string(TRICKY).unwrap();
        let boundaries = TRICKY.char_indices().map(|(at, _)| at).chain([TRICKY.len()]);
        for split in boundaries {
            let mut writer = ArgumentsWriter::new(JsonStyle::Compact);
            writer.open_string("k");
            writer.push(&TRICKY[..split]);
            let mut out = writer.take();
            writer.push("");
            writer.push(&TRICKY[split..]);
            writer.close();
            writer.finish();
            out.push_str(&writer.take());
            assert_eq!(out, format!("{{\"k\":{whole}}}"), "split {split}");
        }
    }

    #[test]
    fn writes_arguments_in_generation_order() {
        let mut writer = ArgumentsWriter::new(JsonStyle::Compact);
        writer.open_string("path");
        writer.push("a.txt");
        writer.close();
        writer.write_value("count", &json!(2));
        writer.open_raw_json("payload");
        writer.push("{\"x\": [1,");
        writer.push(" 2]}");
        writer.close();
        writer.open_string("empty");
        writer.close();
        writer.finish();

        expect![[r#"{"path":"a.txt","count":2,"payload":{"x": [1, 2]},"empty":""}"#]]
            .assert_eq(&writer.take());
    }

    #[test]
    fn writes_the_bytes_of_the_whole_object_in_its_style() {
        let whole = json!({"path": "a.txt", "count": 2, "payload": {"x": [1, null]}});
        for style in [JsonStyle::Compact, JsonStyle::Spaced] {
            let mut writer = ArgumentsWriter::new(style);
            writer.open_string("path");
            writer.push("a.txt");
            writer.close();
            writer.write_value("count", &json!(2));
            writer.write_value("payload", &json!({"x": [1, null]}));
            writer.finish();
            assert_eq!(writer.take(), style.to_string(&whole).unwrap(), "{style:?}");
        }
    }

    #[test]
    fn keeps_the_first_value_of_a_duplicate_key() {
        let mut writer = ArgumentsWriter::new(JsonStyle::Compact);
        writer.open_string("k");
        writer.push("first");
        writer.close();
        writer.open_raw_json("k");
        writer.push("2");
        writer.close();
        writer.write_json("k", "2");
        writer.write_value("k", &json!(3));
        writer.write_value("other", &json!(null));
        writer.finish();

        expect![[r#"{"k":"first","other":null}"#]].assert_eq(&writer.take());
    }

    #[test]
    fn keeps_valid_json_text_as_spelled_and_writes_the_rest_as_strings() {
        let mut writer = ArgumentsWriter::new(JsonStyle::Spaced);
        for (key, text) in [
            ("decimal", "1.50"),
            ("exponent", "1e-05"),
            ("object", "{\"x\":[1, null]}"),
            ("padded", "\n [true] \n"),
            ("word", "Hangzhou"),
            ("python", "{'a': True}"),
            ("nan", "NaN"),
            ("two", "1 2"),
            ("blank", " "),
            ("empty", ""),
        ] {
            writer.write_json(key, text);
        }
        writer.finish();

        expect![[r#"{"decimal": 1.50, "exponent": 1e-05, "object": {"x":[1, null]}, "padded": [true], "word": "Hangzhou", "python": "{'a': True}", "nan": "NaN", "two": "1 2", "blank": " ", "empty": ""}"#]]
            .assert_eq(&writer.take());
    }

    #[test]
    fn escapes_keys_and_writes_an_empty_object() {
        let mut writer = ArgumentsWriter::new(JsonStyle::Compact);
        writer.finish();
        expect!["{}"].assert_eq(&writer.take());

        let mut writer = ArgumentsWriter::new(JsonStyle::Compact);
        writer.write_value("a\"b", &json!("v"));
        writer.finish();
        expect![[r#"{"a\"b":"v"}"#]].assert_eq(&writer.take());
    }

    #[test]
    fn styles_differ_only_in_separators() {
        let value = json!({"y": 1.0, "s": "a, b: c", "items": ["left", {"k": null}]});

        expect_test::expect![[r#"{"y":1.0,"s":"a, b: c","items":["left",{"k":null}]}"#]]
            .assert_eq(&JsonStyle::Compact.to_string(&value).unwrap());
        expect_test::expect![[r#"{"y": 1.0, "s": "a, b: c", "items": ["left", {"k": null}]}"#]]
            .assert_eq(&JsonStyle::Spaced.to_string(&value).unwrap());
    }
}
