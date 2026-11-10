// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! The OpenAI `arguments` JSON of a tool call.

use std::io;

use serde::Serialize;
use serde_json::ser::{Formatter, Serializer};

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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::JsonStyle;

    #[test]
    fn styles_differ_only_in_separators() {
        let value = json!({"y": 1.0, "s": "a, b: c", "items": ["left", {"k": null}]});

        expect_test::expect![[r#"{"y":1.0,"s":"a, b: c","items":["left",{"k":null}]}"#]]
            .assert_eq(&JsonStyle::Compact.to_string(&value).unwrap());
        expect_test::expect![[r#"{"y": 1.0, "s": "a, b: c", "items": ["left", {"k": null}]}"#]]
            .assert_eq(&JsonStyle::Spaced.to_string(&value).unwrap());
    }
}
