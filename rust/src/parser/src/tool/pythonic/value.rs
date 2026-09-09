// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python literal values inside pythonic tool calls.
//!
//! Tool call arguments are Python literals while the OpenAI API expects JSON
//! text, so every value parsed here is converted into a `serde_json::Value`
//! (`True` -> `true`, `None` -> `null`, `'text'` -> `"text"`, ...).

use serde_json::{Number, Value};
use winnow::ascii::multispace0 as ws0;
use winnow::combinator::{alt, delimited, opt, separated, terminated};
use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};
use winnow::prelude::*;
use winnow::token::{literal, one_of, take_while};

use super::PythonicInput;
use crate::tool::Result;
use crate::utils::incomplete;
use crate::utils::recursion::ParserRecursionGuard;

/// One decoded run of a Python string literal body.
pub(super) struct StringRun {
    /// Decoded text of this run.
    pub(super) text: String,
    /// Bytes consumed from the input.
    pub(super) consumed: usize,
    /// Whether the run ended at the literal's closing quote.
    pub(super) closed: bool,
}

/// Parse the opening quote of a Python string literal.
pub(super) fn string_quote(input: &mut PythonicInput<'_>) -> ModalResult<char> {
    one_of(['\'', '"']).parse_next(input)
}

/// Parse a Python literal value.
///
/// A value is only emitted once it is complete, so a nested container is
/// buffered until its closing bracket arrives; only a top-level string argument
/// is streamed incrementally (see [`decode_string_run`]).
///
/// TODO: the Python `get_parameter_value` also accepts tuples, sets,
/// placeholder-free f-strings and non-string dict keys. Those are rejected here
/// so far; models emitting them fall back to plain text.
pub(super) fn python_value(input: &mut PythonicInput<'_>) -> ModalResult<Value> {
    alt((
        python_string.map(Value::String),
        python_list,
        python_dict,
        python_keyword,
        python_number,
    ))
    .parse_next(input)
}

/// Parse a Python string literal.
fn python_string(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    let quote = string_quote.parse_next(input)?;
    let run = decode_string_run(input, quote)?;
    if !run.closed {
        return incomplete();
    }
    Ok(run.text)
}

/// Parse a Python list literal.
fn python_list(input: &mut PythonicInput<'_>) -> ModalResult<Value> {
    literal("[").parse_next(input)?;
    let _guard = ParserRecursionGuard::enter()?;
    terminated(python_items, literal("]")).map(Value::Array).parse_next(input)
}

/// Parse the comma-separated items of a Python list literal.
fn python_items(input: &mut PythonicInput<'_>) -> ModalResult<Vec<Value>> {
    delimited(
        ws0,
        terminated(
            separated(0.., python_value, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)
}

/// Parse a Python dict literal.
fn python_dict(input: &mut PythonicInput<'_>) -> ModalResult<Value> {
    literal("{").parse_next(input)?;
    let _guard = ParserRecursionGuard::enter()?;
    terminated(python_entries, literal("}"))
        .map(|entries: Vec<(String, Value)>| Value::Object(entries.into_iter().collect()))
        .parse_next(input)
}

/// Parse the comma-separated entries of a Python dict literal.
fn python_entries(input: &mut PythonicInput<'_>) -> ModalResult<Vec<(String, Value)>> {
    delimited(
        ws0,
        terminated(
            separated(0.., python_entry, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)
}

/// Parse one `'key': value` entry of a Python dict literal.
///
/// JSON object keys are strings, so only string keys are accepted here.
fn python_entry(input: &mut PythonicInput<'_>) -> ModalResult<(String, Value)> {
    (
        python_string,
        delimited(ws0, literal(":"), ws0),
        python_value,
    )
        .map(|(key, _, value)| (key, value))
        .parse_next(input)
}

/// Parse a comma separator between Python literals.
fn comma_separator(input: &mut PythonicInput<'_>) -> ModalResult<()> {
    delimited(ws0, literal(","), ws0).void().parse_next(input)
}

/// Parse a Python keyword literal.
///
/// JSON-style spellings are accepted as well because some models (e.g. OLMo 3)
/// emit `true` / `false` / `null` inside otherwise pythonic calls, matching
/// `_JSON_NAME_LITERALS` in the Python parser.
fn python_keyword(input: &mut PythonicInput<'_>) -> ModalResult<Value> {
    alt((
        literal("True").value(Value::Bool(true)),
        literal("False").value(Value::Bool(false)),
        literal("None").value(Value::Null),
        literal("true").value(Value::Bool(true)),
        literal("false").value(Value::Bool(false)),
        literal("null").value(Value::Null),
    ))
    .parse_next(input)
}

/// Parse a Python numeric literal.
///
/// Signs are part of the literal here; Python parses `-1` as a unary operation
/// over a constant, which the Python parser converts back into a number.
fn python_number(input: &mut PythonicInput<'_>) -> ModalResult<Value> {
    let raw = take_while(1.., ('0'..='9', '.', 'e', 'E', '+', '-')).parse_next(input)?;
    number_value(raw).ok_or_else(|| cut_error("Python number"))
}

/// Convert a Python numeric literal into a JSON number.
fn number_value(raw: &str) -> Option<Value> {
    if let Ok(value) = raw.parse::<i64>() {
        return Some(value.into());
    }
    if let Ok(value) = raw.parse::<u64>() {
        return Some(value.into());
    }
    // Non-finite floats have no JSON representation and are rejected here, like
    // `_is_json_finite` does on the Python side.
    Number::from_f64(raw.parse::<f64>().ok()?).map(Value::Number)
}

/// Decode the body of a Python string literal up to its closing `quote`.
///
/// Decoding stops at the closing quote or at the last complete escape sequence,
/// so a trailing partial escape stays buffered until the next chunk arrives.
/// The input is advanced past the decoded run.
pub(super) fn decode_string_run(
    input: &mut PythonicInput<'_>,
    quote: char,
) -> ModalResult<StringRun> {
    let text = **input;
    let mut decoded = String::new();
    let mut index = 0;

    while let Some(char) = text[index..].chars().next() {
        if char == quote {
            let consumed = index + char.len_utf8();
            input.next_slice(consumed);
            return Ok(StringRun {
                text: decoded,
                consumed,
                closed: true,
            });
        }
        if char == '\\' {
            let Some(len) = decode_escape(&text[index..], &mut decoded)? else {
                break;
            };
            index += len;
            continue;
        }
        decoded.push(char);
        index += char.len_utf8();
    }

    input.next_slice(index);
    Ok(StringRun {
        text: decoded,
        consumed: index,
        closed: false,
    })
}

/// Decode one Python escape sequence at the start of `text`.
///
/// Returns the number of bytes consumed, or `None` when the sequence is cut
/// short by the end of the buffered input.
fn decode_escape(text: &str, decoded: &mut String) -> ModalResult<Option<usize>> {
    let Some(escape) = text[1..].chars().next() else {
        return Ok(None);
    };
    let end = 1 + escape.len_utf8();

    match escape {
        // A backslash before a line break is a line continuation.
        '\n' => Ok(Some(end)),
        '\\' | '\'' | '"' => Ok(Some(push_escaped(decoded, escape, end))),
        'a' => Ok(Some(push_escaped(decoded, '\u{7}', end))),
        'b' => Ok(Some(push_escaped(decoded, '\u{8}', end))),
        'f' => Ok(Some(push_escaped(decoded, '\u{c}', end))),
        'n' => Ok(Some(push_escaped(decoded, '\n', end))),
        'r' => Ok(Some(push_escaped(decoded, '\r', end))),
        't' => Ok(Some(push_escaped(decoded, '\t', end))),
        'v' => Ok(Some(push_escaped(decoded, '\u{b}', end))),
        '0'..='7' => decode_octal_escape(text, decoded),
        'x' => decode_fixed_escape(text, decoded, 2),
        'u' => decode_unicode_escape(text, decoded),
        'U' => decode_fixed_escape(text, decoded, 8),
        // Python keeps unknown escapes such as `\d` verbatim (with a syntax
        // warning that the Python parser suppresses), and so does this.
        // TODO: `\N{NAME}` named escapes need a Unicode name table and are
        // kept verbatim for now.
        _ => {
            decoded.push('\\');
            Ok(Some(push_escaped(decoded, escape, end)))
        }
    }
}

/// Push one decoded escape character and return the bytes it consumed.
fn push_escaped(decoded: &mut String, char: char, end: usize) -> usize {
    decoded.push(char);
    end
}

/// Decode a Python octal escape such as `\101`.
fn decode_octal_escape(text: &str, decoded: &mut String) -> ModalResult<Option<usize>> {
    const MAX_DIGITS: usize = 3;

    let mut end = 1;
    let mut value = 0;
    for digit in text[1..].chars().take(MAX_DIGITS).map_while(|char| char.to_digit(8)) {
        value = value * 8 + digit;
        end += 1;
    }

    // Fewer than three digits at the end of the buffer may still grow.
    if end < 1 + MAX_DIGITS && end == text.len() {
        return Ok(None);
    }
    let Some(char) = char::from_u32(value) else {
        return Err(cut_error("Python string escape"));
    };
    Ok(Some(push_escaped(decoded, char, end)))
}

/// Decode a fixed-width Python escape such as `\xff` or `\U0001f600`.
fn decode_fixed_escape(
    text: &str,
    decoded: &mut String,
    digits: usize,
) -> ModalResult<Option<usize>> {
    let Some((value, end)) = escape_code_point(text, digits)? else {
        return Ok(None);
    };
    let Some(char) = char::from_u32(value) else {
        return Err(cut_error("Python string escape"));
    };
    Ok(Some(push_escaped(decoded, char, end)))
}

/// Decode a `\uXXXX` escape, pairing surrogates like JSON does.
fn decode_unicode_escape(text: &str, decoded: &mut String) -> ModalResult<Option<usize>> {
    const LEADING: std::ops::RangeInclusive<u32> = 0xD800..=0xDBFF;
    const TRAILING: std::ops::RangeInclusive<u32> = 0xDC00..=0xDFFF;

    let Some((leading, end)) = escape_code_point(text, 4)? else {
        return Ok(None);
    };
    if !LEADING.contains(&leading) {
        return decode_fixed_escape(text, decoded, 4);
    }

    // A leading surrogate only forms a character together with the trailing
    // surrogate that follows it, like a JSON `😀` pair.
    let rest = &text[end..];
    if !rest.starts_with("\\u") {
        if "\\u".starts_with(rest) {
            return Ok(None);
        }
        return Err(cut_error("Python string escape"));
    }
    let Some((trailing, trailing_end)) = escape_code_point(rest, 4)? else {
        return Ok(None);
    };
    if !TRAILING.contains(&trailing) {
        return Err(cut_error("Python string escape"));
    }

    let value = 0x10000 + ((leading - 0xD800) << 10) + (trailing - 0xDC00);
    let Some(char) = char::from_u32(value) else {
        return Err(cut_error("Python string escape"));
    };
    Ok(Some(push_escaped(decoded, char, end + trailing_end)))
}

/// Read the `digits` hex digits of an escape sequence at the start of `text`.
///
/// Returns the code point and the byte offset just past the sequence, or `None`
/// when the buffered text may still grow into a complete sequence.
fn escape_code_point(text: &str, digits: usize) -> ModalResult<Option<(u32, usize)>> {
    let end = 2 + digits;
    let hex_digits = |text: &str| text.chars().all(|char| char.is_ascii_hexdigit());
    let Some(raw) = text.get(2..end).filter(|raw| hex_digits(raw)) else {
        // A short tail of hex digits can still be completed by the next chunk.
        if hex_digits(&text[2..]) {
            return Ok(None);
        }
        return Err(cut_error("Python string escape"));
    };
    let value = u32::from_str_radix(raw, 16).map_err(|_| cut_error("Python string escape"))?;
    Ok(Some((value, end)))
}

/// Encode a key as a quoted JSON object key.
pub(super) fn json_object_key(key: &str) -> Result<String> {
    serde_json::to_string(key)
        .map_err(|error| parsing_failed!("failed to serialize argument name: {}", error))
}

/// Encode decoded text as JSON string content, without the enclosing quotes.
///
/// String arguments are streamed between the quotes emitted when the argument
/// starts, so only the escaped content belongs in the delta.
pub(super) fn json_string_content(text: &str) -> Result<String> {
    let encoded = serde_json::to_string(text)
        .map_err(|error| parsing_failed!("failed to serialize string argument: {}", error))?;
    encoded
        .strip_prefix('"')
        .and_then(|content| content.strip_suffix('"'))
        .map(str::to_string)
        .ok_or_else(|| parsing_failed!("JSON string argument is not quoted"))
}

/// Build a cut error for an invalid Python literal.
fn cut_error(label: &'static str) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    ErrMode::Cut(error)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use winnow::error::ErrMode;
    use winnow::stream::Partial;

    use super::{PythonicInput, python_value};

    fn parse(text: &str) -> String {
        let mut input = Partial::new(text);
        let value = python_value(&mut input).expect("value should parse");
        format!("{value} | rest {:?}", *input)
    }

    fn error(text: &str) -> ErrMode<winnow::error::ContextError> {
        let mut input: PythonicInput<'_> = Partial::new(text);
        python_value(&mut input).expect_err("value should not parse")
    }

    #[test]
    fn python_value_converts_scalars_to_json() {
        expect![[r#"37 | rest " ""#]].assert_eq(&parse("37 "));
        expect!["-2.5 | rest \")\""].assert_eq(&parse("-2.5)"));
        expect!["1000.0 | rest \")\""].assert_eq(&parse("1e3)"));
        expect!["true | rest \"\""].assert_eq(&parse("True"));
        expect!["false | rest \"\""].assert_eq(&parse("False"));
        expect!["null | rest \"\""].assert_eq(&parse("None"));
        expect!["true | rest \"\""].assert_eq(&parse("true"));
        expect!["null | rest \"\""].assert_eq(&parse("null"));
    }

    #[test]
    fn python_value_decodes_string_escapes() {
        expect![[r#""Martha's Vineyard" | rest """#]].assert_eq(&parse(r"'Martha\'s Vineyard'"));
        expect![[r#""\"cool units\"" | rest """#]].assert_eq(&parse(r#"'\"cool units\"'"#));
        expect![[r#""a\nb\tc" | rest """#]].assert_eq(&parse(r"'a\nb\tc'"));
        expect![[r#""ünïcødé ☃ 😀" | rest """#]].assert_eq(&parse(r"'ün\xefc\xf8dé ☃ 😀'"));
        expect![[r#""A\\d" | rest """#]].assert_eq(&parse(r"'\101\d'"));
        expect![[r#""one line" | rest """#]].assert_eq(&parse("'one \\\nline'"));
    }

    #[test]
    fn python_value_parses_nested_containers() {
        expect![[r#"{"city":"SF","tags":["a","b"],"nested":{"deep":[1,{"ok":true}]}} | rest """#]]
            .assert_eq(&parse(
                "{'city': 'SF', 'tags': ['a', 'b'], 'nested': {'deep': [1, {'ok': True}]}}",
            ));
        expect![[r#"[] | rest """#]].assert_eq(&parse("[]"));
        expect![[r#"{} | rest """#]].assert_eq(&parse("{}"));
        expect![[r#"[1,2] | rest """#]].assert_eq(&parse("[1, 2,]"));
    }

    #[test]
    fn python_value_reports_incomplete_for_partial_literals() {
        for text in [
            "'unterminated",
            r"'trailing escape \",
            r"'short unicode \u12",
            r"'lone leading surrogate \ud83d",
            "[1, 2",
            "{'k': ",
            "Tru",
            "1.5",
        ] {
            assert!(
                matches!(error(text), ErrMode::Incomplete(_)),
                "{text:?} should be incomplete"
            );
        }
    }

    #[test]
    fn python_value_rejects_unsupported_literals() {
        for text in [
            "f'formatted'",
            "(1, 2)",
            "{1: 'int key'}",
            "'bad \\ud83d\\u0041'",
        ] {
            assert!(
                !matches!(error(text), ErrMode::Incomplete(_)),
                "{text:?} should fail"
            );
        }
    }
}
