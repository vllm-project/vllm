// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Python literal values inside pythonic tool calls.
//!
//! Tool call arguments are Python literals while the OpenAI API expects JSON
//! text, so every value parsed here is converted into compact JSON text
//! (`True` -> `true`, `None` -> `null`, `'text'` -> `"text"`, ...).

use serde_json::Number;
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

/// Parse a Python literal value into compact JSON text.
///
/// A value is only emitted once it is complete, so a nested container is
/// buffered until its closing bracket arrives; only a top-level string argument
/// is streamed incrementally (see [`decode_string_run`]).
///
/// Values are built as JSON text rather than as a `serde_json::Value` because
/// `serde_json::Number` cannot hold an integer wider than `u64`. Rounding one
/// through `f64` would silently change a tool call argument, while the Python
/// parser passes it through exactly.
///
/// TODO: the Python `get_parameter_value` also accepts sets, placeholder-free
/// f-strings, triple-quoted strings and non-string dict keys. Those are
/// rejected here so far; models emitting them fall back to plain text.
pub(super) fn python_value(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    alt((
        json_string,
        python_sequence,
        python_dict,
        python_keyword,
        python_number,
    ))
    .parse_next(input)
}

/// Parse a Python string literal into a quoted JSON string.
fn json_string(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    let text = python_string.parse_next(input)?;
    let content = json_string_content(&text).map_err(|_| cut_error("Python string"))?;
    Ok(format!("\"{content}\""))
}

/// Parse a Python string literal.
fn python_string(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    let quote = string_quote.parse_next(input)?;
    let run = decode_string_run(input, quote, false)?;
    if !run.closed {
        return incomplete();
    }
    Ok(run.text)
}

/// Parse a Python list or tuple literal into a JSON array.
///
/// Tuples have no JSON counterpart, so the Python parser turns them into arrays
/// and so does this. Parentheses without a trailing comma are a grouping rather
/// than a tuple, as in Python: `(1)` is `1` while `(1,)` is `[1]`.
fn python_sequence(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    let closing = alt((literal("[").value("]"), literal("(").value(")"))).parse_next(input)?;
    let _guard = ParserRecursionGuard::enter()?;
    let (items, trailing_comma) = terminated(python_items, literal(closing)).parse_next(input)?;
    if closing == ")" && items.len() == 1 && !trailing_comma {
        return Ok(items.into_iter().next().unwrap_or_default());
    }
    Ok(format!("[{}]", items.join(",")))
}

/// Parse the comma-separated items of a Python sequence literal.
///
/// Reports whether the items ended with a trailing comma, which is what
/// distinguishes a one-element tuple from a parenthesised value.
fn python_items(input: &mut PythonicInput<'_>) -> ModalResult<(Vec<String>, bool)> {
    let (items, trailing_comma) = delimited(
        ws0,
        (
            separated(0.., python_value, comma_separator),
            opt(comma_separator),
        ),
        ws0,
    )
    .parse_next(input)?;
    Ok((items, trailing_comma.is_some()))
}

/// Parse a Python dict literal into a JSON object.
fn python_dict(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    literal("{").parse_next(input)?;
    let _guard = ParserRecursionGuard::enter()?;
    let entries: Vec<String> = terminated(python_entries, literal("}")).parse_next(input)?;
    Ok(format!("{{{}}}", entries.join(",")))
}

/// Parse the comma-separated entries of a Python dict literal.
fn python_entries(input: &mut PythonicInput<'_>) -> ModalResult<Vec<String>> {
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
fn python_entry(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    let (key, _, value) = (
        python_string,
        delimited(ws0, literal(":"), ws0),
        python_value,
    )
        .parse_next(input)?;
    let key = json_object_key(&key).map_err(|_| cut_error("Python dict key"))?;
    Ok(format!("{key}:{value}"))
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
fn python_keyword(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    alt((
        literal("True").value("true"),
        literal("False").value("false"),
        literal("None").value("null"),
        literal("true").value("true"),
        literal("false").value("false"),
        literal("null").value("null"),
    ))
    .map(|keyword: &str| keyword.to_string())
    .parse_next(input)
}

/// Parse a Python numeric literal into JSON number text.
///
/// Signs are part of the literal here; Python parses `-1` as a unary operation
/// over a constant, which the Python parser converts back into a number.
fn python_number(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    alt((radix_integer, decimal_number)).parse_next(input)
}

/// Parse a binary, octal or hexadecimal Python integer literal.
fn radix_integer(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    let (sign, _, marker, digits) = (
        opt(one_of(['+', '-'])),
        literal("0"),
        one_of(['b', 'B', 'o', 'O', 'x', 'X']),
        take_while(1.., ('0'..='9', 'a'..='f', 'A'..='F', '_')),
    )
        .parse_next(input)?;
    let radix = match marker {
        'b' | 'B' => 2,
        'o' | 'O' => 8,
        _ => 16,
    };
    let digits = digits.replace('_', "");
    let value =
        i128::from_str_radix(&digits, radix).map_err(|_| cut_error("Python integer literal"))?;
    let value = if sign == Some('-') { -value } else { value };
    Ok(value.to_string())
}

/// Parse a decimal Python integer or float literal.
fn decimal_number(input: &mut PythonicInput<'_>) -> ModalResult<String> {
    let raw = take_while(1.., ('0'..='9', '.', 'e', 'E', '+', '-', '_')).parse_next(input)?;
    number_json(raw).ok_or_else(|| cut_error("Python number"))
}

/// Convert a decimal Python numeric literal into JSON number text.
///
/// Integers keep their digits so values wider than `u64` stay exact; only
/// floats go through `f64`. Non-finite floats have no JSON representation and
/// are rejected here, like `_is_json_finite` does on the Python side.
fn number_json(raw: &str) -> Option<String> {
    let raw = raw.replace('_', "");
    let digits = raw.strip_prefix(['+', '-']).unwrap_or(&raw);
    if !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()) {
        // JSON rejects leading zeros, and so does Python outside of `0`.
        let digits = digits.trim_start_matches('0');
        let sign = if raw.starts_with('-') { "-" } else { "" };
        return Some(match digits {
            "" => "0".to_string(),
            digits => format!("{sign}{digits}"),
        });
    }
    Number::from_f64(raw.parse::<f64>().ok()?).map(|number| number.to_string())
}

/// Decode the body of a Python string literal up to its closing `quote`.
///
/// Decoding stops at the closing quote or at the last complete escape sequence,
/// so a trailing partial escape stays buffered until the next chunk arrives.
/// The input is advanced past the decoded run.
///
/// `commit_decoded` selects what happens when an invalid escape follows text
/// that has already been decoded. A streamed argument sets it so the decoded
/// text is returned and the escape is reported on the next call, because a
/// chunk boundary just before the escape would have committed that text too and
/// committed output cannot be taken back. A string inside a container clears it
/// and the invalid escape is reported right away.
pub(super) fn decode_string_run(
    input: &mut PythonicInput<'_>,
    quote: char,
    commit_decoded: bool,
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
            match decode_escape(&text[index..], &mut decoded) {
                Ok(Some(len)) => {
                    index += len;
                    continue;
                }
                Ok(None) => break,
                Err(_) if commit_decoded && index > 0 => break,
                Err(error) => return Err(error),
            }
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
        // Decoding `\N{NAME}` needs a Unicode name table. Keeping it verbatim
        // would hand the tool a different string than the model wrote, so the
        // call is rejected and the text falls back to content instead.
        // TODO: decode named escapes once a name table is available.
        'N' => Err(cut_error("Python named string escape")),
        // Python keeps unknown escapes such as `\d` verbatim (with a syntax
        // warning that the Python parser suppresses), and so does this.
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

    /// Python has no JSON counterpart for tuples, so the Python parser turns
    /// them into arrays. Parentheses without a trailing comma group a value.
    #[test]
    fn python_value_converts_tuples_to_arrays() {
        expect![[r#"[1,2] | rest """#]].assert_eq(&parse("(1, 2)"));
        expect![[r#"[1] | rest """#]].assert_eq(&parse("(1,)"));
        expect![[r#"1 | rest """#]].assert_eq(&parse("(1)"));
        expect![[r#"[] | rest """#]].assert_eq(&parse("()"));
        expect![[r#"[[1,2],["a"]] | rest """#]].assert_eq(&parse("[(1, 2), ('a',)]"));
    }

    /// `serde_json::Number` tops out at `u64`, so wide integers are carried
    /// through as text; rounding them into an `f64` would silently change the
    /// argument the tool receives.
    #[test]
    fn python_value_keeps_wide_integers_exact() {
        expect!["123456789012345678901234567890 | rest \")\""]
            .assert_eq(&parse("123456789012345678901234567890)"));
        expect!["-9223372036854775809 | rest \")\""].assert_eq(&parse("-9223372036854775809)"));
        expect!["18446744073709551615 | rest \")\""].assert_eq(&parse("18446744073709551615)"));
        expect!["9007199254740993 | rest \")\""].assert_eq(&parse("9007199254740993)"));
        expect!["0 | rest \")\""].assert_eq(&parse("-0)"));
        expect!["7 | rest \")\""].assert_eq(&parse("007)"));
    }

    #[test]
    fn python_value_parses_radix_and_grouped_integers() {
        expect!["31 | rest \")\""].assert_eq(&parse("0x1f)"));
        expect!["255 | rest \")\""].assert_eq(&parse("0XFF)"));
        expect!["15 | rest \")\""].assert_eq(&parse("0o17)"));
        expect!["5 | rest \")\""].assert_eq(&parse("0b101)"));
        expect!["-31 | rest \")\""].assert_eq(&parse("-0x1F)"));
        expect!["1000 | rest \")\""].assert_eq(&parse("1_000)"));
        expect!["1000.5 | rest \")\""].assert_eq(&parse("1_000.5)"));
        expect!["65535 | rest \")\""].assert_eq(&parse("0xFF_FF)"));
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
            "{'a', 'b'}",
            "{1: 'int key'}",
            "'bad \\ud83d\\u0041'",
            // A named escape would need a Unicode name table; passing it
            // through verbatim would silently change the decoded string.
            r"'\N{BULLET}'",
        ] {
            assert!(
                !matches!(error(text), ErrMode::Incomplete(_)),
                "{text:?} should fail"
            );
        }
    }
}
