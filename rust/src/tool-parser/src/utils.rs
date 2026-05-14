//! Shared helpers for tool parsers.

use std::borrow::Cow;

use winnow::error::{ContextError, ErrMode, ModalResult, Needed, StrContext, StrContextValue};
use winnow::stream::{Offset, Partial, Stream};

use super::Result;

/// Return the byte length of the longest proper prefix of `token` that is also
/// a suffix of `buffer`.
///
/// Streaming parsers use this to keep only the trailing fragment that might
/// still grow into a full marker after the next decoded chunk arrives.
///
/// The returned length is always a valid UTF-8 boundary in `token`, so callers
/// can safely slice `&token[..len]` even when markers contain non-ASCII
/// characters such as DeepSeek's DSML delimiters.
pub(super) fn partial_prefix_len(buffer: &str, token: &str) -> usize {
    let Some(first_byte) = token.as_bytes().first().copied() else {
        return 0;
    };

    let max_len = buffer.len().min(token.len().saturating_sub(1));
    let tail_start = buffer.len() - max_len;
    let buffer_bytes = buffer.as_bytes();
    let token_bytes = token.as_bytes();

    // Scan from the longest possible suffix to preserve overlapping prefixes.
    for index in tail_start..buffer.len() {
        if buffer_bytes[index] != first_byte {
            continue;
        }

        let len = buffer.len() - index;
        if buffer.is_char_boundary(index)
            && token.is_char_boundary(len)
            && token_bytes[..len] == buffer_bytes[index..]
        {
            return len;
        }
    }

    0
}

/// Parse a safe text run before the next marker.
///
/// Returns the text length in bytes, and advances the input.
pub(super) fn safe_text_len(input: &mut Partial<&str>, marker: &str) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    if let Some(start_idx) = text.find(marker) {
        input.next_slice(start_idx);
        return Ok(start_idx);
    }

    let keep_len = partial_prefix_len(text, marker);
    let emit_len = text.len().saturating_sub(keep_len);
    if emit_len == 0 {
        return incomplete();
    }

    input.next_slice(emit_len);
    Ok(emit_len)
}

/// Decode XML/HTML entities in XML-style parameter values.
pub(super) fn xml_unescape(value: &str) -> Cow<'_, str> {
    if !value.as_bytes().contains(&b'&') {
        return Cow::Borrowed(value);
    }

    let mut output: Option<String> = None;
    let mut copied_len = 0;
    let mut rest = value;

    while let Some(ampersand) = rest.find('&') {
        let before_ampersand = &rest[..ampersand];
        let after_ampersand = &rest[ampersand + '&'.len_utf8()..];
        if let Some(semicolon) = after_ampersand.find(';') {
            let entity = &after_ampersand[..semicolon];
            if let Some(decoded) = decode_xml_entity(entity) {
                match &mut output {
                    Some(output) => output.push_str(before_ampersand),
                    None => {
                        let mut new_output = String::with_capacity(value.len());
                        new_output.push_str(&value[..copied_len + ampersand]);
                        output = Some(new_output);
                    }
                }
                let output = output.as_mut().expect("output is initialized above");
                output.push(decoded);
                let consumed_len = ampersand + '&'.len_utf8() + semicolon + ';'.len_utf8();
                copied_len += consumed_len;
                rest = &rest[consumed_len..];
                continue;
            }
        }

        if let Some(output) = &mut output {
            output.push_str(before_ampersand);
            output.push('&');
        }
        let consumed_len = ampersand + '&'.len_utf8();
        copied_len += consumed_len;
        rest = after_ampersand;
    }

    if let Some(mut output) = output {
        output.push_str(rest);
        Cow::Owned(output)
    } else {
        Cow::Borrowed(value)
    }
}

fn decode_xml_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        entity if entity.starts_with("#x") || entity.starts_with("#X") => {
            u32::from_str_radix(&entity[2..], 16).ok().and_then(char::from_u32)
        }
        entity if entity.starts_with('#') => {
            entity[1..].parse::<u32>().ok().and_then(char::from_u32)
        }
        _ => None,
    }
}

/// Streaming lexical state for a top-level JSON object.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct JsonObjectScanState {
    object_depth: usize,
    array_depth: usize,
    in_string: bool,
    escape: bool,
    phase: JsonObjectScanPhase,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum JsonObjectScanPhase {
    #[default]
    Initial,
    Scanning,
    Complete,
}

impl JsonObjectScanState {
    /// Returns whether the top-level JSON object has closed.
    pub(super) const fn complete(&self) -> bool {
        matches!(self.phase, JsonObjectScanPhase::Complete)
    }
}

/// Parse a raw top-level JSON object argument prefix.
///
/// The returned length is safe to emit as raw argument text. This scans only
/// lexical boundaries from `{` through the matching `}`, preserving
/// malformed-but-balanced JSON without deserializing or normalizing it.
pub(super) fn take_json_object(
    input: &mut Partial<&str>,
    state: &mut JsonObjectScanState,
) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }
    if state.complete() {
        return Err(json_scan_error(
            "JSON object argument",
            StrContextValue::Description("active JSON object scan"),
        ));
    }

    let bytes = text.as_bytes();
    let just_started = matches!(state.phase, JsonObjectScanPhase::Initial);
    if just_started {
        if bytes[0] != b'{' {
            return Err(json_scan_error(
                "JSON object argument",
                StrContextValue::CharLiteral('{'),
            ));
        }
        state.phase = JsonObjectScanPhase::Scanning;
        state.object_depth = 1;
    }

    let mut index = usize::from(just_started);

    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;

        if state.in_string {
            if state.escape {
                state.escape = false;
            } else if byte == b'\\' {
                state.escape = true;
            } else if byte == b'"' {
                state.in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => state.in_string = true,
            b'{' => state.object_depth += 1,
            b'}' => {
                state.object_depth = state.object_depth.checked_sub(1).ok_or_else(|| {
                    json_scan_error(
                        "JSON object argument",
                        StrContextValue::Description("balanced object braces"),
                    )
                })?;
                if state.object_depth == 0 && state.array_depth == 0 {
                    state.phase = JsonObjectScanPhase::Complete;
                    input.next_slice(index);
                    return Ok(index);
                }
                if state.object_depth == 0 {
                    return Err(json_scan_error(
                        "JSON object argument",
                        StrContextValue::Description(
                            "nested arrays to close before the top-level object",
                        ),
                    ));
                }
            }
            b'[' => state.array_depth += 1,
            b']' => {
                state.array_depth = state.array_depth.checked_sub(1).ok_or_else(|| {
                    json_scan_error(
                        "JSON object argument",
                        StrContextValue::Description("balanced array brackets"),
                    )
                })?;
            }
            _ => {}
        }
    }

    input.next_slice(text.len());
    Ok(text.len())
}

/// Parse a JSON string literal.
pub(super) fn json_str(input: &mut Partial<&str>) -> ModalResult<String> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    let bytes = text.as_bytes();
    if bytes[0] != b'"' {
        return Err(json_scan_error(
            "JSON string",
            StrContextValue::CharLiteral('"'),
        ));
    }

    let mut escape = false;
    let mut index = 1;
    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;

        if escape {
            escape = false;
            continue;
        }

        match byte {
            b'\\' => escape = true,
            b'"' => {
                let raw = &text[..index];
                let value = serde_json::from_str::<String>(raw).map_err(|_| {
                    json_scan_error(
                        "JSON string",
                        StrContextValue::Description("valid JSON string"),
                    )
                })?;
                input.next_slice(index);
                return Ok(value);
            }
            _ => {}
        }
    }

    incomplete()
}

fn json_scan_error(label: &'static str, expected: StrContextValue) -> ErrMode<ContextError> {
    let mut error = ContextError::new();
    error.push(StrContext::Label(label));
    error.push(StrContext::Expected(expected));
    ErrMode::Cut(error)
}

/// Parse one event from a buffered streaming input.
///
/// Returns:
/// - `Ok(Some((event, consumed_len)))` if an event was successfully parsed,
///   along with the number of bytes consumed from the buffer.
/// - `Ok(None)` if the buffer does not contain a full event yet, and more data
///   is needed.
/// - `Err` if a parsing error occurred.
pub(super) fn parse_buffered_event<E>(
    buffer: &str,
    parse: impl FnOnce(&mut Partial<&str>) -> ModalResult<E>,
) -> Result<Option<(E, usize)>> {
    let mut input = Partial::new(buffer);
    let checkpoint = input.checkpoint();
    let event = match parse(&mut input) {
        Ok(event) => event,
        Err(ErrMode::Incomplete(_)) => return Ok(None),
        Err(ErrMode::Backtrack(e) | ErrMode::Cut(e)) => {
            // TODO: enrich context for error reporting
            return Err(parsing_failed!("{}", e));
        }
    };
    let consumed_len = input.offset_from(&checkpoint);
    if consumed_len == 0 {
        return Ok(None);
    }

    Ok(Some((event, consumed_len)))
}

/// Returns an error indicating that we need more data to continue parsing.
pub(super) fn incomplete<T>() -> ModalResult<T> {
    Err(ErrMode::Incomplete(Needed::Unknown))
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use expect_test::expect;
    use winnow::error::ErrMode;
    use winnow::stream::{Offset, Partial, Stream};

    use super::{
        JsonObjectScanState, json_str, partial_prefix_len, safe_text_len, take_json_object,
        xml_unescape,
    };

    #[test]
    fn partial_prefix_len_handles_ascii_markers() {
        assert_eq!(
            partial_prefix_len("hello<|tool", "<|tool_call>"),
            "<|tool".len()
        );
        assert_eq!(partial_prefix_len("hello world", "<|tool_call>"), 0);
    }

    #[test]
    fn partial_prefix_len_prefers_longest_overlapping_prefix() {
        assert_eq!(partial_prefix_len("chunk ending in aba", "ababa"), 3);
    }

    #[test]
    fn partial_prefix_len_handles_unicode_markers() {
        let token = "<｜DSML｜function_calls>";
        assert_eq!(
            partial_prefix_len("prefix <｜DSML｜fun", token),
            "<｜DSML｜fun".len()
        );
        assert_eq!(partial_prefix_len("prefix <｜DSML", token), "<｜DSML".len());
    }

    #[test]
    fn safe_text_len_stops_before_marker() {
        let mut input = Partial::new("hello<tool_call>");
        let checkpoint = input.checkpoint();

        let len = safe_text_len(&mut input, "<tool_call>").unwrap();

        assert_eq!(len, "hello".len());
        assert_eq!(input.offset_from(&checkpoint), "hello".len());
    }

    #[test]
    fn safe_text_len_holds_back_partial_marker() {
        let mut input = Partial::new("hello<tool");
        let checkpoint = input.checkpoint();

        let len = safe_text_len(&mut input, "<tool_call>").unwrap();

        assert_eq!(len, "hello".len());
        assert_eq!(input.offset_from(&checkpoint), "hello".len());
    }

    #[test]
    fn safe_text_len_reports_incomplete_for_only_partial_marker() {
        let mut input = Partial::new("<tool");

        let error = safe_text_len(&mut input, "<tool_call>").unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
    }

    #[test]
    fn xml_unescape_decodes_common_entities() {
        assert_eq!(
            xml_unescape("&lt;tag attr=&quot;value&quot;&gt;Tom &amp; Jerry&apos;s&lt;/tag&gt;"),
            r#"<tag attr="value">Tom & Jerry's</tag>"#
        );
    }

    #[test]
    fn xml_unescape_decodes_numeric_entities() {
        assert_eq!(xml_unescape("&#60;tag&#x3E;&#x1F600;"), "<tag>😀");
    }

    #[test]
    fn xml_unescape_preserves_unknown_and_incomplete_entities() {
        let output = xml_unescape("Tom & Jerry &unknown; &amp");

        assert!(matches!(output, Cow::Borrowed(_)));
        assert_eq!(output, "Tom & Jerry &unknown; &amp");
    }

    #[test]
    fn xml_unescape_borrows_when_no_entity_is_present() {
        let input = "plain text";
        let output = xml_unescape(input);

        assert!(matches!(output, Cow::Borrowed(_)));
        assert_eq!(output, input);
    }

    #[test]
    fn take_json_object_consumes_simple_object() {
        let mut state = JsonObjectScanState::default();
        let buffer = r#"{"location":"Paris"}<end>"#;
        let mut input = Partial::new(buffer);
        let checkpoint = input.checkpoint();

        let len = take_json_object(&mut input, &mut state).unwrap();

        assert_eq!(len, r#"{"location":"Paris"}"#.len());
        assert_eq!(input.offset_from(&checkpoint), len);
        assert!(state.complete());
    }

    #[test]
    fn take_json_object_tracks_nested_values_and_strings() {
        let mut state = JsonObjectScanState::default();
        let arguments = r#"{"nested":{"items":[{"text":"} <|tool_call_end|> \" \\"}]}}"#;
        let buffer = format!("{arguments}<end>");
        let mut input = Partial::new(buffer.as_str());

        let len = take_json_object(&mut input, &mut state).unwrap();

        assert_eq!(len, arguments.len());
        assert!(state.complete());
    }

    #[test]
    fn take_json_object_rejects_leading_whitespace() {
        let mut state = JsonObjectScanState::default();
        let mut input = Partial::new(" {\"x\":1}");

        let error = take_json_object(&mut input, &mut state).unwrap_err();

        let ErrMode::Cut(error) = error else {
            panic!("expected cut error");
        };
        expect![[r#"
            invalid JSON object argument
            expected `{`"#]]
        .assert_eq(&error.to_string());
    }

    #[test]
    fn take_json_object_leaves_trailing_whitespace_to_caller() {
        let mut state = JsonObjectScanState::default();
        let mut input = Partial::new("{\"x\":1}\n<end>");
        let checkpoint = input.checkpoint();

        let len = take_json_object(&mut input, &mut state).unwrap();

        assert_eq!(len, "{\"x\":1}".len());
        assert_eq!(input.offset_from(&checkpoint), len);
        assert!(state.complete());
    }

    #[test]
    fn take_json_object_continues_across_chunks() {
        let mut state = JsonObjectScanState::default();
        let chunks = [
            r#"{"text":"literal "#,
            r#"<|tool_call_end|>"#,
            r#" inside"}<end>"#,
        ];
        let mut collected = String::new();

        for chunk in chunks {
            let mut input = Partial::new(chunk);
            let len = take_json_object(&mut input, &mut state).unwrap();
            collected.push_str(&chunk[..len]);
        }

        assert_eq!(collected, r#"{"text":"literal <|tool_call_end|> inside"}"#);
        assert!(state.complete());
    }

    #[test]
    fn take_json_object_rejects_non_object_top_level() {
        let mut state = JsonObjectScanState::default();
        let mut input = Partial::new(r#"[{"x":1}]"#);

        let error = take_json_object(&mut input, &mut state).unwrap_err();

        let ErrMode::Cut(error) = error else {
            panic!("expected cut error");
        };
        expect![[r#"
            invalid JSON object argument
            expected `{`"#]]
        .assert_eq(&error.to_string());
    }

    #[test]
    fn take_json_object_reports_unbalanced_array() {
        let mut state = JsonObjectScanState::default();
        let mut input = Partial::new(r#"{"x":]}"#);

        let error = take_json_object(&mut input, &mut state).unwrap_err();

        let ErrMode::Cut(error) = error else {
            panic!("expected cut error");
        };
        expect![[r#"
            invalid JSON object argument
            expected balanced array brackets"#]]
        .assert_eq(&error.to_string());
    }

    #[test]
    fn take_json_object_reports_top_level_close_before_nested_array() {
        let mut state = JsonObjectScanState::default();
        let mut input = Partial::new(r#"{"x":[}"#);

        let error = take_json_object(&mut input, &mut state).unwrap_err();

        let ErrMode::Cut(error) = error else {
            panic!("expected cut error");
        };
        expect![[r#"
            invalid JSON object argument
            expected nested arrays to close before the top-level object"#]]
        .assert_eq(&error.to_string());
    }

    #[test]
    fn json_str_decodes_escaped_content() {
        let mut input = Partial::new(r#""say_\"hi\u0021" rest"#);

        let value = json_str(&mut input).unwrap();

        assert_eq!(value, "say_\"hi!");
        assert_eq!(*input, " rest");
    }

    #[test]
    fn json_str_reports_incomplete_escaped_string() {
        let mut input = Partial::new(r#""say_\"#);

        let error = json_str(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
    }
}
