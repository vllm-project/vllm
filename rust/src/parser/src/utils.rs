//! Shared helpers for streaming parsers.

use winnow::Parser;
use winnow::error::{ContextError, ErrMode, ModalResult, Needed, StrContext, StrContextValue};
use winnow::stream::{FindSlice, Offset, Partial, Stream};

use crate::tool::{Result, ToolParserError};

/// Return the byte length of the longest proper prefix of `token` that is also
/// a suffix of `buffer`.
///
/// Streaming parsers use this to keep only the trailing fragment that might
/// still grow into a full marker after the next decoded chunk arrives.
///
/// The returned length is always a valid UTF-8 boundary in `token`, so callers
/// can safely slice `&token[..len]` even when markers contain non-ASCII
/// characters such as DeepSeek's DSML delimiters.
pub fn partial_prefix_len(buffer: &str, token: &str) -> usize {
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
/// This is the single-marker variant of [`safe_text_len_mul`].
///
/// Returns the text length in bytes, and advances the input.
pub fn safe_text_len(input: &mut Partial<&str>, marker: &str) -> ModalResult<usize> {
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

/// Parse a safe text run before the earliest next marker.
/// This is the multi-marker variant of [`safe_text_len`].
///
/// Returns the text length in bytes, and advances the input.
pub fn safe_text_len_mul(input: &mut Partial<&str>, markers: &[&str]) -> ModalResult<usize> {
    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    if let Some(start_idx) = find_slice_mul(text, markers) {
        input.next_slice(start_idx);
        return Ok(start_idx);
    }

    let keep_len = markers.iter().map(|marker| partial_prefix_len(text, marker)).max().unwrap_or(0);
    let emit_len = text.len().saturating_sub(keep_len);
    if emit_len == 0 {
        return incomplete();
    }

    input.next_slice(emit_len);
    Ok(emit_len)
}

#[inline(always)]
fn find_slice_mul(text: &str, markers: &[&str]) -> Option<usize> {
    let range = match markers {
        // Use the fast specialized `winnow::stream::FindSlice` impl for 1-3 markers.
        [first] => text.find_slice(*first),
        [first, second] => text.find_slice((*first, *second)),
        [first, second, third] => text.find_slice((*first, *second, *third)),
        // Fall back to a linear scan for 4+ markers.
        _ => return markers.iter().filter_map(|marker| text.find(marker)).min(),
    };
    range.map(|range| range.start)
}

/// Streaming scan state for a buffered marker search [`take_until_marker`],
/// so that we don't have to rescan the whole buffered prefix when resuming.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MarkerScanState {
    scan_start: usize,
}

impl MarkerScanState {
    pub fn reset(&mut self) {
        self.scan_start = 0;
    }
}

/// Parse text until `marker`, resuming from the last safe scan checkpoint.
///
/// This is the streaming-buffered variant of `winnow::token::take_until(0..,
/// marker)`: it returns the slice before `marker` and leaves `marker` for the
/// caller to consume. On incomplete input, it stores the earliest byte offset
/// that can still match `marker` and returns `Incomplete` without consuming
/// input, so the next parse can avoid rescanning the whole buffered prefix.
///
/// Use this for outer parser states that keep the full buffered input across
/// chunks while waiting for a closing marker. Plain `take_until` is still a
/// better fit for one-shot parsers over a complete body, and for `1..` cases
/// where an empty slice before the marker should be rejected.
pub fn take_until_marker<'i, 'a>(
    marker: &'a str,
    state: &'a mut MarkerScanState,
) -> impl Parser<Partial<&'i str>, &'i str, ErrMode<ContextError>> + 'a {
    move |input: &mut Partial<&'i str>| take_until_marker_(input, marker, state)
}

fn take_until_marker_<'i>(
    input: &mut Partial<&'i str>,
    marker: &str,
    state: &mut MarkerScanState,
) -> ModalResult<&'i str> {
    debug_assert!(!marker.is_empty());

    let text = **input;
    if text.is_empty() {
        return incomplete();
    }

    // Normal updates store a char boundary; this keeps stale or misused state from panicking.
    let scan_start = floor_char_boundary(text, state.scan_start);

    if let Some(offset) = text[scan_start..].find(marker) {
        let marker_start = scan_start + offset;
        let body = &text[..marker_start];
        input.next_slice(marker_start);
        state.reset();
        return Ok(body);
    }

    let keep_len = partial_prefix_len(text, marker);
    state.scan_start = text.len() - keep_len;
    incomplete()
}

fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Streaming lexical state for a top-level JSON object.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JsonObjectScanState {
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
    pub const fn complete(&self) -> bool {
        matches!(self.phase, JsonObjectScanPhase::Complete)
    }
}

/// Parse a raw top-level JSON object argument prefix.
///
/// The returned length is safe to emit as raw argument text. This scans only
/// lexical boundaries from `{` through the matching `}`, preserving
/// malformed-but-balanced JSON without deserializing or normalizing it.
pub fn take_json_object(
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

/// Streaming lexical state for a JSON string literal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JsonStringScanState {
    scanned_len: usize,
    escape: bool,
}

/// Parse a raw JSON string literal, resuming from the last scanned byte.
///
/// The returned length covers the quoted JSON string. This only scans for the
/// string boundary; callers that need the decoded value should pass the raw
/// slice to [`decode_json_str`].
pub fn take_json_string(
    input: &mut Partial<&str>,
    state: &mut JsonStringScanState,
) -> ModalResult<usize> {
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

    let mut index = if state.scanned_len == 0 {
        1
    } else if state.scanned_len <= bytes.len() {
        state.scanned_len
    } else {
        return incomplete();
    };

    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;

        if state.escape {
            state.escape = false;
            continue;
        }

        match byte {
            b'\\' => state.escape = true,
            b'"' => {
                input.next_slice(index);
                return Ok(index);
            }
            _ => {}
        }
    }

    state.scanned_len = text.len();
    incomplete()
}

/// Parse a JSON string literal.
pub fn json_str(input: &mut Partial<&str>) -> ModalResult<String> {
    let text = **input;
    let checkpoint = input.checkpoint();
    let mut state = JsonStringScanState::default();
    let len = take_json_string(input, &mut state)?;
    decode_json_str(&text[..len]).inspect_err(|_| {
        input.reset(&checkpoint);
    })
}

/// Decode a complete JSON string literal.
pub fn decode_json_str(raw: &str) -> ModalResult<String> {
    serde_json::from_str::<String>(raw).map_err(|_| {
        json_scan_error(
            "JSON string",
            StrContextValue::Description("valid JSON string"),
        )
    })
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
/// - `Ok(Some((event, consumed_len)))` if an event was successfully parsed, along with the number
///   of bytes consumed from the buffer.
/// - `Ok(None)` if the buffer does not contain a full event yet, and more data is needed.
/// - `Err` if a parsing error occurred.
pub fn parse_buffered_event<E>(
    buffer: &str,
    parse: impl FnOnce(&mut Partial<&str>) -> ModalResult<E>,
) -> Result<Option<(E, usize)>> {
    let mut input = Partial::new(buffer);
    let checkpoint = input.checkpoint();
    let event = match parse(&mut input) {
        Ok(event) => event,
        Err(ErrMode::Incomplete(_)) => return Ok(None),
        Err(ErrMode::Backtrack(e) | ErrMode::Cut(e)) => {
            let snippet = buffer.char_indices().nth(80).map_or(buffer, |(i, _)| &buffer[..i]);
            return Err(ToolParserError::ParsingFailed {
                message: format!("near {snippet:?}: {e}"),
            });
        }
    };
    let consumed_len = input.offset_from(&checkpoint);
    if consumed_len == 0 {
        return Ok(None);
    }

    Ok(Some((event, consumed_len)))
}

/// Returns an error indicating that we need more data to continue parsing.
pub fn incomplete<T>() -> ModalResult<T> {
    Err(ErrMode::Incomplete(Needed::Unknown))
}

#[cfg(test)]
mod tests {

    use expect_test::expect;
    use winnow::Parser;
    use winnow::error::ErrMode;
    use winnow::stream::{Offset, Partial, Stream};

    use super::{
        JsonObjectScanState, JsonStringScanState, MarkerScanState, json_str, parse_buffered_event,
        partial_prefix_len, safe_text_len, safe_text_len_mul, take_json_object, take_json_string,
        take_until_marker,
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
    fn safe_text_len_mul_stops_before_earliest_marker() {
        let mut input = Partial::new("hello<channel|><|tool_call>");
        let checkpoint = input.checkpoint();

        let len = safe_text_len_mul(&mut input, &["<|tool_call>", "<channel|>"]).unwrap();

        assert_eq!(len, "hello".len());
        assert_eq!(input.offset_from(&checkpoint), "hello".len());
        assert_eq!(*input, "<channel|><|tool_call>");
    }

    #[test]
    fn safe_text_len_mul_holds_back_longest_partial_marker() {
        let mut input = Partial::new("hello<|tool");
        let checkpoint = input.checkpoint();

        let len = safe_text_len_mul(&mut input, &["<|tool_call>", "<|channel>thought\n"]).unwrap();

        assert_eq!(len, "hello".len());
        assert_eq!(input.offset_from(&checkpoint), "hello".len());
        assert_eq!(*input, "<|tool");
    }

    #[test]
    fn safe_text_len_mul_skips_false_same_prefix_candidate() {
        let mut input = Partial::new("hello<not_marker><|tool_call>");
        let checkpoint = input.checkpoint();

        let len = safe_text_len_mul(&mut input, &["<|tool_call>", "<|channel>thought\n"]).unwrap();

        assert_eq!(len, "hello<not_marker>".len());
        assert_eq!(input.offset_from(&checkpoint), "hello<not_marker>".len());
        assert_eq!(*input, "<|tool_call>");
    }

    #[test]
    fn safe_text_len_mul_reports_incomplete_for_only_partial_marker() {
        let mut input = Partial::new("<|channel>thought");

        let error =
            safe_text_len_mul(&mut input, &["<|tool_call>", "<|channel>thought\n"]).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
    }

    #[test]
    fn take_until_marker_stops_before_marker() {
        let mut state = MarkerScanState::default();
        let mut input = Partial::new("body</end>tail");
        let checkpoint = input.checkpoint();

        let body = take_until_marker("</end>", &mut state).parse_next(&mut input).unwrap();

        assert_eq!(body, "body");
        assert_eq!(input.offset_from(&checkpoint), "body".len());
        assert_eq!(*input, "</end>tail");
        assert_eq!(state, MarkerScanState::default());
    }

    #[test]
    fn take_until_marker_resumes_after_split_marker() {
        let mut state = MarkerScanState::default();
        let mut input = Partial::new("body</to");

        let error = take_until_marker("</tool_call>", &mut state)
            .parse_next(&mut input)
            .unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
        assert_eq!(state.scan_start, "body".len());

        let mut input = Partial::new("body</tool_call>tail");
        let body = take_until_marker("</tool_call>", &mut state).parse_next(&mut input).unwrap();

        assert_eq!(body, "body");
        assert_eq!(*input, "</tool_call>tail");
        assert_eq!(state, MarkerScanState::default());
    }

    #[test]
    fn take_until_marker_advances_checkpoint_for_long_prefix() {
        let mut state = MarkerScanState::default();
        let text = format!("{}{}", "x".repeat(1024), "</too");
        let mut input = Partial::new(text.as_str());

        let error = take_until_marker("</tool_call>", &mut state)
            .parse_next(&mut input)
            .unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
        assert_eq!(state.scan_start, 1024);
    }

    #[test]
    fn take_until_marker_keeps_unicode_marker_boundaries() {
        let marker = "<｜DSML｜function_calls>";
        let mut state = MarkerScanState::default();
        let mut input = Partial::new("prefix <｜DSML｜fun");

        let error = take_until_marker(marker, &mut state).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
        assert_eq!(state.scan_start, "prefix ".len());
        assert!("prefix <｜DSML｜fun".is_char_boundary(state.scan_start));

        let mut input = Partial::new("prefix <｜DSML｜function_calls>tail");
        let body = take_until_marker(marker, &mut state).parse_next(&mut input).unwrap();

        assert_eq!(body, "prefix ");
        assert_eq!(*input, "<｜DSML｜function_calls>tail");
        assert_eq!(state, MarkerScanState::default());
    }

    #[test]
    fn take_until_marker_floors_stale_checkpoint_to_char_boundary() {
        let mut state = MarkerScanState { scan_start: 1 };
        let mut input = Partial::new("é</end>");

        let body = take_until_marker("</end>", &mut state).parse_next(&mut input).unwrap();

        assert_eq!(body, "é");
        assert_eq!(*input, "</end>");
        assert_eq!(state, MarkerScanState::default());
    }

    #[test]
    fn take_until_marker_handles_overlapping_prefixes() {
        let mut state = MarkerScanState::default();
        let mut input = Partial::new("xxaba");

        let error = take_until_marker("ababa", &mut state).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
        assert_eq!(state.scan_start, 2);

        let mut input = Partial::new("xxababa!");
        let body = take_until_marker("ababa", &mut state).parse_next(&mut input).unwrap();

        assert_eq!(body, "xx");
        assert_eq!(*input, "ababa!");
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
    fn take_json_string_consumes_complete_string() {
        let mut state = JsonStringScanState::default();
        let mut input = Partial::new(r#""say_\"hi\u0021" rest"#);
        let checkpoint = input.checkpoint();

        let len = take_json_string(&mut input, &mut state).unwrap();

        assert_eq!(len, r#""say_\"hi\u0021""#.len());
        assert_eq!(input.offset_from(&checkpoint), len);
        assert_eq!(*input, " rest");
    }

    #[test]
    fn take_json_string_resumes_after_incomplete_input() {
        let mut state = JsonStringScanState::default();
        let mut input = Partial::new(r#""{\"data\":\"partial"#);
        let checkpoint = input.checkpoint();

        let error = take_json_string(&mut input, &mut state).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
        assert_eq!(input.offset_from(&checkpoint), 0);
        assert_eq!(state.scanned_len, r#""{\"data\":\"partial"#.len());

        let mut input = Partial::new(r#""{\"data\":\"partial string\"}" tail"#);
        let len = take_json_string(&mut input, &mut state).unwrap();

        assert_eq!(len, r#""{\"data\":\"partial string\"}""#.len());
        assert_eq!(*input, " tail");
    }

    #[test]
    fn take_json_string_tracks_escape_across_chunks() {
        let mut state = JsonStringScanState::default();
        let mut input = Partial::new(r#""abc\"#);

        let error = take_json_string(&mut input, &mut state).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
        assert!(state.escape);

        let mut input = Partial::new(r#""abc\"def" tail"#);
        let len = take_json_string(&mut input, &mut state).unwrap();

        assert_eq!(len, r#""abc\"def""#.len());
        assert_eq!(*input, " tail");
    }

    #[test]
    fn take_json_string_rejects_non_string_start() {
        let mut state = JsonStringScanState::default();
        let mut input = Partial::new("42");

        let error = take_json_string(&mut input, &mut state).unwrap_err();

        let ErrMode::Cut(error) = error else {
            panic!("expected cut error");
        };
        expect![[r#"
            invalid JSON string
            expected `"`"#]]
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

    #[test]
    fn parse_buffered_event_error_includes_input_snippet() {
        let result = parse_buffered_event(" {\"x\":1}", |input| {
            take_json_object(input, &mut JsonObjectScanState::default())
        });
        let err = result.unwrap_err().to_string();
        assert!(err.contains("near \""), "error must include snippet");
    }

    #[test]
    fn parse_buffered_event_error_truncates_long_input() {
        let long_input = format!(" {}", "x".repeat(100));
        let result = parse_buffered_event(&long_input, |input| {
            take_json_object(input, &mut JsonObjectScanState::default())
        });
        let err = result.unwrap_err().to_string();
        assert!(err.contains("near \""), "error must include snippet");
        assert!(
            !err.contains(&long_input),
            "snippet must be truncated for long input"
        );
    }
}
