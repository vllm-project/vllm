//! Shared helpers for tool parsers.

use winnow::error::{ErrMode, ModalResult, Needed};
use winnow::stream::{Offset, Partial, Stream};

use super::{Result, ToolParserError, parsing_failed};

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
    token
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(token.len()))
        .filter(|&len| len < token.len())
        .rev()
        .find(|&len| buffer.ends_with(&token[..len]))
        .unwrap_or(0)
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

/// Parse one event from a buffered streaming input.
///
/// Returns:
/// - `Ok(Some((event, consumed_len)))` if an event was successfully parsed, along with the number
///   of bytes consumed from the buffer.
/// - `Ok(None)` if the buffer does not contain a full event yet, and more data is needed.
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
    use winnow::error::ErrMode;
    use winnow::stream::{Offset, Partial, Stream};

    use super::{partial_prefix_len, safe_text_len};

    #[test]
    fn partial_prefix_len_handles_ascii_markers() {
        assert_eq!(
            partial_prefix_len("hello<|tool", "<|tool_call>"),
            "<|tool".len()
        );
        assert_eq!(partial_prefix_len("hello world", "<|tool_call>"), 0);
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
}
