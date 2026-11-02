// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Token-aware marker matching for streaming parsers.
//!
//! Model protocols delimit channels with dedicated special tokens such as
//! `<|close|>` and `<|sep|>`. Ordinary BPE tokens can decode to the very same
//! spelling, so matching a marker by text alone mistakes model-written content
//! for structure. The helpers here keep text as the parse axis and consult the
//! generated-token attribution carried by [`DecodedText`] only where a marker's
//! bytes must have come from a special token.
//!
//! - [`Attributed`] is a winnow input over one `DecodedText` buffer: the text
//!   plus its buffer-local token anchors, composed from winnow's own stream
//!   wrappers so every existing combinator keeps working.
//! - [`Marker`] is a fixed marker string whose special-token segments are
//!   *guarded*: they match only when the expected special token produced them.
//!   A marker without guards matches by spelling alone, which is exactly what a
//!   plain `&str` marker is; [`MarkerRef`] is the borrowed view shared by both.
//!   One marker definition is both the parser that consumes the marker and the
//!   definition the safe-text scanner in the parent module stops in front of,
//!   so the two never disagree.
//! - [`MarkerStream`] abstracts over `Partial<&str>` (text-only parsers) and
//!   [`Attributed`], letting one generic scanner serve both.

use std::fmt::Debug;
use std::ops::Range;

use auto_impl::auto_impl;
use educe::Educe;
use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};
use winnow::Parser;
use winnow::error::{ContextError, ErrMode, ModalResult};
use winnow::stream::{
    Compare, FindSlice, LocatingSlice, Location, Partial, Stateful, Stream, StreamIsPartial,
};
use winnow::token::literal;

use super::partial_prefix_len;

/// Streaming winnow input over one [`DecodedText`] buffer.
///
/// `LocatingSlice` reports the cursor's byte offset from the buffer start, which
/// is the coordinate system of the anchors; `Partial` keeps the incomplete-input
/// semantics streaming parsers rely on; `Stateful` carries the immutable
/// [`AttributedState`]. winnow forwards `Compare`, `FindSlice`, `Location`, and
/// `UpdateSlice` through all three, so `literal`, `alt`, `seq!`, `take_until`,
/// and `rest` work unchanged.
pub type Attributed<'i, M = ()> = Stateful<Partial<LocatingSlice<&'i str>>, AttributedState<'i, M>>;

/// Immutable state of an [`Attributed`] input.
#[derive(Debug, Educe)]
#[educe(Clone, Copy)]
pub struct AttributedState<'i, M = ()> {
    /// Token anchors of the buffer, sorted by byte offset.
    anchors: &'i [TokenAttribution],
    /// The parser's markers; see [`AttributedExt::markers`].
    markers: &'i M,
}

#[easy_ext::ext(AttributedExt)]
pub impl<'i, M> Attributed<'i, M> {
    /// The parser's markers, reachable from every grammar function through the
    /// input instead of an extra parameter.
    fn markers(&self) -> &'i M {
        self.state.markers
    }
}

/// Create an [`Attributed`] input over `buffer`, carrying `markers`.
pub fn attributed_with_markers<'i, M>(
    buffer: &'i DecodedText,
    markers: &'i M,
) -> Attributed<'i, M> {
    Stateful {
        input: Partial::new(LocatingSlice::new(buffer.text.as_str())),
        state: AttributedState {
            anchors: &buffer.attributions,
            markers,
        },
    }
}

/// A `&str`-sliced streaming input that can say where markers may begin.
///
/// Implemented for `Partial<&str>` (text-only parsers, no anchors) and
/// [`Attributed`].
pub trait MarkerStream<'i>:
    Stream<Slice = &'i str> + StreamIsPartial + Clone + for<'m> Compare<&'m str>
{
    /// Remaining input from the cursor.
    fn remaining(&self) -> &'i str;

    /// Byte offset of the cursor in the coordinate system of
    /// [`Self::anchors`] and of resumable scan state.
    ///
    /// `Partial<&str>` tracks no origin and reports 0, so its offsets are
    /// relative to the cursor at the time of the call; [`Attributed`] reports
    /// the offset from the buffer start.
    fn offset(&self) -> usize;

    /// Token anchors of the buffer, sorted by byte offset; empty for text-only
    /// input.
    fn anchors(&self) -> &'i [TokenAttribution];

    /// Earliest offset at or after `from` where one of `markers` may begin.
    ///
    /// A guarded marker can only begin at a visible anchor of its first special
    /// token, so text-only input never yields one. Any other marker begins at a
    /// complete occurrence of its spelling, or at a proper prefix of it that
    /// ends the input.
    fn next_candidate<M: MarkerLike>(&self, markers: &[M], from: usize) -> Option<usize> {
        let start = self.offset();
        let from = from.max(start);
        let markers = || markers.iter().map(MarkerLike::as_marker);

        let anchored = markers().any(|marker| marker.first_token_id().is_some()).then(|| {
            visible_in(self.anchors(), from..usize::MAX).find_map(|(at, token_id)| {
                markers().any(|marker| marker.first_token_id() == Some(token_id)).then_some(at)
            })
        });
        let by_text = text_candidate(
            self.remaining(),
            markers()
                .filter(|marker| marker.first_token_id().is_none())
                .map(MarkerRef::as_str),
            from - start,
        )
        .map(|at| start + at);

        match (anchored.flatten(), by_text) {
            (Some(anchored), Some(by_text)) => Some(anchored.min(by_text)),
            (anchored, by_text) => anchored.or(by_text),
        }
    }
}

impl<'i> MarkerStream<'i> for Partial<&'i str> {
    fn remaining(&self) -> &'i str {
        **self
    }

    fn offset(&self) -> usize {
        0
    }

    fn anchors(&self) -> &'i [TokenAttribution] {
        &[]
    }
}

impl<'i, M: Debug> MarkerStream<'i> for Attributed<'i, M> {
    fn remaining(&self) -> &'i str {
        ****self
    }

    fn offset(&self) -> usize {
        self.current_token_start()
    }

    fn anchors(&self) -> &'i [TokenAttribution] {
        self.state.anchors
    }
}

fn anchor_offset(anchor: TokenAnchor) -> usize {
    match anchor {
        TokenAnchor::Visible { byte_offset } | TokenAnchor::ZeroWidth { byte_offset } => {
            byte_offset as usize
        }
    }
}

/// Visible anchors within `range`, as `(byte_offset, token_id)`, in generation
/// order.
fn visible_in(
    anchors: &[TokenAttribution],
    range: Range<usize>,
) -> impl Iterator<Item = (usize, u32)> + '_ {
    let first =
        anchors.partition_point(|attribution| anchor_offset(attribution.anchor) < range.start);
    anchors[first..]
        .iter()
        .take_while(move |attribution| anchor_offset(attribution.anchor) < range.end)
        .filter_map(|attribution| match attribution.anchor {
            TokenAnchor::Visible { byte_offset } => {
                Some((byte_offset as usize, attribution.token_id))
            }
            TokenAnchor::ZeroWidth { .. } => None,
        })
}

/// Return whether `range` is exactly the visible span of the single token
/// `token_id`: one visible anchor at `range.start` with that ID and no other
/// visible anchor inside. Zero-width records are ignored.
///
/// Special tokens never merge with neighbouring text, so this is the identity
/// test for a special token's spelling. Anchors beyond the available input do
/// not exist yet, so a partially arrived genuine spelling passes and is left to
/// `literal` to report as incomplete.
fn is_single_token(anchors: &[TokenAttribution], range: Range<usize>, token_id: u32) -> bool {
    let mut visible = visible_in(anchors, range.clone());
    visible.next() == Some((range.start, token_id)) && visible.next().is_none()
}

/// Earliest text position at or after `from` where one of `spellings` occurs
/// in full, or where a proper prefix of one of them ends the input.
fn text_candidate<'m>(
    text: &str,
    spellings: impl Iterator<Item = &'m str> + Clone,
    from: usize,
) -> Option<usize> {
    // Round up: a marker starts on a char boundary at or after `from`, and a
    // rejected candidate inside a multi-byte char must not be found again.
    let from = text.ceil_char_boundary(from);
    let tail = &text[from..];
    if tail.is_empty() {
        return None;
    }
    if let Some(at) = find_earliest(tail, spellings.clone()) {
        return Some(from + at);
    }
    let keep_len = spellings.map(|spelling| partial_prefix_len(tail, spelling)).max().unwrap_or(0);
    (keep_len > 0).then(|| text.len() - keep_len)
}

/// Earliest occurrence of any spelling in `text`.
#[inline(always)]
fn find_earliest<'m>(text: &str, mut spellings: impl Iterator<Item = &'m str>) -> Option<usize> {
    let first = spellings.next()?;
    let second = spellings.next();
    let third = second.and_then(|_| spellings.next());
    let fourth = third.and_then(|_| spellings.next());
    // Use the fast specialized `FindSlice` impls for 1-3 spellings, and fall
    // back to a linear scan for 4+.
    let range = match (second, third, fourth) {
        (None, _, _) => text.find_slice(first),
        (Some(second), None, _) => text.find_slice((first, second)),
        (Some(second), Some(third), None) => text.find_slice((first, second, third)),
        (Some(second), Some(third), Some(fourth)) => {
            return [first, second, third, fourth]
                .into_iter()
                .chain(spellings)
                .filter_map(|spelling| text.find(spelling))
                .min();
        }
    };
    range.map(|range| range.start)
}

/// A special token's spelling and its ID in the active tokenizer, resolved
/// together so a marker cannot pair one token's spelling with another's ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecialToken {
    pub text: String,
    pub id: u32,
}

/// A fixed marker string whose special-token segments are guarded by token
/// identity.
///
/// Build it once from tokenizer IDs, then use it both as the parser that
/// consumes the marker (it implements [`Parser`] by reference) and as the
/// definition the safe-text scanner stops in front of.
///
/// ```ignore
/// let think_close = Marker::special(&close).then_text("think").then_special(&sep);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Marker {
    text: String,
    guards: Vec<Guard>,
}

/// One special-token segment of a marker.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Guard {
    /// Byte range of the spelling within the marker text.
    range: Range<usize>,
    token_id: u32,
}

impl Marker {
    /// A marker matched by spelling alone.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            guards: Vec::new(),
        }
    }

    /// A marker beginning with the special token `token`.
    pub fn special(token: &SpecialToken) -> Self {
        Self::text(String::new()).then_special(token)
    }

    /// Append ordinary text, matched by spelling.
    #[must_use]
    pub fn then_text(mut self, text: &str) -> Self {
        self.text.push_str(text);
        self
    }

    /// Append the special token `token`.
    #[must_use]
    pub fn then_special(mut self, token: &SpecialToken) -> Self {
        let start = self.text.len();
        self.text.push_str(&token.text);
        self.guards.push(Guard {
            range: start..self.text.len(),
            token_id: token.id,
        });
        self
    }

    /// Drop the token-identity guards, so the marker matches by spelling alone,
    /// for callers whose input carries no token attribution.
    #[must_use]
    pub fn without_guards(mut self) -> Self {
        self.guards.clear();
        self
    }

    /// The full spelling of the marker.
    pub fn as_str(&self) -> &str {
        &self.text
    }
}

/// Anything the scanning helpers accept as a marker, viewed through
/// [`MarkerRef`].
///
/// A plain `str` is a marker without guards, so text-only parsers keep passing
/// `&[&str]` (or `&String`); token-aware parsers pass `&Marker`.
#[auto_impl(&)]
pub trait MarkerLike {
    /// Borrow the marker's spelling and guards.
    fn as_marker(&self) -> MarkerRef<'_>;
}

impl MarkerLike for str {
    fn as_marker(&self) -> MarkerRef<'_> {
        MarkerRef {
            text: self,
            guards: &[],
        }
    }
}

impl MarkerLike for String {
    fn as_marker(&self) -> MarkerRef<'_> {
        self.as_str().as_marker()
    }
}

impl MarkerLike for Marker {
    fn as_marker(&self) -> MarkerRef<'_> {
        MarkerRef {
            text: &self.text,
            guards: &self.guards,
        }
    }
}

/// Borrowed view of a marker: its spelling and guarded segments.
///
/// A plain `&str` is a marker without guards; see [`MarkerLike`].
#[derive(Clone, Copy, Debug)]
pub struct MarkerRef<'m> {
    text: &'m str,
    guards: &'m [Guard],
}

impl<'m> MarkerRef<'m> {
    /// The full spelling of the marker.
    pub fn as_str(self) -> &'m str {
        self.text
    }

    /// The special token the marker begins with, if it begins with one.
    pub fn first_token_id(self) -> Option<u32> {
        self.guards
            .first()
            .filter(|guard| guard.range.start == 0)
            .map(|guard| guard.token_id)
    }

    /// Parse the marker at the cursor, returning its spelling.
    ///
    /// Every guard whose first byte is available is checked before the spelling:
    /// when those bytes do not come from the expected special token the parser
    /// backtracks at once instead of asking for more input, so an ordinary
    /// partial lookalike streams as content immediately. A genuine special token
    /// split by output holdback still reports incomplete, because its anchor
    /// arrived with its first byte. Nothing is consumed on failure.
    pub fn parse_next<'i, I: MarkerStream<'i>>(self, input: &mut I) -> ModalResult<&'i str> {
        let start = input.offset();
        let available = input.eof_offset();
        let anchors = input.anchors();
        for guard in self.guards.iter().take_while(|guard| guard.range.start < available) {
            let range = start + guard.range.start..start + guard.range.end;
            if !is_single_token(anchors, range, guard.token_id) {
                return Err(ErrMode::Backtrack(ContextError::new()));
            }
        }
        literal(self.text).parse_next(input)
    }
}

impl<'i, I: MarkerStream<'i>> Parser<I, &'i str, ErrMode<ContextError>> for &Marker {
    fn parse_next(&mut self, input: &mut I) -> ModalResult<&'i str> {
        self.as_marker().parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use vllm_tokenizer::{DecodedText, TokenAnchor, TokenAttribution};
    use winnow::Parser;
    use winnow::error::ErrMode;
    use winnow::stream::{Partial, Stream};

    use super::{
        Attributed, Marker, MarkerLike, MarkerStream, SpecialToken, attributed_with_markers,
    };
    use crate::utils::safe_text_len_mul;

    const CLOSE_ID: u32 = 256;
    const SEP_ID: u32 = 258;

    /// Build attributed text from `(piece, token_id)` pairs, one visible anchor
    /// at the first byte of each piece.
    fn pieces(pieces: &[(&str, u32)]) -> DecodedText {
        let mut decoded = DecodedText::default();
        for &(piece, token_id) in pieces {
            decoded.attributions.push(TokenAttribution {
                token_id,
                anchor: TokenAnchor::Visible {
                    byte_offset: decoded.text.len() as u32,
                },
            });
            decoded.text.push_str(piece);
        }
        decoded
    }

    fn token(text: &str, id: u32) -> SpecialToken {
        SpecialToken {
            text: text.to_string(),
            id,
        }
    }

    /// An input over `buffer` without parser markers.
    fn attributed(buffer: &DecodedText) -> Attributed<'_> {
        attributed_with_markers(buffer, &())
    }

    fn close() -> Marker {
        Marker::special(&token("<|close|>", CLOSE_ID))
    }

    fn think_close() -> Marker {
        close().then_text("think").then_special(&token("<|sep|>", SEP_ID))
    }

    /// `<|close|>` spelled by ordinary tokens.
    fn ordinary_close() -> Vec<(&'static str, u32)> {
        vec![("<", 60), ("|", 124), ("close", 900), ("|", 124), (">", 62)]
    }

    #[test]
    fn special_marker_accepts_single_token_with_expected_id() {
        let buffer = pieces(&[("<|close|>", CLOSE_ID), ("x", 1)]);
        let mut input = attributed(&buffer);

        let matched = (&close()).parse_next(&mut input).unwrap();

        assert_eq!(matched, "<|close|>");
        assert_eq!(input.remaining(), "x");
    }

    #[test]
    fn special_marker_rejects_ordinary_tokens_with_same_spelling() {
        let mut fixture = ordinary_close();
        fixture.push(("x", 1));
        let buffer = pieces(&fixture);
        let mut input = attributed(&buffer);

        let error = (&close()).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Backtrack(_)));
        assert_eq!(input.remaining(), "<|close|>x", "nothing consumed");
    }

    #[test]
    fn special_marker_rejects_single_ordinary_token_with_wrong_id() {
        let buffer = pieces(&[("<|close|>", 900)]);
        let mut input = attributed(&buffer);

        let error = (&close()).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Backtrack(_)));
    }

    #[test]
    fn special_marker_reports_incomplete_for_split_genuine_spelling() {
        // Output holdback delivered only the first bytes; the anchor came with them.
        let buffer = pieces(&[("<|clo", CLOSE_ID)]);
        let mut input = attributed(&buffer);

        let error = (&close()).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Incomplete(_)));
    }

    #[test]
    fn special_marker_rejects_partial_lookalike_without_waiting() {
        let buffer = pieces(&[("<", 60), ("|", 124), ("clo", 901)]);
        let mut input = attributed(&buffer);

        let error = (&close()).parse_next(&mut input).unwrap_err();

        assert!(matches!(error, ErrMode::Backtrack(_)));
    }

    #[test]
    fn special_marker_ignores_zero_width_records_inside_span() {
        let mut buffer = pieces(&[("<|close|>", CLOSE_ID)]);
        buffer.attributions.push(TokenAttribution {
            token_id: 7,
            anchor: TokenAnchor::ZeroWidth { byte_offset: 3 },
        });
        let mut input = attributed(&buffer);

        (&close()).parse_next(&mut input).unwrap();
    }

    #[test]
    fn marker_without_guards_matches_spelling_only() {
        let buffer = pieces(&ordinary_close());
        let marker = close().without_guards();

        (&marker).parse_next(&mut attributed(&buffer)).unwrap();
        (&marker).parse_next(&mut Partial::new("<|close|>")).unwrap();
    }

    #[test]
    fn guarded_marker_never_matches_text_only_input() {
        let error = (&close()).parse_next(&mut Partial::new("<|close|>")).unwrap_err();

        assert!(matches!(error, ErrMode::Backtrack(_)));
    }

    #[test]
    fn marker_requires_every_special_segment() {
        let marker = think_close();
        let genuine = pieces(&[("<|close|>", CLOSE_ID), ("think", 5), ("<|sep|>", SEP_ID)]);
        assert_eq!(
            (&marker).parse_next(&mut attributed(&genuine)).unwrap(),
            "<|close|>think<|sep|>"
        );

        // Real close, ordinary separator: the strict rule keeps it as content.
        let mixed = pieces(&[("<|close|>", CLOSE_ID), ("think", 5), ("<|sep|>", 902)]);
        let mut input = attributed(&mixed);
        assert!(matches!(
            (&marker).parse_next(&mut input).unwrap_err(),
            ErrMode::Backtrack(_)
        ));
        assert_eq!(
            input.remaining(),
            "<|close|>think<|sep|>",
            "nothing consumed"
        );
    }

    #[test]
    fn marker_first_token_id_requires_leading_special() {
        assert_eq!(think_close().as_marker().first_token_id(), Some(CLOSE_ID));
        assert_eq!(Marker::text("</think>").as_marker().first_token_id(), None);
        assert_eq!(
            Marker::text("x")
                .then_special(&token("<|sep|>", SEP_ID))
                .as_marker()
                .first_token_id(),
            None
        );
        assert_eq!("</think>".as_marker().first_token_id(), None);
    }

    #[test]
    fn attributed_next_candidate_uses_anchors_for_guarded_markers() {
        let mut fixture = vec![("say ", 1)];
        fixture.extend(ordinary_close()); // lookalike at 4
        fixture.extend([("think", 5), ("<|sep|>", 903), (" then ", 2)]);
        let genuine_at = fixture.iter().map(|(piece, _)| piece.len()).sum::<usize>();
        fixture.extend([("<|close|>", CLOSE_ID), ("think", 5), ("<|sep|>", SEP_ID)]);
        let buffer = pieces(&fixture);
        let input = attributed(&buffer);

        let marker = think_close();
        assert_eq!(input.next_candidate(&[&marker], 0), Some(genuine_at));
        assert_eq!(input.next_candidate(&[&marker], genuine_at + 1), None);
    }

    #[test]
    fn attributed_next_candidate_falls_back_to_text_for_unguarded_markers() {
        let buffer = pieces(&[("a", 1), ("</think>", 2), ("b", 3)]);
        let input = attributed(&buffer);

        assert_eq!(input.next_candidate(&["</think>"], 0), Some(1));
        assert_eq!(input.next_candidate(&["</thinking>"], 0), None);
        // Partial spelling at the end of the input is a candidate too.
        let buffer = pieces(&[("a", 1), ("</thi", 2)]);
        let input = attributed(&buffer);
        assert_eq!(input.next_candidate(&["</think>"], 0), Some(1));
    }

    #[test]
    fn text_next_candidate_matches_previous_scan_semantics() {
        let input = Partial::new("hello<not_marker><|tool");

        assert_eq!(
            input.next_candidate(&["<|tool_call>", "<|channel>thought\n"], 0),
            Some("hello<not_marker>".len())
        );
        assert_eq!(input.next_candidate(&["<|tool_call>"], 18), None);
        assert_eq!(
            Partial::new("hello<channel|><|tool_call>")
                .next_candidate(&["<|tool_call>", "<channel|>"], 0),
            Some(5)
        );
    }

    #[test]
    fn scan_skips_rejected_text_candidate_starting_with_multibyte_char() {
        // A marker whose leading text is found by spelling but whose guard then
        // fails: the scan must move past the whole char, not retry inside it.
        let marker = Marker::text("é").then_special(&token("<|sep|>", SEP_ID));
        let buffer = pieces(&[("aé", 1), ("<", 60), ("|sep|>", 904), ("b", 2)]);
        let mut input = attributed(&buffer);

        let len = safe_text_len_mul(&mut input, &[&marker]).unwrap();

        assert_eq!(len, buffer.text.len());
    }

    #[test]
    fn attributed_offsets_follow_the_cursor() {
        let buffer = pieces(&[("ab", 1), ("<|close|>", CLOSE_ID)]);
        let mut input = attributed(&buffer);
        input.next_slice(2);

        assert_eq!(input.offset(), 2);
        assert_eq!(input.remaining(), "<|close|>");
        (&close()).parse_next(&mut input).unwrap();
        assert_eq!(input.offset(), 11);
    }
}
