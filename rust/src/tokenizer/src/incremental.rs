// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::mem::take;

use smallvec::SmallVec;

use crate::{Result, Tokenizer};

/// Position of one generated token in decoded text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenAnchor {
    /// The token contributed to visible decoded text.
    Visible { byte_offset: u32 },
    /// The token produced no visible decoded bytes.
    ZeroWidth { byte_offset: u32 },
}

impl TokenAnchor {
    fn byte_offset(self) -> u32 {
        match self {
            Self::Visible { byte_offset } | Self::ZeroWidth { byte_offset } => byte_offset,
        }
    }

    fn with_byte_offset(self, byte_offset: u32) -> Self {
        match self {
            Self::Visible { .. } => Self::Visible { byte_offset },
            Self::ZeroWidth { .. } => Self::ZeroWidth { byte_offset },
        }
    }
}

/// Decoded text and the generated tokens attributed to it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodedText {
    /// Decoded UTF-8 text.
    pub text: String,
    /// One entry per generated token, in generation order.
    ///
    /// Offsets are local to `text`. Visible offsets may repeat when multiple
    /// tokens jointly decode into one character. A zero-width offset may equal
    /// `text.len()`.
    pub anchors: SmallVec<[TokenAnchor; 8]>,
}

/// Stateful incremental decoder that emits text chunks one token at a time.
pub trait IncrementalDecoder: Send {
    /// Push one generated token and return how many new string bytes were
    /// added.
    fn push_token(&mut self, token_id: u32) -> Result<usize>;

    /// Consume any text which is currently ready.
    fn next_chunk(&mut self) -> Option<DecodedText>;

    /// Flush any remaining buffered text that has not yet been emitted.
    ///
    /// Called after the final generated token to force out buffered/incomplete
    /// fragments.
    fn flush(
        &mut self,
        truncate_output_to: Option<usize>,
    ) -> Result<(Option<DecodedText>, DecodedText)>;

    /// Return cumulative decoded text so far.
    fn output(&self) -> &str;
}

#[derive(Clone, Copy, Debug)]
enum PendingAnchor {
    Unresolved,
    Resolved(TokenAnchor),
}

/// [`IncrementalDecoder`] built on [`Tokenizer::decode()`] with prefix-diffing.
///
/// This is the same sliding-window algorithm used by `tokenizers::DecodeStream`
pub(crate) struct DecodeStream<'a, T: Tokenizer + ?Sized> {
    tokenizer: &'a T,
    skip_special_tokens: bool,
    min_bytes_to_buffer: usize,
    // mutated state
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    prefix_seeded: bool,
    cumulative_output: String,
    output_index: usize,
    anchors: Vec<PendingAnchor>,
    anchor_output_index: usize,
    pending_anchor_start: Option<usize>,
}

impl<'a, T: Tokenizer + ?Sized> DecodeStream<'a, T> {
    pub(crate) fn new(
        tokenizer: &'a T,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
    ) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            min_bytes_to_buffer,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
            prefix_seeded: prompt_token_ids.is_empty(),
            cumulative_output: String::new(),
            output_index: 0,
            anchors: Vec::new(),
            anchor_output_index: 0,
            pending_anchor_start: None,
        }
    }
}

/// Try a short tail suffix first (covers a CJK glyph straddling 1-2 token
/// boundaries); beyond 6 tokens the fallback full-prompt decode is no worse
/// than baseline so widening the sweep just adds overhead.
const SAFE_SUFFIX_MIN: usize = 4;
const SAFE_SUFFIX_MAX: usize = 6;

fn offset_as_u32(offset: usize) -> u32 {
    u32::try_from(offset).expect("decoded text exceeds 4 GiB")
}

impl<T: Tokenizer + ?Sized> DecodeStream<'_, T> {
    fn push_pending_anchor(&mut self, token_id: u32) {
        let anchor = if self.skip_special_tokens && self.tokenizer.is_special_id(token_id) {
            PendingAnchor::Resolved(TokenAnchor::ZeroWidth {
                byte_offset: offset_as_u32(self.cumulative_output.len()),
            })
        } else {
            self.pending_anchor_start.get_or_insert(self.anchors.len());
            PendingAnchor::Unresolved
        };
        self.anchors.push(anchor);
    }

    fn resolve_pending_visible(&mut self, byte_offset: usize) {
        let Some(pending_anchor_start) = self.pending_anchor_start.take() else {
            return;
        };
        let byte_offset = offset_as_u32(byte_offset);
        for anchor in &mut self.anchors[pending_anchor_start..] {
            if matches!(anchor, PendingAnchor::Unresolved) {
                *anchor = PendingAnchor::Resolved(TokenAnchor::Visible { byte_offset });
            }
        }
    }

    fn resolve_pending_zero_width(&mut self) {
        let Some(pending_anchor_start) = self.pending_anchor_start.take() else {
            return;
        };
        let byte_offset = offset_as_u32(self.cumulative_output.len());
        for anchor in &mut self.anchors[pending_anchor_start..] {
            if matches!(anchor, PendingAnchor::Unresolved) {
                *anchor = PendingAnchor::Resolved(TokenAnchor::ZeroWidth { byte_offset });
            }
        }
    }

    fn truncate_anchors(&mut self, truncate_output_to: usize) {
        let byte_offset = offset_as_u32(truncate_output_to);
        for anchor in &mut self.anchors {
            let PendingAnchor::Resolved(resolved) = anchor else {
                continue;
            };
            if resolved.byte_offset() >= byte_offset {
                *resolved = TokenAnchor::ZeroWidth { byte_offset };
            }
        }
    }

    fn take_ready(&mut self, cutoff: usize) -> Option<DecodedText> {
        let chunk_start = self.output_index;
        let cutoff_u32 = offset_as_u32(cutoff);
        let mut anchor_end = self.anchor_output_index;

        for anchor in &self.anchors[self.anchor_output_index..] {
            let PendingAnchor::Resolved(anchor) = anchor else {
                break;
            };
            let ready = match anchor {
                TokenAnchor::Visible { byte_offset } => *byte_offset < cutoff_u32,
                TokenAnchor::ZeroWidth { byte_offset } => *byte_offset <= cutoff_u32,
            };
            if !ready {
                break;
            }
            anchor_end += 1;
        }

        if cutoff == chunk_start && anchor_end == self.anchor_output_index {
            return None;
        }

        let chunk_start_u32 = offset_as_u32(chunk_start);
        let anchors = self.anchors[self.anchor_output_index..anchor_end]
            .iter()
            .map(|anchor| {
                let PendingAnchor::Resolved(anchor) = anchor else {
                    unreachable!("ready anchors must be resolved")
                };
                anchor.with_byte_offset(anchor.byte_offset() - chunk_start_u32)
            })
            .collect();
        let text = self.cumulative_output[chunk_start..cutoff].to_string();

        self.output_index = cutoff;
        self.anchor_output_index = anchor_end;
        Some(DecodedText { text, anchors })
    }

    /// Decode prompt-only context for prefix seeding.
    ///
    /// Prompt ids may come from the model vocabulary rather than the local
    /// tokenizer vocabulary. For this seeding path, ids that cannot be mapped
    /// back to raw token text are dropped before retrying strict decode. This
    /// tolerance is intentionally limited to prompt context; generated ids are
    /// decoded later through the normal strict path.
    fn decode_prompt_context(&self, ids: &[u32]) -> Result<(String, Vec<u32>)> {
        match self.tokenizer.decode(ids, self.skip_special_tokens) {
            Ok(decoded) => Ok((decoded, ids.to_vec())),
            Err(error) => {
                let filtered = ids
                    .iter()
                    .copied()
                    .filter(|&id| self.tokenizer.id_to_token(id).is_some())
                    .collect::<Vec<_>>();
                if filtered.len() == ids.len() {
                    return Err(error);
                }
                self.tokenizer
                    .decode(&filtered, self.skip_special_tokens)
                    .map(|decoded| (decoded, filtered))
            }
        }
    }

    /// Seed `self.prefix` from the shortest trailing prompt suffix whose
    /// filtered context is still long enough and whose decoded text has no
    /// U+FFFD. A clean decode means the suffix starts and ends at valid
    /// UTF-8/token boundaries, so priming from it is equivalent to priming from
    /// the full prompt.
    fn seed_prefix(&mut self) -> Result<()> {
        let prompt_len = self.ids.len();
        if prompt_len > SAFE_SUFFIX_MIN {
            let max_try = SAFE_SUFFIX_MAX.min(prompt_len - 1);
            for suffix_len in SAFE_SUFFIX_MIN..=max_try {
                let start = prompt_len - suffix_len;
                let (decoded, context_ids) = self.decode_prompt_context(&self.ids[start..])?;
                if !decoded.contains('\u{FFFD}') && context_ids.len() >= SAFE_SUFFIX_MIN {
                    self.prefix = decoded;
                    self.ids = context_ids;
                    self.prefix_index = self.ids.len();
                    return Ok(());
                }
            }
        }
        let (decoded, context_ids) = self.decode_prompt_context(&self.ids)?;
        self.ids = context_ids;
        if !decoded.ends_with('\u{FFFD}') {
            self.prefix = decoded;
            self.prefix_index = self.ids.len();
        }
        Ok(())
    }
}

impl<T: Tokenizer + ?Sized> IncrementalDecoder for DecodeStream<'_, T> {
    fn push_token(&mut self, token_id: u32) -> Result<usize> {
        if !self.prefix_seeded && !self.ids.is_empty() {
            self.seed_prefix()?;
            self.prefix_seeded = true;
        }

        self.ids.push(token_id);
        self.push_pending_anchor(token_id);
        let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        let prefix_len = self.prefix.len();
        if string.len() <= prefix_len || string.ends_with('\u{FFFD}') {
            return Ok(0);
        }
        // Ensure we split at a utf-8 char boundary.
        let new_chunk = &string[string.floor_char_boundary(prefix_len)..];
        self.resolve_pending_visible(self.cumulative_output.len());
        self.cumulative_output.push_str(new_chunk);
        self.ids.drain(..self.prefix_index);
        self.prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        self.prefix_index = self.ids.len();
        Ok(new_chunk.len())
    }

    fn next_chunk(&mut self) -> Option<DecodedText> {
        let cutoff = self.cumulative_output.len().saturating_sub(self.min_bytes_to_buffer);
        // Ensure we split at a utf-8 char boundary.
        let cutoff = self.cumulative_output.floor_char_boundary(cutoff);
        self.take_ready(cutoff)
    }

    fn flush(
        &mut self,
        truncate_output_to: Option<usize>,
    ) -> Result<(Option<DecodedText>, DecodedText)> {
        // If the prefix was never seeded (no push_token was called), `ids`
        // holds only prompt context — decoding it would re-emit prompt text.
        if self.prefix_seeded && !self.ids.is_empty() {
            let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            let prefix_len = self.prefix.len();
            // Ensure we split at a utf-8 char boundary.
            let new_chunk = &string[string.floor_char_boundary(prefix_len)..];
            if !new_chunk.is_empty() {
                self.resolve_pending_visible(self.cumulative_output.len());
                self.cumulative_output.push_str(new_chunk);
            }
        }
        self.resolve_pending_zero_width();
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
        self.prefix_seeded = true;
        if let Some(truncate_output_to) = truncate_output_to {
            self.cumulative_output.truncate(truncate_output_to);
            self.truncate_anchors(truncate_output_to);
        }
        let last_chunk = self.take_ready(self.cumulative_output.len());

        let anchors = take(&mut self.anchors)
            .into_iter()
            .map(|anchor| {
                let PendingAnchor::Resolved(anchor) = anchor else {
                    unreachable!("flush must resolve every token anchor")
                };
                anchor
            })
            .collect();
        let full_text = DecodedText {
            text: take(&mut self.cumulative_output),
            anchors,
        };
        self.output_index = 0;
        self.anchor_output_index = 0;
        self.pending_anchor_start = None;
        Ok((last_chunk, full_text))
    }

    fn output(&self) -> &str {
        &self.cumulative_output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestTokenizer;

    /// Backend that treats each token ID as a raw byte, producing lossy UTF-8.
    #[derive(Debug)]
    struct Utf8Backend;

    impl Tokenizer for Utf8Backend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn encode_ordinary(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            let bytes = token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            unreachable!()
        }
    }

    /// Backend with tokens that decode to whole text pieces.
    #[derive(Debug)]
    struct PieceBackend;

    impl Tokenizer for PieceBackend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn encode_ordinary(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            Ok(token_ids
                .iter()
                .map(|token_id| match token_id {
                    1 => "abcd",
                    2 => "ab",
                    3 => "<stop>",
                    4 => "",
                    _ => unreachable!("unexpected token id: {token_id}"),
                })
                .collect())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            Some(id.to_string())
        }
    }

    const SPECIAL_TOKEN_ID: u32 = 0x100;

    /// Raw-byte backend with one filterable special token.
    #[derive(Debug)]
    struct Utf8SpecialBackend;

    impl Tokenizer for Utf8SpecialBackend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn encode_ordinary(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
            let mut bytes = Vec::new();
            for &token_id in token_ids {
                if token_id == SPECIAL_TOKEN_ID {
                    if !skip_special_tokens {
                        bytes.extend_from_slice(b"<special>");
                    }
                } else {
                    bytes.push(token_id as u8);
                }
            }
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            Some(id.to_string())
        }

        fn is_special_id(&self, token_id: u32) -> bool {
            token_id == SPECIAL_TOKEN_ID
        }
    }

    #[test]
    fn holds_incomplete_utf8_until_complete() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // 你 = U+4F60 = 0xE4 0xBD 0xA0
        assert_eq!(decoder.push_token(0xe4).unwrap(), 0);
        assert_eq!(decoder.push_token(0xbd).unwrap(), 0);
        assert_eq!(decoder.push_token(0xa0).unwrap(), 3); // "你" is 3 bytes
        assert_eq!(decoder.output(), "你");
    }

    #[test]
    fn emits_ascii_immediately() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(decoder.push_token(b'o' as u32).unwrap(), 1);
        assert_eq!(decoder.push_token(b'k' as u32).unwrap(), 1);
        assert_eq!(decoder.output(), "ok");
    }

    #[test]
    fn flush_returns_none_when_fully_consumed() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(decoder.push_token(b'o' as u32).unwrap(), 1);
        assert_eq!(decoder.next_chunk().unwrap().text, "o");
        assert_eq!(decoder.push_token(b'k' as u32).unwrap(), 1);
        assert_eq!(decoder.next_chunk().unwrap().text, "k");
        // All text already consumed via next_chunk
        let (last_chunk, full_text) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk, None);
        assert_eq!(full_text.text, "ok");
    }

    #[test]
    fn flush_emits_buffered_incomplete_utf8() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // Push incomplete multi-byte sequence — step returns 0 bytes.
        assert_eq!(decoder.push_token(0xe4).unwrap(), 0);
        assert_eq!(decoder.push_token(0xbd).unwrap(), 0);

        // Flush forces out whatever the decoder can produce (lossy replacement).
        let (last_chunk, _full_text) = decoder.flush(None).unwrap();
        assert!(last_chunk.is_some());
    }

    /// Backend where token 0 is a special token.
    #[derive(Debug)]
    struct SpecialTokenBackend;

    impl Tokenizer for SpecialTokenBackend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn encode_ordinary(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
            let mut text = String::new();
            for &token_id in token_ids {
                match token_id {
                    0 if !skip_special_tokens => text.push_str("<special>"),
                    0 => {}
                    1 => text.push('a'),
                    2 => text.push('b'),
                    _ => {}
                }
            }
            Ok(text)
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            unreachable!()
        }

        fn is_special_id(&self, token_id: u32) -> bool {
            token_id == 0
        }
    }

    fn visible(byte_offset: u32) -> TokenAnchor {
        TokenAnchor::Visible { byte_offset }
    }

    fn zero_width(byte_offset: u32) -> TokenAnchor {
        TokenAnchor::ZeroWidth { byte_offset }
    }

    fn decoded(text: &str, anchors: &[TokenAnchor]) -> DecodedText {
        DecodedText {
            text: text.to_string(),
            anchors: anchors.iter().copied().collect(),
        }
    }

    struct AttributionCase<'a> {
        name: &'static str,
        tokenizer: &'a dyn Tokenizer,
        prompt_token_ids: Vec<u32>,
        token_ids: Vec<u32>,
        skip_special_tokens: bool,
        min_bytes_to_buffer: usize,
        truncate_output_to: Option<usize>,
        expected_chunks: Vec<DecodedText>,
        expected_full: DecodedText,
    }

    #[test]
    fn token_attribution_invariants() {
        let utf8 = Utf8Backend;
        let pieces = PieceBackend;
        let special = SpecialTokenBackend;
        let utf8_special = Utf8SpecialBackend;

        let cases = vec![
            AttributionCase {
                name: "one visible anchor per ascii token",
                tokenizer: &utf8,
                prompt_token_ids: vec![],
                token_ids: vec![b'o' as u32, b'k' as u32],
                skip_special_tokens: false,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![decoded("o", &[visible(0)]), decoded("k", &[visible(0)])],
                expected_full: decoded("ok", &[visible(0), visible(1)]),
            },
            AttributionCase {
                name: "byte fallback tokens share one visible anchor",
                tokenizer: &utf8,
                prompt_token_ids: vec![],
                token_ids: vec![0xe4, 0xbd, 0xa0],
                skip_special_tokens: false,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![decoded("你", &[visible(0), visible(0), visible(0)])],
                expected_full: decoded("你", &[visible(0), visible(0), visible(0)]),
            },
            AttributionCase {
                name: "holdback splits text without repeating the token anchor",
                tokenizer: &pieces,
                prompt_token_ids: vec![],
                token_ids: vec![1],
                skip_special_tokens: false,
                min_bytes_to_buffer: 2,
                truncate_output_to: None,
                expected_chunks: vec![decoded("ab", &[visible(0)]), decoded("cd", &[])],
                expected_full: decoded("abcd", &[visible(0)]),
            },
            AttributionCase {
                name: "filtered special token is zero width at its byte boundary",
                tokenizer: &special,
                prompt_token_ids: vec![],
                token_ids: vec![1, 0, 2],
                skip_special_tokens: true,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![
                    decoded("a", &[visible(0)]),
                    decoded("", &[zero_width(0)]),
                    decoded("b", &[visible(0)]),
                ],
                expected_full: decoded("ab", &[visible(0), zero_width(1), visible(1)]),
            },
            AttributionCase {
                name: "zero width preserves order inside a byte fallback group",
                tokenizer: &utf8_special,
                prompt_token_ids: vec![],
                token_ids: vec![0xe4, SPECIAL_TOKEN_ID, 0xbd, 0xa0],
                skip_special_tokens: true,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![decoded(
                    "你",
                    &[visible(0), zero_width(0), visible(0), visible(0)],
                )],
                expected_full: decoded("你", &[visible(0), zero_width(0), visible(0), visible(0)]),
            },
            AttributionCase {
                name: "retained special token has a visible anchor",
                tokenizer: &utf8_special,
                prompt_token_ids: vec![],
                token_ids: vec![SPECIAL_TOKEN_ID],
                skip_special_tokens: false,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![decoded("<special>", &[visible(0)])],
                expected_full: decoded("<special>", &[visible(0)]),
            },
            AttributionCase {
                name: "empty decode resolves to zero width on flush",
                tokenizer: &pieces,
                prompt_token_ids: vec![],
                token_ids: vec![4],
                skip_special_tokens: false,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![decoded("", &[zero_width(0)])],
                expected_full: decoded("", &[zero_width(0)]),
            },
            AttributionCase {
                name: "incomplete utf8 tokens share the replacement character anchor on flush",
                tokenizer: &utf8,
                prompt_token_ids: vec![],
                token_ids: vec![0xe4, 0xbd],
                skip_special_tokens: false,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![decoded("�", &[visible(0), visible(0)])],
                expected_full: decoded("�", &[visible(0), visible(0)]),
            },
            AttributionCase {
                name: "truncation converts a fully removed token to zero width",
                tokenizer: &pieces,
                prompt_token_ids: vec![],
                token_ids: vec![2, 3],
                skip_special_tokens: false,
                min_bytes_to_buffer: 32,
                truncate_output_to: Some(2),
                expected_chunks: vec![decoded("ab", &[visible(0), zero_width(2)])],
                expected_full: decoded("ab", &[visible(0), zero_width(2)]),
            },
            AttributionCase {
                name: "truncation inside a token retains its first-byte anchor",
                tokenizer: &pieces,
                prompt_token_ids: vec![],
                token_ids: vec![1],
                skip_special_tokens: false,
                min_bytes_to_buffer: 32,
                truncate_output_to: Some(2),
                expected_chunks: vec![decoded("ab", &[visible(0)])],
                expected_full: decoded("ab", &[visible(0)]),
            },
            AttributionCase {
                name: "prompt context contributes no generated-token anchors",
                tokenizer: &utf8,
                prompt_token_ids: vec![b'H' as u32, b'i' as u32],
                token_ids: vec![b'!' as u32],
                skip_special_tokens: false,
                min_bytes_to_buffer: 0,
                truncate_output_to: None,
                expected_chunks: vec![decoded("!", &[visible(0)])],
                expected_full: decoded("!", &[visible(0)]),
            },
        ];

        for case in cases {
            let mut decoder = case.tokenizer.create_decode_stream(
                &case.prompt_token_ids,
                case.skip_special_tokens,
                case.min_bytes_to_buffer,
            );
            let mut chunks = Vec::new();

            for token_id in &case.token_ids {
                decoder.push_token(*token_id).unwrap();
                while let Some(chunk) = decoder.next_chunk() {
                    chunks.push(chunk);
                }
            }
            let (remaining, full) = decoder.flush(case.truncate_output_to).unwrap();
            if let Some(remaining) = remaining {
                chunks.push(remaining);
            }

            assert_eq!(chunks, case.expected_chunks, "{}: chunks", case.name);
            assert_eq!(full, case.expected_full, "{}: full output", case.name);
            assert_eq!(
                full.anchors.len(),
                case.token_ids.len(),
                "{}: one anchor per pushed token",
                case.name
            );

            let mut reconstructed_text = String::new();
            let mut reconstructed_anchors = Vec::new();
            for chunk in &chunks {
                let chunk_start = reconstructed_text.len() as u32;
                reconstructed_text.push_str(&chunk.text);
                reconstructed_anchors.extend(
                    chunk
                        .anchors
                        .iter()
                        .map(|anchor| anchor.with_byte_offset(anchor.byte_offset() + chunk_start)),
                );
            }
            assert_eq!(
                reconstructed_text, full.text,
                "{}: chunk text reconstructs full text",
                case.name
            );
            assert_eq!(
                reconstructed_anchors.as_slice(),
                full.anchors.as_slice(),
                "{}: chunk anchors reconstruct full anchors",
                case.name
            );
        }
    }

    #[test]
    fn respects_skip_special_tokens() {
        let backend = SpecialTokenBackend;
        let mut skip_decoder = backend.create_decode_stream(&[], true, 0);
        let mut keep_decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(skip_decoder.push_token(0).unwrap(), 0);
        assert_eq!(keep_decoder.push_token(0).unwrap(), 9); // "<special>" is 9 bytes
        assert_eq!(keep_decoder.output(), "<special>");
    }

    #[test]
    fn prompt_tokens_provide_context_without_re_emission() {
        let backend = Utf8Backend;
        let prompt = &[b'H' as u32, b'i' as u32];
        let mut decoder = backend.create_decode_stream(prompt, false, 0);

        // First generated token should not re-emit "Hi".
        let added = decoder.push_token(b'!' as u32).unwrap();
        assert_eq!(added, 1);
        assert_eq!(decoder.output(), "!");
    }

    #[test]
    fn prompt_context_filters_unknown_ids() {
        let tokenizer = TestTokenizer::new();
        assert_eq!(tokenizer.id_to_token(10_000), None);

        let cases: &[(&str, &[u32], u32, &str)] = &[
            (
                "suffix seed",
                &[
                    b'a' as u32,
                    b'b' as u32,
                    b'c' as u32,
                    10_000,
                    b'H' as u32,
                    b'i' as u32,
                ],
                b'!' as u32,
                "!",
            ),
            ("all unknown", &[10_000], b'!' as u32, "!"),
            (
                "unknown before incomplete utf-8",
                &[10_000, 0xe4, 0xbd],
                0xa0,
                "你",
            ),
            (
                "incomplete utf-8 before filtered suffix",
                &[0xe4, 0xbd, 10_000, 10_001, 10_002, 10_003, 10_004, 10_005],
                0xa0,
                "你",
            ),
        ];

        for &(name, prompt, token_id, output) in cases {
            let mut decoder = tokenizer.create_decode_stream(prompt, false, 0);
            assert_eq!(
                decoder.push_token(token_id).unwrap(),
                output.len(),
                "{name}"
            );
            assert_eq!(decoder.output(), output, "{name}");
        }
    }

    #[test]
    fn generated_unknown_ids_still_return_decode_error() {
        let tokenizer = TestTokenizer::new();
        assert_eq!(tokenizer.id_to_token(10_000), None);

        let prompt = &[10_000, b'H' as u32, b'i' as u32];
        let mut decoder = tokenizer.create_decode_stream(prompt, false, 0);

        let error = decoder.push_token(10_000).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("test tokenizer cannot decode unknown token id 10000")
        );
    }

    #[test]
    fn chunks_concatenate_to_full_text() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        let input = b"Hello, world!";
        let mut out = String::new();
        for &byte in input {
            decoder.push_token(byte as u32).unwrap();
            if let Some(chunk) = decoder.next_chunk() {
                out.push_str(&chunk.text);
            }
        }
        let (last_chunk, full) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk, None); // all consumed via next_chunk
        assert_eq!(out, "Hello, world!");
        assert_eq!(full.text, "Hello, world!");
    }

    /// Backend simulating non-monotonic decode where adding a token changes how
    /// earlier tokens decode (context-dependent normalization), causing
    /// prefix_len to land mid-UTF-8. Reproduces the class of bug from
    /// vllm-project/vllm#17448.
    #[derive(Debug)]
    struct NonMonotonicBackend;

    impl Tokenizer for NonMonotonicBackend {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn encode_ordinary(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            match token_ids {
                [1] => Ok("abc".into()),
                [1, 2] => Ok("ab".into()),
                // Token 3 triggers a normalization change: "ab" becomes emoji + "d".
                // prefix_len=3 ("abc") lands inside the 4-byte emoji 🎉.
                [1, 2, 3] => Ok("🎉d".into()), // 🎉 is 4 bytes + d = 5 bytes
                [2, 3] => Ok("🎉d".into()),    // prefix recompute after drain
                [3] => Ok("d".into()),         // after drain
                _ => panic!("unexpected decode: {:?}", token_ids),
            }
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            unreachable!()
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            unreachable!()
        }
    }

    /// Without the char-boundary fix, this panics slicing mid-emoji.
    #[test]
    fn non_monotonic_decode_does_not_panic() {
        let backend = NonMonotonicBackend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // Token 1: "abc", prefix="abc"
        assert_eq!(decoder.push_token(1).unwrap(), 3);
        // Token 2: "ab" (shorter), no emit
        assert_eq!(decoder.push_token(2).unwrap(), 0);
        // Token 3: "🎉d" — prefix_len=3 is mid-emoji. Without fix this panics.
        let added = decoder.push_token(3).unwrap();
        assert!(added > 0);
    }

    #[test]
    fn next_chunk_with_hold_back() {
        let backend = Utf8Backend;
        // hold_back_bytes: 3 means we buffer the last 3 bytes
        let mut decoder = backend.create_decode_stream(&[], false, 3);

        let input = b"Hello!";
        let mut chunks = String::new();
        for &byte in input {
            decoder.push_token(byte as u32).unwrap();
            if let Some(chunk) = decoder.next_chunk() {
                chunks.push_str(&chunk.text);
            }
        }
        // With hold_back_bytes=3, last 3 bytes ("lo!") are held back
        assert_eq!(chunks, "Hel");
        // Flush returns the rest
        let (last_chunk, full) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk.unwrap().text, "lo!");
        assert_eq!(full.text, "Hello!");
    }

    #[test]
    fn next_chunk_cutoff_respects_char_boundary() {
        // Regression: next_chunk's cutoff (len - min_bytes_to_buffer) must be
        // aligned to a UTF-8 char boundary like push_token/flush; otherwise
        // streaming multi-byte output (CJK/emoji) with a hold-back buffer (set
        // by a stop string) panics slicing cumulative_output mid-character.
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 2);
        let mut out = String::new();
        for byte in "你好A".bytes() {
            decoder.push_token(u32::from(byte)).unwrap();
            if let Some(chunk) = decoder.next_chunk() {
                out.push_str(&chunk.text);
            }
        }
        let (last_chunk, full) = decoder.flush(None).unwrap();
        if let Some(chunk) = last_chunk {
            out.push_str(&chunk.text);
        }
        assert_eq!(full.text, "你好A");
        assert_eq!(out, "你好A");
    }

    #[test]
    fn flush_without_push_token_does_not_leak_prompt() {
        let backend = Utf8Backend;
        let prompt: Vec<u32> = b"The quick brown fox jumps over the lazy dog. "
            .iter()
            .cycle()
            .take(7001)
            .map(|&b| b as u32)
            .collect();
        let mut decoder = backend.create_decode_stream(&prompt, false, 0);

        let (last_chunk, full) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk, None);
        assert_eq!(full.text, "");
    }

    #[test]
    fn flush_without_push_token_does_not_leak_undecodable_prompt_tail() {
        let backend = Utf8Backend;
        let prompt = vec![0xe4, 0xbd];
        let mut decoder = backend.create_decode_stream(&prompt, false, 0);

        let (last_chunk, full) = decoder.flush(None).unwrap();
        assert_eq!(last_chunk, None);
        assert_eq!(full.text, "");
    }
}
