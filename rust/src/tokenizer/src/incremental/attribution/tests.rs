// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::*;
use crate::{Result, Tokenizer, test_utils::TestTokenizer};

const SPECIAL_TOKEN_ID: u32 = 0x100;

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

fn tokenizer_with_special_token() -> TestTokenizer {
    TestTokenizer::new().with_special_token("<special>", SPECIAL_TOKEN_ID)
}

fn visible(byte_offset: u32) -> TokenAnchor {
    TokenAnchor::Visible { byte_offset }
}

fn zero_width(byte_offset: u32) -> TokenAnchor {
    TokenAnchor::ZeroWidth { byte_offset }
}

fn decoded(text: &str, token_ids: &[u32], anchors: &[TokenAnchor]) -> DecodedText {
    assert_eq!(token_ids.len(), anchors.len());
    DecodedText {
        text: text.to_string(),
        attributions: token_ids
            .iter()
            .copied()
            .zip(anchors.iter().copied())
            .map(|(token_id, anchor)| TokenAttribution { token_id, anchor })
            .collect(),
    }
}

#[test]
fn token_anchor_offset_by_preserves_anchor_kind() {
    for (anchor, delta, expected) in [
        (visible(3), -2, visible(1)),
        (zero_width(1), 4, zero_width(5)),
    ] {
        assert_eq!(anchor.offset_by(delta), expected);
    }
}

#[test]
fn decoded_text_append_rebases_attributions() {
    let mut combined = decoded("a", &[1], &[visible(0)]);
    combined.append(decoded(
        "你",
        &[2, 3, 4, 5],
        &[visible(0), visible(0), visible(0), zero_width(3)],
    ));

    assert_eq!(
        combined,
        decoded(
            "a你",
            &[1, 2, 3, 4, 5],
            &[
                visible(0),
                visible(1),
                visible(1),
                visible(1),
                zero_width(4),
            ],
        )
    );
}

#[test]
fn decoded_text_append_moves_allocations_into_empty_receiver() {
    let source = decoded(
        "hello",
        &[1, 2, 3, 4, 5],
        &[visible(0), visible(1), visible(2), visible(3), visible(4)],
    );
    let text_ptr = source.text.as_ptr();
    let attributions_ptr = source.attributions.as_ptr();
    let mut combined = DecodedText::default();

    combined.append(source);

    assert_eq!(combined.text.as_ptr(), text_ptr);
    assert_eq!(combined.attributions.as_ptr(), attributions_ptr);
}

#[test]
fn decoded_text_append_preserves_zero_width_only_receiver() {
    let mut combined = decoded("", &[1], &[zero_width(0)]);
    combined.append(decoded("a", &[2], &[visible(0)]));

    assert_eq!(
        combined,
        decoded("a", &[1, 2], &[zero_width(0), visible(0)])
    );
}

struct AttributionCase<'a> {
    tokenizer: &'a dyn Tokenizer,
    prompt_token_ids: Vec<u32>,
    token_ids: Vec<u32>,
    skip_special_tokens: bool,
    min_bytes_to_buffer: usize,
    truncate_output_to: Option<usize>,
    expected_chunks: Vec<DecodedText>,
    expected_full: DecodedText,
}

fn run_attribution_case(case: AttributionCase<'_>) {
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

    assert_eq!(chunks, case.expected_chunks, "chunks");
    assert_eq!(full, case.expected_full, "full output");
    assert_eq!(
        full.attributions.len(),
        case.token_ids.len(),
        "one attribution per pushed token"
    );
    assert_eq!(
        full.attributions
            .iter()
            .map(|attribution| attribution.token_id)
            .collect::<Vec<_>>(),
        case.token_ids,
        "attribution preserves pushed token IDs"
    );

    let mut reconstructed_text = String::new();
    let mut reconstructed_attributions = Vec::new();
    for chunk in &chunks {
        let chunk_start = reconstructed_text.len() as u32;
        reconstructed_text.push_str(&chunk.text);
        reconstructed_attributions.extend(chunk.attributions.iter().map(|attribution| {
            TokenAttribution {
                token_id: attribution.token_id,
                anchor: attribution.anchor.offset_by(i64::from(chunk_start)),
            }
        }));
    }
    assert_eq!(
        reconstructed_text, full.text,
        "chunk text reconstructs full text"
    );
    assert_eq!(
        reconstructed_attributions.as_slice(),
        full.attributions.as_slice(),
        "chunk attributions reconstruct full attributions"
    );
}

#[test]
fn attribution_ascii_tokens_have_one_visible_anchor_each() {
    let tokenizer = TestTokenizer::new();
    run_attribution_case(AttributionCase {
        tokenizer: &tokenizer,
        prompt_token_ids: vec![],
        token_ids: vec![b'o' as u32, b'k' as u32],
        skip_special_tokens: false,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![
            decoded("o", &[b'o' as u32], &[visible(0)]),
            decoded("k", &[b'k' as u32], &[visible(0)]),
        ],
        expected_full: decoded("ok", &[b'o' as u32, b'k' as u32], &[visible(0), visible(1)]),
    });
}

#[test]
fn attribution_byte_fallback_tokens_share_one_visible_anchor() {
    let tokenizer = TestTokenizer::new();
    run_attribution_case(AttributionCase {
        tokenizer: &tokenizer,
        prompt_token_ids: vec![],
        token_ids: vec![0xe4, 0xbd, 0xa0],
        skip_special_tokens: false,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![decoded(
            "你",
            &[0xe4, 0xbd, 0xa0],
            &[visible(0), visible(0), visible(0)],
        )],
        expected_full: decoded(
            "你",
            &[0xe4, 0xbd, 0xa0],
            &[visible(0), visible(0), visible(0)],
        ),
    });
}

#[test]
fn attribution_holdback_splits_text_without_repeating_token() {
    run_attribution_case(AttributionCase {
        tokenizer: &PieceBackend,
        prompt_token_ids: vec![],
        token_ids: vec![1],
        skip_special_tokens: false,
        min_bytes_to_buffer: 2,
        truncate_output_to: None,
        expected_chunks: vec![decoded("ab", &[1], &[visible(0)]), decoded("cd", &[], &[])],
        expected_full: decoded("abcd", &[1], &[visible(0)]),
    });
}

#[test]
fn attribution_filtered_special_token_is_zero_width_at_its_byte_boundary() {
    let tokenizer = tokenizer_with_special_token();
    run_attribution_case(AttributionCase {
        tokenizer: &tokenizer,
        prompt_token_ids: vec![],
        token_ids: vec![b'a' as u32, SPECIAL_TOKEN_ID, b'b' as u32],
        skip_special_tokens: true,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![
            decoded("a", &[b'a' as u32], &[visible(0)]),
            decoded("", &[SPECIAL_TOKEN_ID], &[zero_width(0)]),
            decoded("b", &[b'b' as u32], &[visible(0)]),
        ],
        expected_full: decoded(
            "ab",
            &[b'a' as u32, SPECIAL_TOKEN_ID, b'b' as u32],
            &[visible(0), zero_width(1), visible(1)],
        ),
    });
}

#[test]
fn attribution_zero_width_preserves_order_inside_byte_fallback_group() {
    run_attribution_case(AttributionCase {
        tokenizer: &Utf8SpecialBackend,
        prompt_token_ids: vec![],
        token_ids: vec![0xe4, SPECIAL_TOKEN_ID, 0xbd, 0xa0],
        skip_special_tokens: true,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![decoded(
            "你",
            &[0xe4, SPECIAL_TOKEN_ID, 0xbd, 0xa0],
            &[visible(0), zero_width(0), visible(0), visible(0)],
        )],
        expected_full: decoded(
            "你",
            &[0xe4, SPECIAL_TOKEN_ID, 0xbd, 0xa0],
            &[visible(0), zero_width(0), visible(0), visible(0)],
        ),
    });
}

#[test]
fn attribution_retained_special_token_has_visible_anchor() {
    run_attribution_case(AttributionCase {
        tokenizer: &Utf8SpecialBackend,
        prompt_token_ids: vec![],
        token_ids: vec![SPECIAL_TOKEN_ID],
        skip_special_tokens: false,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![decoded("<special>", &[SPECIAL_TOKEN_ID], &[visible(0)])],
        expected_full: decoded("<special>", &[SPECIAL_TOKEN_ID], &[visible(0)]),
    });
}

#[test]
fn attribution_empty_decode_resolves_to_zero_width_on_flush() {
    run_attribution_case(AttributionCase {
        tokenizer: &PieceBackend,
        prompt_token_ids: vec![],
        token_ids: vec![4],
        skip_special_tokens: false,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![decoded("", &[4], &[zero_width(0)])],
        expected_full: decoded("", &[4], &[zero_width(0)]),
    });
}

#[test]
fn attribution_incomplete_utf8_tokens_share_replacement_anchor_on_flush() {
    let tokenizer = TestTokenizer::new();
    run_attribution_case(AttributionCase {
        tokenizer: &tokenizer,
        prompt_token_ids: vec![],
        token_ids: vec![0xe4, 0xbd],
        skip_special_tokens: false,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![decoded("�", &[0xe4, 0xbd], &[visible(0), visible(0)])],
        expected_full: decoded("�", &[0xe4, 0xbd], &[visible(0), visible(0)]),
    });
}

#[test]
fn attribution_truncation_converts_fully_removed_token_to_zero_width() {
    run_attribution_case(AttributionCase {
        tokenizer: &PieceBackend,
        prompt_token_ids: vec![],
        token_ids: vec![2, 3],
        skip_special_tokens: false,
        min_bytes_to_buffer: 32,
        truncate_output_to: Some(2),
        expected_chunks: vec![decoded("ab", &[2, 3], &[visible(0), zero_width(2)])],
        expected_full: decoded("ab", &[2, 3], &[visible(0), zero_width(2)]),
    });
}

#[test]
fn attribution_truncation_inside_token_retains_first_byte_anchor() {
    run_attribution_case(AttributionCase {
        tokenizer: &PieceBackend,
        prompt_token_ids: vec![],
        token_ids: vec![1],
        skip_special_tokens: false,
        min_bytes_to_buffer: 32,
        truncate_output_to: Some(2),
        expected_chunks: vec![decoded("ab", &[1], &[visible(0)])],
        expected_full: decoded("ab", &[1], &[visible(0)]),
    });
}

#[test]
fn attribution_excludes_prompt_context() {
    let tokenizer = TestTokenizer::new();
    run_attribution_case(AttributionCase {
        tokenizer: &tokenizer,
        prompt_token_ids: vec![b'H' as u32, b'i' as u32],
        token_ids: vec![b'!' as u32],
        skip_special_tokens: false,
        min_bytes_to_buffer: 0,
        truncate_output_to: None,
        expected_chunks: vec![decoded("!", &[b'!' as u32], &[visible(0)])],
        expected_full: decoded("!", &[b'!' as u32], &[visible(0)]),
    });
}
