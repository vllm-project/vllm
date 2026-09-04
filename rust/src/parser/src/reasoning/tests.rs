// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use vllm_tokenizer::test_utils::TestTokenizer;
use vllm_tokenizer::{DecodedText, DynTokenizer, TokenAnchor, TokenAttribution};

use super::{
    CohereCmdReasoningParser, DeepSeekR1ReasoningParser, DelimitedReasoningParser,
    KimiReasoningParser, MiniMaxM3ReasoningParser, Qwen3ReasoningParser, ReasoningDelta,
    ReasoningParser, Result, SeedOssReasoningParser, Step3p5ReasoningParser,
};

pub(crate) const THINK_START_ID: u32 = 256;
pub(crate) const THINK_END_ID: u32 = 257;
pub(crate) const START_THINKING_ID: u32 = 258;
pub(crate) const END_THINKING_ID: u32 = 259;
pub(crate) const MINIMAX_THINK_START_ID: u32 = 260;
pub(crate) const MINIMAX_THINK_END_ID: u32 = 261;
pub(crate) const SPECIAL_BOUNDARY_ID: u32 = 262;
pub(crate) const MM_THINK_START_ID: u32 = 263;
pub(crate) const MM_THINK_END_ID: u32 = 264;
pub(crate) const SEED_THINK_START_ID: u32 = 265;
pub(crate) const SEED_THINK_END_ID: u32 = 266;

pub(crate) fn fake_tokenizer() -> TestTokenizer {
    TestTokenizer::new()
        .with_regular_token("<think>", THINK_START_ID)
        .with_regular_token("</think>", THINK_END_ID)
        .with_regular_token("<|START_THINKING|>", START_THINKING_ID)
        .with_regular_token("<|END_THINKING|>", END_THINKING_ID)
        .with_regular_token("◁think▷", MINIMAX_THINK_START_ID)
        .with_regular_token("◁/think▷", MINIMAX_THINK_END_ID)
        .with_special_token("<special-boundary>", SPECIAL_BOUNDARY_ID)
        .with_regular_token("<mm:think>", MM_THINK_START_ID)
        .with_regular_token("</mm:think>", MM_THINK_END_ID)
        .with_regular_token("<seed:think>", SEED_THINK_START_ID)
        .with_regular_token("</seed:think>", SEED_THINK_END_ID)
}

/// Feed an unattributed text delta into a reasoning parser.
pub(crate) fn push_str<P: ReasoningParser + ?Sized>(parser: &mut P, delta: &str) -> ReasoningDelta {
    parser.push(DecodedText::unattributed(delta)).unwrap()
}

/// Return the reasoning text of a delta, if any.
pub(crate) fn reasoning_str(delta: &ReasoningDelta) -> Option<&str> {
    delta.reasoning.as_ref().map(|piece| piece.text.as_str())
}

/// Return the content text of a delta, if any.
pub(crate) fn content_str(delta: &ReasoningDelta) -> Option<&str> {
    delta.content.as_ref().map(|piece| piece.text.as_str())
}

#[test]
fn delimited_content_only_stream() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    let delta = parser.push(DecodedText::unattributed("plain content"));
    assert_eq!(content_str(&delta), Some("plain content"));
}

#[test]
fn delimited_single_chunk_with_reasoning_and_content() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    let delta = parser.push(DecodedText::unattributed("<think>reason</think>answer"));
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn delimited_partial_tokens_across_chunks() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    assert!(parser.push(DecodedText::unattributed("<thi")).is_empty());
    let delta = parser.push(DecodedText::unattributed("nk>reason</think>answer"));
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn delimited_finish_flushes_buffer() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();
    parser.initialize(&[THINK_START_ID]);

    let delta = parser.push(DecodedText::unattributed("unfinished</thi"));
    assert_eq!(reasoning_str(&delta), Some("unfinished"));
    let final_delta = parser.finish();
    assert_eq!(reasoning_str(&final_delta), Some("</thi"));
}

#[test]
fn qwen3_without_prompt_markers_expects_start_token() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    let delta = push_str(&mut parser, "reason</think>answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(content_str(&delta), Some("reason</think>answer"));
}

#[test]
fn qwen3_prompt_end_marker_starts_in_content() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[THINK_END_ID]).unwrap();

    let delta = push_str(&mut parser, "answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn qwen3_tolerates_old_and_new_formats() {
    let tokenizer = Arc::new(fake_tokenizer());

    let mut old_parser = Qwen3ReasoningParser::new(tokenizer.clone()).unwrap();
    let old = push_str(&mut old_parser, "<think>reason</think>answer");
    assert_eq!(reasoning_str(&old), Some("reason"));
    assert_eq!(content_str(&old), Some("answer"));

    let mut new_parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
    new_parser.initialize(&[THINK_START_ID]).unwrap();
    let new = push_str(&mut new_parser, "reason</think>answer");
    assert_eq!(reasoning_str(&new), Some("reason"));
    assert_eq!(content_str(&new), Some("answer"));
}

#[test]
fn qwen3_stops_scanning_at_last_special_token() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[THINK_START_ID, SPECIAL_BOUNDARY_ID]).unwrap();

    let delta = push_str(&mut parser, "answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn deepseek_r1_defaults_to_reasoning_without_prompt_boundary() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DeepSeekR1ReasoningParser::new(tokenizer).unwrap();

    let delta = push_str(&mut parser, "reason</think>answer");
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn deepseek_r1_stops_scanning_at_last_special_token() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DeepSeekR1ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[THINK_END_ID, SPECIAL_BOUNDARY_ID]).unwrap();

    let delta = push_str(&mut parser, "reason</think>answer");
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn minimax_m3_handles_explicit_think_delimiters() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    let delta = push_str(&mut parser, "<mm:think>reason</mm:think>answer");
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn minimax_m3_drops_leading_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    let delta = push_str(&mut parser, "</mm:think>answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn minimax_m3_preserves_non_leading_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    let delta = push_str(&mut parser, "XXX</mm:think>YYY");
    assert_eq!(delta.reasoning, None);
    assert_eq!(content_str(&delta), Some("XXX</mm:think>YYY"));
}

#[test]
fn minimax_m3_drops_split_leading_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    assert!(push_str(&mut parser, "</mm").is_empty());
    let delta = push_str(&mut parser, ":think>answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn minimax_m3_uses_prompt_prefilled_start_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[MM_THINK_START_ID]).unwrap();

    let delta = push_str(&mut parser, "reason</mm:think>answer");
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn minimax_m3_uses_prompt_prefilled_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[MM_THINK_END_ID]).unwrap();

    let delta = push_str(&mut parser, "answer");
    assert_eq!(delta.reasoning, None);
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn delimited_zero_width_only_piece_is_attributed_to_current_state() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();
    parser.initialize(&[THINK_START_ID]);

    // A filtered special token produces a zero-width attribution with no text;
    // it must survive as a reasoning piece rather than being dropped.
    let delta = parser.push(DecodedText {
        text: String::new(),
        attributions: [TokenAttribution {
            token_id: 42,
            anchor: TokenAnchor::ZeroWidth { byte_offset: 0 },
        }]
        .into_iter()
        .collect(),
    });
    assert_eq!(reasoning_str(&delta), Some(""));
    assert_eq!(
        delta
            .reasoning
            .as_ref()
            .unwrap()
            .attributions
            .iter()
            .map(|attr| attr.token_id)
            .collect::<Vec<_>>(),
        [42]
    );
}

#[test]
fn delimited_marker_tokens_are_dropped_from_attributions() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    let mut collected = CollectedAttributions::default();
    for chunk in attributed_chunks(&[
        Some("<think>"),
        Some("reason"),
        Some("</think>"),
        Some("answer"),
    ]) {
        collected.record(parser.push(chunk));
    }
    collected.record(parser.finish());

    assert_eq!(collected.reasoning_text, "reason");
    assert_eq!(collected.content_text, "answer");
    // The marker tokens (1 and 3) are dropped with their spans; only the
    // reasoning and content tokens remain attributed.
    assert_eq!(collected.reasoning_ids, [2]);
    assert_eq!(collected.content_ids, [4]);
}

#[test]
fn reasoning_parsers_conserve_token_attributions() {
    type ParserCase = (
        fn(DynTokenizer) -> Result<Box<dyn ReasoningParser>>,
        u32,
        &'static str,
    );

    let parsers: Vec<ParserCase> = vec![
        (Qwen3ReasoningParser::create, THINK_START_ID, "qwen3"),
        (
            DeepSeekR1ReasoningParser::create,
            THINK_START_ID,
            "deepseek_r1",
        ),
        (KimiReasoningParser::create, MINIMAX_THINK_START_ID, "kimi"),
        (
            CohereCmdReasoningParser::create,
            START_THINKING_ID,
            "cohere_cmd",
        ),
        (
            SeedOssReasoningParser::create,
            SEED_THINK_START_ID,
            "seed_oss",
        ),
        (
            MiniMaxM3ReasoningParser::create,
            MM_THINK_START_ID,
            "minimax_m3",
        ),
        (Step3p5ReasoningParser::create, THINK_START_ID, "step3p5"),
    ];

    for (create, think_start_id, name) in parsers {
        let mut parser = create(Arc::new(fake_tokenizer())).unwrap();
        // Prompt prefills the start marker, opening reasoning before the stream.
        parser.initialize(&[think_start_id]).unwrap();

        // Script: reasoning text (1), a zero-width token inside reasoning (2),
        // the end marker (3), then visible content (4). The zero-width token
        // is attributed to reasoning, matching Python's current-state rule;
        // the marker token is dropped with its span.
        let end_marker = match name {
            "kimi" => "◁/think▷",
            "cohere_cmd" => "<|END_THINKING|>",
            "seed_oss" => "</seed:think>",
            "minimax_m3" => "</mm:think>",
            _ => "</think>",
        };
        let chunks = attributed_chunks(&[Some("reason"), None, Some(end_marker), Some("answer")]);

        let collected = collect_attributed(parser.as_mut(), chunks);

        assert_eq!(collected.reasoning_text, "reason", "{name}");
        assert_eq!(collected.content_text, "answer", "{name}");
        assert_eq!(collected.reasoning_ids, [1, 2], "{name}");
        assert_eq!(collected.content_ids, [4], "{name}");
    }
}

/// Build one chunk per piece, each carrying a single token attributed to it.
///
/// A `Some(text)` piece produces one token visibly anchored at the piece start;
/// a `None` piece produces one zero-width token. Token IDs are 1-based
/// positions in the script.
pub(crate) fn attributed_chunks(pieces: &[Option<&str>]) -> Vec<DecodedText> {
    pieces
        .iter()
        .enumerate()
        .map(|(index, piece)| {
            let token_id = index as u32 + 1;
            match piece {
                Some(text) => DecodedText {
                    text: (*text).to_string(),
                    attributions: [TokenAttribution {
                        token_id,
                        anchor: TokenAnchor::Visible { byte_offset: 0 },
                    }]
                    .into_iter()
                    .collect(),
                },
                None => DecodedText {
                    text: String::new(),
                    attributions: [TokenAttribution {
                        token_id,
                        anchor: TokenAnchor::ZeroWidth { byte_offset: 0 },
                    }]
                    .into_iter()
                    .collect(),
                },
            }
        })
        .collect()
}

/// Token attributions and text collected from a stream of reasoning deltas.
#[derive(Default)]
pub(crate) struct CollectedAttributions {
    pub reasoning_ids: Vec<u32>,
    pub content_ids: Vec<u32>,
    pub reasoning_text: String,
    pub content_text: String,
}

impl CollectedAttributions {
    pub fn record(&mut self, delta: ReasoningDelta) {
        if let Some(reasoning) = delta.reasoning {
            self.reasoning_ids
                .extend(reasoning.attributions.iter().map(|attr| attr.token_id));
            self.reasoning_text.push_str(&reasoning.text);
        }
        if let Some(content) = delta.content {
            self.content_ids.extend(content.attributions.iter().map(|attr| attr.token_id));
            self.content_text.push_str(&content.text);
        }
    }
}

/// Push attributed chunks through a reasoning parser and collect what is
/// attributed to reasoning and content pieces.
pub(crate) fn collect_attributed(
    parser: &mut dyn ReasoningParser,
    chunks: Vec<DecodedText>,
) -> CollectedAttributions {
    let mut collected = CollectedAttributions::default();
    for chunk in chunks {
        collected.record(parser.push(chunk).unwrap());
    }
    collected.record(parser.finish().unwrap());
    collected
}
