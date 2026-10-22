// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use vllm_tokenizer::test_utils::TestTokenizer;
use vllm_tokenizer::{DecodedText, DynTokenizer, TokenAnchor, TokenAttribution};

use super::{
    CohereCmdReasoningParser, DeepSeekR1ReasoningParser, DelimitedReasoningParser,
    DelimitedReasoningParserBuilder, KimiReasoningParser, MiniMaxM3ReasoningParser,
    Qwen3ReasoningParser, ReasoningDelta, ReasoningParser, Result, SeedOssReasoningParser,
    Step3p5ReasoningParser,
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
    let mut parser = DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
        .build()
        .unwrap();

    let delta = parser.push(DecodedText::unattributed("plain content"));
    assert_eq!(content_str(&delta), Some("plain content"));
}

#[test]
fn delimited_single_chunk_with_reasoning_and_content() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
        .build()
        .unwrap();

    let delta = parser.push(DecodedText::unattributed("<think>reason</think>answer"));
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn delimited_partial_tokens_across_chunks() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
        .build()
        .unwrap();

    assert!(parser.push(DecodedText::unattributed("<thi")).is_empty());
    let delta = parser.push(DecodedText::unattributed("nk>reason</think>answer"));
    assert_eq!(reasoning_str(&delta), Some("reason"));
    assert_eq!(content_str(&delta), Some("answer"));
}

#[test]
fn delimited_finish_flushes_buffer() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
        .build()
        .unwrap();
    parser.initialize(&[THINK_START_ID]).unwrap();

    let delta = parser.push(DecodedText::unattributed("unfinished</thi"));
    assert_eq!(reasoning_str(&delta), Some("unfinished"));
    let final_delta = parser.finish();
    assert_eq!(reasoning_str(&final_delta), Some("</thi"));
}

#[test]
fn delimited_zero_width_only_piece_is_attributed_to_current_state() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
        .build()
        .unwrap();
    parser.initialize(&[THINK_START_ID]).unwrap();

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
    let mut parser = DelimitedReasoningParserBuilder::new(tokenizer, "<think>", "</think>")
        .build()
        .unwrap();

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

fn framed_parser() -> DelimitedReasoningParser {
    DelimitedReasoningParserBuilder::new(Arc::new(fake_tokenizer()), "<think>", "</think>")
        .with_after_start("\n")
        .with_before_end("\n")
        .with_after_end("\n\n")
        .build()
        .unwrap()
}

fn collect_framed(parser: &mut DelimitedReasoningParser, chunks: &[&str]) -> (String, String) {
    let mut result = (String::new(), String::new());
    for chunk in chunks.iter().copied().map(Some).chain(std::iter::once(None)) {
        let delta = match chunk {
            Some(chunk) => parser.push(DecodedText::unattributed(chunk)),
            None => parser.finish(),
        };
        if let Some(reasoning) = delta.reasoning {
            result.0.push_str(&reasoning.text);
        }
        if let Some(content) = delta.content {
            result.1.push_str(&content.text);
        }
    }
    result
}

#[test]
fn delimited_framing_preserves_body_whitespace_at_every_split() {
    let wire = "<think>\n\n  reason\t\n\n</think>\n\n\n    answer\n";
    let expected = ("\n  reason\t\n".to_string(), "\n    answer\n".to_string());
    for split in 0..=wire.len() {
        let mut parser = framed_parser();
        assert_eq!(
            collect_framed(&mut parser, &[&wire[..split], &wire[split..]]),
            expected,
            "split at {split}"
        );
    }
    let chars = wire
        .as_bytes()
        .chunks(1)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(collect_framed(&mut framed_parser(), &chars), expected);
}

#[test]
fn delimited_partial_framing_is_preserved_on_mismatch_or_finish() {
    for (wire, reasoning, content) in [
        ("<think>reason</think>\nanswer", "reason", "\nanswer"),
        ("<think>\nreason\nmore\n", "reason\nmore\n", ""),
        ("<think>\nreason\n</thi", "reason\n</thi", ""),
        ("<think>\nreason\n</think>\n", "reason", "\n"),
        ("  plain\n", "", "  plain\n"),
    ] {
        for split in 0..=wire.len() {
            assert_eq!(
                collect_framed(&mut framed_parser(), &[&wire[..split], &wire[split..]]),
                (reasoning.to_string(), content.to_string()),
                "wire {wire:?}, split {split}"
            );
        }
    }
}

#[test]
fn delimited_prompt_initialization_consumes_only_remaining_framing() {
    use vllm_tokenizer::Tokenizer;
    let tokenizer = fake_tokenizer();
    for (prompt, wire, reasoning, content) in [
        (
            "<think>",
            "\nreason\n</think>\n\nanswer",
            "reason",
            "answer",
        ),
        (
            "<think>\n",
            "\nreason\n</think>\n\nanswer",
            "\nreason",
            "answer",
        ),
        (
            "<think>\nprefill",
            "\nreason\n</think>\n\nanswer",
            "\nreason",
            "answer",
        ),
        ("</think>", "\n\n\nanswer", "", "\nanswer"),
        ("</think>\n", "\n\nanswer", "", "\nanswer"),
        ("</think>\n\n", "\nanswer", "", "\nanswer"),
        ("</think>\n\nprefill", "\nanswer", "", "\nanswer"),
    ] {
        let mut parser = framed_parser();
        parser.initialize(&tokenizer.encode(prompt, false).unwrap()).unwrap();
        assert_eq!(
            collect_framed(&mut parser, &[wire]),
            (reasoning.to_string(), content.to_string()),
            "prompt {prompt:?}"
        );
    }
}

#[test]
fn delimited_framing_holds_only_boundary_candidates() {
    let mut parser = framed_parser();
    let first = parser.push(DecodedText::unattributed("<think>\nreason\n"));
    assert_eq!(reasoning_str(&first), Some("reason"));
    let more = parser.push(DecodedText::unattributed("more\n"));
    assert_eq!(reasoning_str(&more), Some("\nmore"));
    assert!(parser.push(DecodedText::unattributed("</think>\n")).is_empty());
    let answer = parser.push(DecodedText::unattributed("\n    answer"));
    assert_eq!(content_str(&answer), Some("    answer"));
}

#[test]
fn delimited_start_prefix_is_consumed_only_before_the_marker() {
    for wire in [
        "\n<think>reason</think>answer",
        "\n\n<think>reason</think>answer",
    ] {
        for split in 0..=wire.len() {
            let mut parser = DelimitedReasoningParserBuilder::new(
                Arc::new(fake_tokenizer()),
                "<think>",
                "</think>",
            )
            .with_before_start("\n")
            .build()
            .unwrap();
            let expected_content = if wire.starts_with("\n\n") {
                "\nanswer"
            } else {
                "answer"
            };
            assert_eq!(
                collect_framed(&mut parser, &[&wire[..split], &wire[split..]]),
                ("reason".to_string(), expected_content.to_string())
            );
        }
    }
}

#[test]
fn delimited_framing_preserves_body_token_attributions() {
    let mut parser = framed_parser();
    let mut collected = CollectedAttributions::default();
    for chunk in attributed_chunks(&[
        Some("<think>"),
        Some("\n"),
        Some("reason"),
        None,
        Some("\n"),
        Some("</think>"),
        Some("\n"),
        Some("\n"),
        Some("    answer"),
    ]) {
        collected.record(parser.push(chunk));
    }
    collected.record(parser.finish());
    assert_eq!(collected.reasoning_text, "reason");
    assert_eq!(collected.content_text, "    answer");
    assert_eq!(collected.reasoning_ids, [3, 4]);
    assert_eq!(collected.content_ids, [9]);
}
