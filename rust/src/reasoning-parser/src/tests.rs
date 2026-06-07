use std::sync::Arc;

use vllm_tokenizer::Tokenizer;

use super::{
    DeepSeekR1ReasoningParser, DelimitedReasoningParser, Qwen3ReasoningParser, ReasoningParser,
    SeedOssReasoningParser, Step3p5ReasoningParser,
};

struct FakeTokenizer;

impl Tokenizer for FakeTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn decode(
        &self,
        token_ids: &[u32],
        _skip_special_tokens: bool,
    ) -> vllm_tokenizer::Result<String> {
        Ok(token_ids
            .iter()
            .map(|token_id| char::from_u32(*token_id).unwrap_or('\u{FFFD}'))
            .collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<think>" => Some(1),
            "</think>" => Some(2),
            "<|START_THINKING|>" => Some(3),
            "<|END_THINKING|>" => Some(4),
            "◁think▷" => Some(5),
            "◁/think▷" => Some(6),
            "<seed:think>" => Some(10),
            "</seed:think>" => Some(11),
            _ => None,
        }
    }

    fn is_special_id(&self, token_id: u32) -> bool {
        token_id == 7
    }
}

#[test]
fn delimited_content_only_stream() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    assert_eq!(
        parser.push("plain content").content.as_deref(),
        Some("plain content")
    );
}

#[test]
fn delimited_single_chunk_with_reasoning_and_content() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    let delta = parser.push("<think>reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn delimited_partial_tokens_across_chunks() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    assert!(parser.push("<thi").is_empty());
    let delta = parser.push("nk>reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn delimited_finish_flushes_buffer() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();
    parser.initialize(&[1]);

    let delta = parser.push("unfinished</thi");
    assert_eq!(delta.reasoning.as_deref(), Some("unfinished"));
    let final_delta = parser.finish();
    assert_eq!(final_delta.reasoning.as_deref(), Some("</thi"));
}

#[test]
fn qwen3_without_prompt_markers_expects_start_token() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("reason</think>answer"));
}

#[test]
fn qwen3_prompt_end_marker_starts_in_content() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[2]).unwrap();

    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn qwen3_tolerates_old_and_new_formats() {
    let tokenizer = Arc::new(FakeTokenizer);

    let mut old_parser = Qwen3ReasoningParser::new(tokenizer.clone()).unwrap();
    let old = old_parser.push("<think>reason</think>answer").unwrap();
    assert_eq!(old.reasoning.as_deref(), Some("reason"));
    assert_eq!(old.content.as_deref(), Some("answer"));

    let mut new_parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
    new_parser.initialize(&[1]).unwrap();
    let new = new_parser.push("reason</think>answer").unwrap();
    assert_eq!(new.reasoning.as_deref(), Some("reason"));
    assert_eq!(new.content.as_deref(), Some("answer"));
}

#[test]
fn qwen3_stops_scanning_at_last_special_token() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[1, 7]).unwrap();

    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn deepseek_r1_defaults_to_reasoning_without_prompt_boundary() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = DeepSeekR1ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn deepseek_r1_stops_scanning_at_last_special_token() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = DeepSeekR1ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[2, 7]).unwrap();

    let delta = parser.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn seed_oss_without_prompt_markers_expects_start_token() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("implicit reasoning</seed:think>answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(
        delta.content.as_deref(),
        Some("implicit reasoning</seed:think>answer")
    );
}

#[test]
fn seed_oss_picks_up_prompt_start_boundary() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
    // Prompt prefills `<seed:think>` (id 10), opening reasoning before the stream.
    parser.initialize(&[10]).unwrap();

    let delta = parser.push("reason</seed:think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn seed_oss_respects_prompt_end_boundary() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
    // Prompt already closed reasoning with `</seed:think>` (id 11).
    parser.initialize(&[11]).unwrap();

    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn step3p5_picks_up_prompt_start_boundary() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();
    // Prompt prefills `<think>` (id 1), opening reasoning before the stream.
    parser.initialize(&[1]).unwrap();

    let delta = parser.push("This is a reasoning section</think>This is the rest").unwrap();
    assert_eq!(
        delta.reasoning.as_deref(),
        Some("This is a reasoning section")
    );
    assert_eq!(delta.content.as_deref(), Some("This is the rest"));
}

#[test]
fn step3p5_handles_unterminated_reasoning() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let pushed = parser.push("<think>reason without end").unwrap();
    assert_eq!(pushed.reasoning.as_deref(), Some("reason without end"));
    assert_eq!(pushed.content, None);

    let flushed = parser.finish().unwrap();
    assert!(flushed.is_empty());
}

#[test]
fn step3p5_handles_empty_input() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let pushed = parser.push("").unwrap();
    assert!(pushed.is_empty());
    let flushed = parser.finish().unwrap();
    assert!(flushed.is_empty());
}

#[test]
fn step3p5_complex_newline_pattern_trims_only_single_framing_newline_each_side() {
    // Only the immediately-adjacent framing `\n` is dropped on each side of
    // `</think>`; surrounding newlines remain part of reasoning/content.
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[1]).unwrap();

    let delta = parser
        .push("\n This is a \n reasoning section\n\n\n</think>\n\nThis is the rest")
        .unwrap();
    assert_eq!(
        delta.reasoning.as_deref(),
        Some("\n This is a \n reasoning section\n\n")
    );
    assert_eq!(delta.content.as_deref(), Some("\nThis is the rest"));
}

#[test]
fn step3p5_drops_framing_newlines_in_single_push() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("<think>reason\n</think>\nanswer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn step3p5_drops_framing_newlines_across_pushes() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    // The trailing `\n` from the first push is held until we know whether
    // `</think>` follows.
    let first = parser.push("<think>reason\n").unwrap();
    assert_eq!(first.reasoning.as_deref(), Some("reason"));
    assert_eq!(first.content, None);

    // `</think>` arrives standalone; the held newline should be dropped.
    let second = parser.push("</think>").unwrap();
    assert!(second.is_empty());

    // The leading newline of the first content delta is dropped.
    let third = parser.push("\nanswer").unwrap();
    assert_eq!(third.reasoning, None);
    assert_eq!(third.content.as_deref(), Some("answer"));
}

#[test]
fn step3p5_replays_held_newline_when_more_reasoning_follows() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let first = parser.push("<think>reason\n").unwrap();
    assert_eq!(first.reasoning.as_deref(), Some("reason"));

    let second = parser.push("more reason").unwrap();
    assert_eq!(second.reasoning.as_deref(), Some("\nmore reason"));
    assert_eq!(second.content, None);
}

#[test]
fn step3p5_finish_flushes_held_newline_in_unterminated_stream() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let first = parser.push("<think>reason\n").unwrap();
    assert_eq!(first.reasoning.as_deref(), Some("reason"));

    let flushed = parser.finish().unwrap();
    assert_eq!(flushed.reasoning.as_deref(), Some("\n"));
    assert_eq!(flushed.content, None);
}

#[test]
fn step3p5_preserves_inner_newlines_in_reasoning() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("<think>line1\nline2</think>tail").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("line1\nline2"));
    assert_eq!(delta.content.as_deref(), Some("tail"));
}

#[test]
fn step3p5_trims_only_one_trailing_reasoning_newline() {
    // Only the single framing newline immediately before `</think>` is
    // dropped; earlier newlines in the reasoning body are preserved.
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("<think>reason\n\n</think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason\n"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn step3p5_drops_only_first_content_newline_after_transition() {
    // The leading-`\n` drop applies only to the first content delta after
    // `</think>`; later deltas pass through untouched.
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let first = parser.push("<think>reason</think>").unwrap();
    assert_eq!(first.reasoning.as_deref(), Some("reason"));
    assert_eq!(first.content, None);

    let second = parser.push("\nfirst").unwrap();
    assert_eq!(second.reasoning, None);
    assert_eq!(second.content.as_deref(), Some("first"));

    // A `\n` arriving in a later content delta must NOT be dropped.
    let third = parser.push("\nsecond").unwrap();
    assert_eq!(third.reasoning, None);
    assert_eq!(third.content.as_deref(), Some("\nsecond"));
}

#[test]
fn step3p5_passes_through_clean_boundary_without_framing_newlines() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("<think>reason</think>tail").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("tail"));
}

#[test]
fn step3p5_handles_empty_reasoning_section() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = Step3p5ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("<think></think>answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn seed_oss_handles_explicit_start_token() {
    // An explicit start delimiter must not leak into reasoning text.
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("<seed:think>reason</seed:think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn seed_oss_streams_explicit_start_token_across_pushes() {
    // Start token, reasoning body, end token, and content arrive in separate
    // streaming deltas.
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();

    let mut reasoning = String::new();
    let mut content = String::new();
    for delta_str in [
        "<seed:think>",
        "Some ",
        "reasoning ",
        "content",
        "</seed:think>",
        "Final ",
        "answer",
    ] {
        let delta = parser.push(delta_str).unwrap();
        if let Some(r) = delta.reasoning {
            reasoning.push_str(&r);
        }
        if let Some(c) = delta.content {
            content.push_str(&c);
        }
    }
    assert_eq!(reasoning, "Some reasoning content");
    assert_eq!(content, "Final answer");
}

#[test]
fn seed_oss_handles_partial_delimiters_across_pushes() {
    let tokenizer = Arc::new(FakeTokenizer);
    let mut parser = SeedOssReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[10]).unwrap();

    // Closing delimiter `</seed:think>` arrives in two halves.
    let first = parser.push("reason</seed:").unwrap();
    assert_eq!(first.reasoning.as_deref(), Some("reason"));
    assert_eq!(first.content, None);

    let second = parser.push("think>answer").unwrap();
    assert_eq!(second.reasoning, None);
    assert_eq!(second.content.as_deref(), Some("answer"));
}
