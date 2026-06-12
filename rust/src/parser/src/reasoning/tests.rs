use std::sync::Arc;

use vllm_tokenizer::test_utils::TestTokenizer;

use super::{
    DeepSeekR1ReasoningParser, DelimitedReasoningParser, MiniMaxM3ReasoningParser,
    MistralReasoningParser, Qwen3ReasoningParser, ReasoningParser,
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
pub(crate) const MISTRAL_THINK_START_ID: u32 = 267;
pub(crate) const MISTRAL_THINK_END_ID: u32 = 268;

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
        .with_special_token("[THINK]", MISTRAL_THINK_START_ID)
        .with_special_token("[/THINK]", MISTRAL_THINK_END_ID)
}

/// Push every delta through `parser`, then `finish()`, returning the
/// accumulated `(reasoning, content)` (empty collapses to `None`).
pub(crate) fn run_streaming(
    parser: &mut dyn ReasoningParser,
    output: &[&str],
) -> (Option<String>, Option<String>) {
    let mut reasoning = String::new();
    let mut content = String::new();

    for delta in output {
        let result = parser.push(delta).unwrap();
        if let Some(next) = result.reasoning {
            reasoning.push_str(&next);
        }
        if let Some(next) = result.content {
            content.push_str(&next);
        }
    }

    let final_delta = parser.finish().unwrap();
    if let Some(next) = final_delta.reasoning {
        reasoning.push_str(&next);
    }
    if let Some(next) = final_delta.content {
        content.push_str(&next);
    }

    (
        (!reasoning.is_empty()).then_some(reasoning),
        (!content.is_empty()).then_some(content),
    )
}

#[test]
fn delimited_content_only_stream() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    assert_eq!(
        parser.push("plain content").content.as_deref(),
        Some("plain content")
    );
}

#[test]
fn delimited_single_chunk_with_reasoning_and_content() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    let delta = parser.push("<think>reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn delimited_partial_tokens_across_chunks() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();

    assert!(parser.push("<thi").is_empty());
    let delta = parser.push("nk>reason</think>answer");
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn delimited_finish_flushes_buffer() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser =
        DelimitedReasoningParser::new(tokenizer, "<think>", "</think>", false).unwrap();
    parser.initialize(&[THINK_START_ID]);

    let delta = parser.push("unfinished</thi");
    assert_eq!(delta.reasoning.as_deref(), Some("unfinished"));
    let final_delta = parser.finish();
    assert_eq!(final_delta.reasoning.as_deref(), Some("</thi"));
}

#[test]
fn qwen3_without_prompt_markers_expects_start_token() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("reason</think>answer"));
}

#[test]
fn qwen3_prompt_end_marker_starts_in_content() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[THINK_END_ID]).unwrap();

    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn qwen3_tolerates_old_and_new_formats() {
    let tokenizer = Arc::new(fake_tokenizer());

    let mut old_parser = Qwen3ReasoningParser::new(tokenizer.clone()).unwrap();
    let old = old_parser.push("<think>reason</think>answer").unwrap();
    assert_eq!(old.reasoning.as_deref(), Some("reason"));
    assert_eq!(old.content.as_deref(), Some("answer"));

    let mut new_parser = Qwen3ReasoningParser::new(tokenizer).unwrap();
    new_parser.initialize(&[THINK_START_ID]).unwrap();
    let new = new_parser.push("reason</think>answer").unwrap();
    assert_eq!(new.reasoning.as_deref(), Some("reason"));
    assert_eq!(new.content.as_deref(), Some("answer"));
}

#[test]
fn qwen3_stops_scanning_at_last_special_token() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = Qwen3ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[THINK_START_ID, SPECIAL_BOUNDARY_ID]).unwrap();

    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn deepseek_r1_defaults_to_reasoning_without_prompt_boundary() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DeepSeekR1ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn deepseek_r1_stops_scanning_at_last_special_token() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = DeepSeekR1ReasoningParser::new(tokenizer).unwrap();

    parser.initialize(&[THINK_END_ID, SPECIAL_BOUNDARY_ID]).unwrap();

    let delta = parser.push("reason</think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn minimax_m3_handles_explicit_think_delimiters() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("<mm:think>reason</mm:think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn minimax_m3_drops_leading_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("</mm:think>answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn minimax_m3_preserves_non_leading_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    let delta = parser.push("XXX</mm:think>YYY").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("XXX</mm:think>YYY"));
}

#[test]
fn minimax_m3_drops_split_leading_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();

    assert!(parser.push("</mm").unwrap().is_empty());
    let delta = parser.push(":think>answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn minimax_m3_uses_prompt_prefilled_start_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[MM_THINK_START_ID]).unwrap();

    let delta = parser.push("reason</mm:think>answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn minimax_m3_uses_prompt_prefilled_end_marker() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MiniMaxM3ReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[MM_THINK_END_ID]).unwrap();

    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn mistral_streaming_handles_think_delimited_outputs() {
    let cases = [
        (
            "no_think_token",
            vec!["implicit reasoning[/THINK]answer"],
            None,
            Some("implicit reasoning[/THINK]answer"),
        ),
        (
            "reasoning_and_content",
            vec!["[THINK]reason[/THINK]answer"],
            Some("reason"),
            Some("answer"),
        ),
        (
            "streamed_across_deltas",
            vec![
                "[THINK]",
                "Some ",
                "reasoning ",
                "content",
                "[/THINK]",
                "Final ",
                "answer",
            ],
            Some("Some reasoning content"),
            Some("Final answer"),
        ),
        (
            "no_end_delimiter",
            vec!["[THINK]reason without end"],
            Some("reason without end"),
            None,
        ),
        ("empty", vec![""], None, None),
        (
            "empty_reasoning",
            vec!["[THINK][/THINK]answer"],
            None,
            Some("answer"),
        ),
        (
            "content_around_reasoning",
            vec!["pre[THINK]reason[/THINK]post"],
            Some("reason"),
            Some("prepost"),
        ),
    ];

    for (name, output, expected_reasoning, expected_content) in cases {
        let mut parser = MistralReasoningParser::new(Arc::new(fake_tokenizer())).unwrap();
        let (reasoning, content) = run_streaming(&mut parser, &output);
        assert_eq!(reasoning.as_deref(), expected_reasoning, "{name}");
        assert_eq!(content.as_deref(), expected_content, "{name}");
    }
}

#[test]
fn mistral_picks_up_prompt_start_boundary() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
    // Prompt prefills `[THINK]`, opening reasoning before the stream.
    parser.initialize(&[MISTRAL_THINK_START_ID]).unwrap();

    let delta = parser.push("reason[/THINK]answer").unwrap();
    assert_eq!(delta.reasoning.as_deref(), Some("reason"));
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn mistral_respects_prompt_end_boundary() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
    // Prompt already closed reasoning with `[/THINK]`.
    parser.initialize(&[MISTRAL_THINK_END_ID]).unwrap();

    let delta = parser.push("answer").unwrap();
    assert_eq!(delta.reasoning, None);
    assert_eq!(delta.content.as_deref(), Some("answer"));
}

#[test]
fn mistral_handles_partial_delimiters_across_pushes() {
    let tokenizer = Arc::new(fake_tokenizer());
    let mut parser = MistralReasoningParser::new(tokenizer).unwrap();
    parser.initialize(&[MISTRAL_THINK_START_ID]).unwrap();

    // Closing delimiter `[/THINK]` arrives in two halves.
    let first = parser.push("reason[/TH").unwrap();
    assert_eq!(first.reasoning.as_deref(), Some("reason"));
    assert_eq!(first.content, None);

    let second = parser.push("INK]answer").unwrap();
    assert_eq!(second.reasoning, None);
    assert_eq!(second.content.as_deref(), Some("answer"));
}

#[test]
fn mistral_preserves_special_tokens_for_decoding() {
    let tokenizer = Arc::new(fake_tokenizer());
    let parser = MistralReasoningParser::new(tokenizer).unwrap();
    assert!(parser.preserve_special_tokens());
}
