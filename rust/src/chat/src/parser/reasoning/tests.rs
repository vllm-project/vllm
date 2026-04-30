use std::sync::Arc;

use vllm_text::tokenizer::Tokenizer;

use super::{
    DeepSeekR1ReasoningParser, DelimitedReasoningParser, Qwen3ReasoningParser, ReasoningParser,
    ReasoningParserFactory, names,
};

struct FakeTokenizer;

impl Tokenizer for FakeTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> vllm_text::Result<Vec<u32>> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> vllm_text::Result<String> {
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
fn factory_contains_and_lists_registered_parsers() {
    let factory = ReasoningParserFactory::new();
    assert!(factory.contains(names::QWEN3));
    assert!(factory.contains(names::DEEPSEEK_V4));
    assert!(factory.list().contains(&names::QWEN3.to_string()));
    assert!(factory.list().contains(&names::DEEPSEEK_V4.to_string()));
}

#[test]
fn factory_resolves_deepseek_v4_to_qwen3_alias() {
    let factory = ReasoningParserFactory::new();
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-V4"),
        Some(names::DEEPSEEK_V4)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek_v4"),
        Some(names::DEEPSEEK_V4)
    );
}

#[test]
fn factory_rejects_unknown_parser_names() {
    let tokenizer = Arc::new(FakeTokenizer);
    let factory = ReasoningParserFactory::new();
    let error = match factory.create("missing", tokenizer) {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("choose from"));
}
