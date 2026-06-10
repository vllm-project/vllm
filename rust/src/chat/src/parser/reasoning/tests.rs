use std::sync::Arc;

use vllm_tokenizer::Tokenizer;

use super::{ReasoningParserFactory, names};

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

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

#[test]
fn factory_contains_and_lists_registered_parsers() {
    let factory = ReasoningParserFactory::new();
    assert!(factory.contains(names::QWEN3));
    assert!(factory.contains(names::DEEPSEEK_V4));
    assert!(factory.contains(names::SEED_OSS));
    assert!(factory.contains(names::STEP3P5));
    assert!(factory.list().contains(&names::QWEN3.to_string()));
    assert!(factory.list().contains(&names::DEEPSEEK_V4.to_string()));
    assert!(factory.list().contains(&names::SEED_OSS.to_string()));
    assert!(factory.list().contains(&names::STEP3P5.to_string()));
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
fn factory_routes_step3p5_models_to_dedicated_parser() {
    let factory = ReasoningParserFactory::new();
    // step3p5 patterns must beat the bare `step3` substring.
    assert_eq!(
        factory.resolve_name_for_model("step-3p5-instruct"),
        Some(names::STEP3P5)
    );
    assert_eq!(
        factory.resolve_name_for_model("step3p5"),
        Some(names::STEP3P5)
    );
    assert_eq!(
        factory.resolve_name_for_model("step-3.5-base"),
        Some(names::STEP3P5)
    );
    assert_eq!(
        factory.resolve_name_for_model("step3-base"),
        Some(names::STEP3)
    );
}

#[test]
fn factory_routes_seed_oss_models() {
    let factory = ReasoningParserFactory::new();
    assert_eq!(
        factory.resolve_name_for_model("ByteDance-Seed/Seed-OSS-36B-Instruct"),
        Some(names::SEED_OSS)
    );
    assert_eq!(
        factory.resolve_name_for_model("seedoss-7b"),
        Some(names::SEED_OSS)
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
