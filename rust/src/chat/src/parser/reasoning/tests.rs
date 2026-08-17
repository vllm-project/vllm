// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use vllm_tokenizer::test_utils::TestTokenizer;

use super::{ReasoningParserFactory, names};

#[test]
fn factory_contains_and_lists_registered_parsers() {
    let factory = ReasoningParserFactory::new();
    assert!(factory.contains(names::QWEN3));
    assert!(factory.contains(names::DEEPSEEK_V4));
    assert!(factory.contains(names::SEED_OSS));
    assert!(factory.contains(names::STEP3P5));
    assert!(factory.contains(names::HUNYUAN_A13B));
    assert!(factory.contains(names::MINIMAX_M3));
    assert!(factory.contains(names::GEMMA4));
    assert!(factory.list().contains(&names::QWEN3.to_string()));
    assert!(factory.list().contains(&names::DEEPSEEK_V4.to_string()));
    assert!(factory.list().contains(&names::SEED_OSS.to_string()));
    assert!(factory.list().contains(&names::STEP3P5.to_string()));
    assert!(factory.list().contains(&names::HUNYUAN_A13B.to_string()));
    assert!(factory.list().contains(&names::MINIMAX_M3.to_string()));
    assert!(factory.list().contains(&names::GEMMA4.to_string()));
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
fn factory_resolves_minimax_m3_before_generic_minimax() {
    let factory = ReasoningParserFactory::new();
    assert_eq!(
        factory.resolve_name_for_model("MiniMaxAI/Minimax-M3-preview"),
        Some(names::MINIMAX_M3)
    );
    assert_eq!(
        factory.resolve_name_for_model("mm-m3"),
        Some(names::MINIMAX_M3)
    );
}

#[test]
fn factory_routes_hunyuan_a13b_models() {
    let factory = ReasoningParserFactory::new();
    // The pattern is deliberately narrow. `resolve_name_for_model` is a
    // case-insensitive substring scan, so a bare `hunyuan` would also claim
    // HunyuanOCR and the dense Hunyuan-7B, neither of which has a thinking mode.
    assert_eq!(
        factory.resolve_name_for_model("tencent/Hunyuan-A13B-Instruct"),
        Some(names::HUNYUAN_A13B)
    );
    assert_eq!(
        factory.resolve_name_for_model("hunyuan_a13b"),
        Some(names::HUNYUAN_A13B)
    );
    assert_eq!(factory.resolve_name_for_model("tencent/HunyuanOCR"), None);
    assert_eq!(
        factory.resolve_name_for_model("tencent/Hunyuan-7B-Instruct"),
        None
    );
}

#[test]
fn factory_creates_hunyuan_a13b_without_reasoning_tokens() {
    // Hunyuan's delimiters are not vocabulary tokens, so an empty vocabulary is
    // enough to construct this parser: it never resolves a delimiter ID.
    let tokenizer = Arc::new(TestTokenizer::new());
    let factory = ReasoningParserFactory::new();

    let mut parser = factory
        .create(names::HUNYUAN_A13B, tokenizer)
        .expect("hunyuan parser needs no reasoning delimiter tokens");

    expect_test::expect![[r#"
        ReasoningDelta {
            reasoning: Some(
                "reason",
            ),
            content: Some(
                "answer",
            ),
        }
    "#]]
    .assert_debug_eq(
        &parser.push("<think>\nreason\n</think>\n<answer>\nanswer\n</answer>").unwrap(),
    );
}

#[test]
fn factory_rejects_unknown_parser_names() {
    let tokenizer = Arc::new(TestTokenizer::new());
    let factory = ReasoningParserFactory::new();
    let error = match factory.create("missing", tokenizer) {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("choose from"));
}
