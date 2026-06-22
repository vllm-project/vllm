use vllm_parser::tool::{Result, ToolParserOutput};

use super::{ToolParser, ToolParserFactory, names};
use crate::Error;
use crate::request::ChatTool;

struct FakeToolParser;

impl ToolParser for FakeToolParser {
    fn create(_tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, _chunk: &str, _output: &mut ToolParserOutput) -> Result<()> {
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        Ok(ToolParserOutput::default())
    }

    fn reset(&mut self) -> String {
        String::new()
    }
}

#[test]
fn default_factory_starts_empty() {
    let factory = ToolParserFactory::default();
    assert!(factory.list().is_empty());
}

#[test]
fn factory_contains_and_creates_registered_parsers() {
    let mut factory = ToolParserFactory::default();
    factory.register_parser::<FakeToolParser>("fake");

    assert!(factory.contains("fake"));
    assert!(factory.list().contains(&"fake".to_string()));
    factory.create("fake", &[]).unwrap();
}

#[test]
fn factory_rejects_unknown_parser_names() {
    let factory = ToolParserFactory::default();
    let error = match factory.create("missing", &[]) {
        Ok(_) => panic!("expected parser lookup to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::ParserUnavailableByName { .. }));
}

#[test]
fn factory_rejects_unknown_models() {
    let factory = ToolParserFactory::default();
    let error = match factory.create_for_model("definitely-unknown-model", &[]) {
        Ok(_) => panic!("expected model lookup to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::ParserUnavailableForModel { .. }));
}

#[test]
fn factory_creates_registered_parser_for_model() {
    let mut factory = ToolParserFactory::default();
    factory
        .register_parser::<FakeToolParser>("fake")
        .register_pattern("fake-model", "fake");

    factory.create_for_model("my-fake-model-v1", &[]).unwrap();
}

#[test]
fn factory_new_resolves_default_patterns() {
    let factory = ToolParserFactory::new();

    assert_eq!(
        factory.resolve_name_for_model("Qwen/Qwen3.5-0.8B"),
        Some(names::QWEN3_CODER)
    );
    assert_eq!(
        factory.resolve_name_for_model("Qwen/Qwen3-0.6B"),
        Some(names::QWEN3_XML)
    );
    assert_eq!(
        factory.resolve_name_for_model("Qwen/Qwen3-Coder-30B"),
        Some(names::QWEN3_CODER)
    );
    assert_eq!(
        factory.resolve_name_for_model("meta-llama-4-maverick"),
        Some(names::LLAMA4_JSON)
    );
    assert_eq!(
        factory.resolve_name_for_model("meta-llama-3.2-3b-instruct"),
        Some(names::LLAMA3_JSON)
    );
    assert_eq!(
        factory.resolve_name_for_model("meta-llama/Llama-3.1-8B-Instruct"),
        Some(names::LLAMA3_JSON)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-V4"),
        Some(names::DEEPSEEK_V4)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-V3.2-Exp"),
        Some(names::DEEPSEEK_V32)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-V4-Chat"),
        Some(names::DEEPSEEK_V4)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek_v4"),
        Some(names::DEEPSEEK_V4)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-R1-0528"),
        Some(names::DEEPSEEK_V3)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-V3.1"),
        Some(names::DEEPSEEK_V31)
    );
    assert_eq!(
        factory.resolve_name_for_model("zai-org/GLM-5-32B-Chat"),
        Some(names::GLM47)
    );
    assert_eq!(
        factory.resolve_name_for_model("zai-org/GLM-5.1-32B-Instruct"),
        Some(names::GLM47)
    );
    assert_eq!(
        factory.resolve_name_for_model("glm-4.7"),
        Some(names::GLM47)
    );
    assert_eq!(
        factory.resolve_name_for_model("google/gemma-4-27b-it"),
        Some(names::GEMMA4)
    );
    assert_eq!(
        factory.resolve_name_for_model("ibm-granite/granite-4.0-h-tiny"),
        Some(names::GRANITE4)
    );
    assert_eq!(
        factory.resolve_name_for_model("NousResearch/Hermes-3-Llama-3.1-8B"),
        Some(names::HERMES)
    );
    assert_eq!(
        factory.resolve_name_for_model("tencent/Hy3-preview"),
        Some(names::HY_V3)
    );
    assert_eq!(
        factory.resolve_name_for_model("MiniMax/MiniMax-M3-Text"),
        Some(names::MINIMAX_M3)
    );
    assert_eq!(
        factory.resolve_name_for_model("org/mm-m3-base"),
        Some(names::MINIMAX_M3)
    );
    assert_eq!(
        factory.resolve_name_for_model("MiniMax/MiniMax-M2-01"),
        Some(names::MINIMAX_M2)
    );
    assert_eq!(
        factory.resolve_name_for_model("org/mm-m2-base"),
        Some(names::MINIMAX_M2)
    );

    // InternLM2 positive: both dashed and underscored versioned names route.
    assert_eq!(
        factory.resolve_name_for_model("internlm/internlm2-chat-7b"),
        Some(names::INTERNLM)
    );
    assert_eq!(
        factory.resolve_name_for_model("internlm/internlm2_5-7b-chat"),
        Some(names::INTERNLM)
    );

    // Negative: other internlm-org models do NOT route to the InternLM2 parser,
    // since they use unrelated prompt formats.
    //   - InternLM v1 (`internlm-chat-7b`) routes to Llama
    //   - InternLM3 (`internlm3-8b-instruct`) routes to Llama
    //   - Intern-S1 / Intern-S1-Pro have their own parser (Python PR #40115)
    assert_eq!(
        factory.resolve_name_for_model("internlm/internlm-chat-7b"),
        None
    );
    assert_eq!(
        factory.resolve_name_for_model("internlm/internlm3-8b-instruct"),
        None
    );
    assert_eq!(factory.resolve_name_for_model("internlm/Intern-S1"), None);
    assert_eq!(
        factory.resolve_name_for_model("internlm/Intern-S1-Pro"),
        None
    );
}

#[test]
fn factory_new_registers_phi4_mini_json_by_name() {
    // phi-4-mini is registered by explicit name only (matching Python's
    // `--tool-call-parser phi4_mini_json`); it is intentionally not mapped to
    // any model-name pattern.
    let factory = ToolParserFactory::new();

    assert!(factory.contains(names::PHI4_MINI_JSON));
    factory.create(names::PHI4_MINI_JSON, &[]).unwrap();
}

#[test]
fn factory_new_registers_step3_by_name() {
    // Keep step3 exact-name only until the separate step3p5 parser exists, so
    // substring matching does not accidentally route step3p5 models here.
    let factory = ToolParserFactory::new();

    assert!(factory.contains(names::STEP3));
    assert_eq!(factory.resolve_name_for_model("stepfun-ai/step3"), None);
    assert_eq!(factory.resolve_name_for_model("stepfun-ai/step3p5"), None);
    factory.create(names::STEP3, &[]).unwrap();
}
