use async_trait::async_trait;

use super::{
    Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserError, ToolParserFactory,
};
use crate::request::ChatTool;
use crate::tool::names;

struct FakeToolParser;

#[async_trait]
impl ToolParser for FakeToolParser {
    fn create(_tools: &[ChatTool]) -> super::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    async fn parse_complete(&self, _output: &str) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }

    async fn parse_incremental(&mut self, _chunk: &str) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallDelta>> {
        None
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
    assert!(matches!(error, ToolParserError::UnknownParser { .. }));
}

#[test]
fn factory_rejects_unknown_models() {
    let factory = ToolParserFactory::default();
    let error = match factory.create_for_model("definitely-unknown-model", &[]) {
        Ok(_) => panic!("expected model lookup to fail"),
        Err(error) => error,
    };
    assert!(matches!(error, ToolParserError::UnknownModel { .. }));
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
fn factory_new_registers_builtin_json_parser() {
    let factory = ToolParserFactory::new();
    assert!(factory.contains(names::QWEN3_XML));
    assert!(factory.list().contains(&names::QWEN3_XML.to_string()));
}

#[test]
fn factory_new_resolves_external_default_patterns() {
    let factory = ToolParserFactory::new();

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
        Some(names::LLAMA4_PYTHONIC)
    );
    assert_eq!(
        factory.resolve_name_for_model("meta-llama-3.2-3b-instruct"),
        Some(names::LLAMA3_JSON)
    );
    assert_eq!(
        factory.resolve_name_for_model("deepseek-ai/DeepSeek-V3.1"),
        Some(names::DEEPSEEK_V31)
    );
    assert_eq!(
        factory.resolve_name_for_model("glm-4.7"),
        Some(names::GLM47)
    );
    assert_eq!(
        factory.resolve_name_for_model("command-r-plus"),
        Some(names::COHERE)
    );
}
