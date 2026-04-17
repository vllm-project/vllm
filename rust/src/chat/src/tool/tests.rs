use super::{Result, ToolCallDelta, ToolParseResult, ToolParser, ToolParserFactory};
use crate::Error;
use crate::request::{ChatRequest, ChatTool};
use crate::tool::names;

struct FakeToolParser;

impl ToolParser for FakeToolParser {
    fn create(_tools: &[ChatTool]) -> super::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    fn adjust_request(&self, request: &mut ChatRequest) -> Result<()> {
        request.decode_options.skip_special_tokens = false;
        Ok(())
    }

    fn push(&mut self, _chunk: &str) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
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
fn factory_new_registers_builtin_json_parser() {
    let factory = ToolParserFactory::new();
    assert!(factory.contains(names::QWEN3_XML));
    assert!(factory.list().contains(&names::QWEN3_XML.to_string()));
}

#[test]
fn factory_new_resolves_external_default_patterns() {
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
        Some(names::LLAMA4_PYTHONIC)
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
        factory.resolve_name_for_model("command-r-plus"),
        Some(names::COHERE)
    );
}

#[test]
fn default_parse_complete_delegates_through_push_and_finish() {
    struct StreamingParser;

    impl ToolParser for StreamingParser {
        fn create(_tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn push(&mut self, _chunk: &str) -> Result<ToolParseResult> {
            Ok(ToolParseResult {
                normal_text: "prefix ".to_string(),
                calls: vec![
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some("weather".to_string()),
                        arguments: "{\"location\":".to_string(),
                    },
                    ToolCallDelta {
                        tool_index: 0,
                        name: None,
                        arguments: "\"Paris\"".to_string(),
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: Some("time".to_string()),
                        arguments: "{\"timezone\":".to_string(),
                    },
                ],
            })
        }

        fn finish(&mut self) -> Result<ToolParseResult> {
            Ok(ToolParseResult {
                normal_text: "suffix".to_string(),
                calls: vec![
                    ToolCallDelta {
                        tool_index: 0,
                        name: None,
                        arguments: "}".to_string(),
                    },
                    ToolCallDelta {
                        tool_index: 1,
                        name: None,
                        arguments: "\"UTC\"}".to_string(),
                    },
                ],
            })
        }
    }

    let mut parser = StreamingParser;
    let result = parser.parse_complete("ignored").unwrap();
    assert_eq!(result.normal_text, "prefix suffix");
    assert_eq!(
        result.calls,
        vec![
            ToolCallDelta {
                tool_index: 0,
                name: Some("weather".to_string()),
                arguments: "{\"location\":\"Paris\"}".to_string(),
            },
            ToolCallDelta {
                tool_index: 1,
                name: Some("time".to_string()),
                arguments: "{\"timezone\":\"UTC\"}".to_string(),
            },
        ]
    );
}
