use super::{Result, Tool, ToolCallDelta, ToolParseResult, ToolParser};

struct DefaultParser;

impl ToolParser for DefaultParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
    }

    fn push(&mut self, _chunk: &str) -> Result<ToolParseResult> {
        Ok(ToolParseResult::default())
    }
}

#[test]
fn tool_parser_does_not_preserve_special_tokens_by_default() {
    let parser = DefaultParser;

    assert!(!parser.preserve_special_tokens());
}

#[test]
fn default_parse_complete_delegates_through_push_and_finish() {
    struct StreamingParser;

    impl ToolParser for StreamingParser {
        fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
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
