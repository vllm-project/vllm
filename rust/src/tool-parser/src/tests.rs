use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};
use crate::ToolParserTestExt as _;

struct DefaultParser;

impl ToolParser for DefaultParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self))
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
fn tool_parser_does_not_preserve_special_tokens_by_default() {
    let parser = DefaultParser;

    assert!(!parser.preserve_special_tokens());
}

#[test]
fn default_parse_complete_delegates_through_parse_chunk_and_finish() {
    struct StreamingParser;

    impl ToolParser for StreamingParser {
        fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>>
        where
            Self: Sized + 'static,
        {
            Ok(Box::new(Self))
        }

        fn parse_into(&mut self, _chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
            output.normal_text.push_str("prefix ");
            output.calls.extend([
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
            ]);
            Ok(())
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            Ok(ToolParserOutput {
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

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    let mut parser = StreamingParser;
    let output = parser.parse_complete("ignored").unwrap();
    assert_eq!(output.normal_text, "prefix suffix");
    assert_eq!(
        output.calls,
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
