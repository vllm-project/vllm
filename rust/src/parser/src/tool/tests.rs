use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserEvent, ToolParserOutput};
use crate::tool::ToolParserTestExt as _;

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
fn tool_parser_output_coalesces_adjacent_text_events() {
    let mut output = ToolParserOutput::default();
    output.push_text("hello");
    output.push_text(" ");
    output.push_text("world");
    output.push_call(ToolCallDelta {
        tool_index: 0,
        name: Some("lookup".to_string()),
        arguments: "{}".to_string(),
    });
    output.push_text("!");

    assert_eq!(
        output.events,
        vec![
            ToolParserEvent::Text("hello world".to_string()),
            ToolParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("lookup".to_string()),
                arguments: "{}".to_string(),
            }),
            ToolParserEvent::Text("!".to_string()),
        ]
    );
}

#[test]
fn tool_parser_output_append_coalesces_adjacent_text_events() {
    let mut output = ToolParserOutput::default();
    output.push_text("hello");

    let mut other = ToolParserOutput::default();
    other.push_text(" ");
    other.push_text("world");
    output.append(other);

    let mut after_call = ToolParserOutput::default();
    after_call.push_call(ToolCallDelta {
        tool_index: 0,
        name: Some("lookup".to_string()),
        arguments: "{}".to_string(),
    });
    after_call.push_text("!");
    output.append(after_call);

    assert_eq!(
        output.events,
        vec![
            ToolParserEvent::Text("hello world".to_string()),
            ToolParserEvent::ToolCall(ToolCallDelta {
                tool_index: 0,
                name: Some("lookup".to_string()),
                arguments: "{}".to_string(),
            }),
            ToolParserEvent::Text("!".to_string()),
        ]
    );
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
            output.push_text("prefix ");
            for call in [
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
            ] {
                output.push_call(call);
            }
            Ok(())
        }

        fn finish(&mut self) -> Result<ToolParserOutput> {
            let mut output = ToolParserOutput::default();
            output.push_text("suffix");
            output.push_call(ToolCallDelta {
                tool_index: 0,
                name: None,
                arguments: "}".to_string(),
            });
            output.push_call(ToolCallDelta {
                tool_index: 1,
                name: None,
                arguments: "\"UTC\"}".to_string(),
            });
            Ok(output)
        }

        fn reset(&mut self) -> String {
            String::new()
        }
    }

    let mut parser = StreamingParser;
    let output = parser.parse_complete("ignored").unwrap();
    assert_eq!(output.normal_text(), "prefix suffix");
    assert_eq!(
        output.calls().into_iter().cloned().collect::<Vec<_>>(),
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
