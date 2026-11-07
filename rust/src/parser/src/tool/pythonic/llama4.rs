// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{LLAMA4_PYTHONIC_CONFIG, PythonicToolParser};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

/// Tool parser for Llama 4 pythonic tool calls.
///
/// Example tool call content:
///
/// ```text
/// <|python_start|>[get_weather(city='LA', metric='C')]<|python_end|>
/// ```
///
/// Llama 4 emits the same Python list of keyword-argument calls as
/// [`PythonicToolParser`], sometimes wrapped in `<|python_start|>` /
/// `<|python_end|>`. Only those two markers differ, so this delegates to a
/// [`PythonicToolParser`] configured to drop them, mirroring how the Python
/// `Llama4PythonicToolParser` strips them before parsing.
///
/// The markers are tokenizer special tokens, so they are already gone under the
/// production default `skip_special_tokens = true` and the list starts at `[`;
/// the trait default `preserve_special_tokens() == false` is therefore correct
/// and stripping them is only needed when they do survive decoding.
pub struct Llama4PythonicToolParser {
    inner: PythonicToolParser,
}

impl Llama4PythonicToolParser {
    /// Create a Llama 4 pythonic tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: PythonicToolParser::with_config(LLAMA4_PYTHONIC_CONFIG),
        }
    }
}

impl ToolParser for Llama4PythonicToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::Llama4PythonicToolParser;
    use crate::tool::pythonic::{LLAMA4_PYTHONIC_CONFIG, parse_chunkings};
    use crate::tool::test_utils::test_tools;
    use crate::tool::{ToolParser, ToolParserTestExt as _};

    const SIMPLE_CALL: &str = "[get_weather(city='LA', metric='C')]";

    fn parse(text: &str) -> crate::tool::ToolParserOutput {
        parse_chunkings(LLAMA4_PYTHONIC_CONFIG, text)
    }

    #[test]
    fn llama4_pythonic_strips_python_markers() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"LA\",\"metric\":\"C\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&format!(
            "<|python_start|>{SIMPLE_CALL}<|python_end|>"
        )));
    }

    #[test]
    fn llama4_pythonic_parses_bare_call_list() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"LA\",\"metric\":\"C\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(SIMPLE_CALL));
    }

    #[test]
    fn llama4_pythonic_keeps_plain_text() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "How can I help you today?",
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse("How can I help you today?"));
    }

    #[test]
    fn llama4_pythonic_creates_through_tool_parser_trait() {
        let mut parser = Llama4PythonicToolParser::create(&test_tools()).unwrap();
        let output = parser
            .parse_complete("<|python_start|>[register_user(name='Doe', age=9)]<|python_end|>")
            .unwrap();

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "register_user",
                            ),
                            arguments: "{\"name\":\"Doe\",\"age\":9}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }
}
