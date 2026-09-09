// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{OLMO3_CONFIG, PythonicToolParser};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

/// Tool parser for OLMo 3 tool calls.
///
/// Example tool call content:
///
/// ```text
/// <function_calls>
/// get_weather(city='San Francisco', metric='celsius')
/// get_weather(city='New York', metric='celsius')
/// </function_calls>
/// ```
///
/// OLMo 3 emits the same keyword-argument calls as [`PythonicToolParser`], but
/// parallel calls are newline-separated instead of items of a Python list, and
/// the calls are wrapped in `<function_calls>` / `</function_calls>`. JSON
/// `true` / `false` / `null` literals are accepted next to the Python ones,
/// which the shared value parser already does.
///
/// Newlines are ordinary whitespace in the shared grammar, so blank lines,
/// indentation and calls spanning several lines are all accepted. That is more
/// tolerant than the Python parser, which joins non-empty stripped lines with
/// `", "` before parsing them as one list and therefore breaks a call split
/// across lines.
///
/// Both markers are optional: text that opens with `name(` is parsed as calls
/// without the wrapper, like the non-streaming Python path, which only strips
/// the wrapper when it is there, and a block that is cut short by the end of
/// the stream keeps the calls parsed so far. Text that opens with neither is
/// plain text, and text that opens like a call but is not one is a parse error
/// the streaming layer recovers as content, both as in [`PythonicToolParser`].
///
/// The markers are ordinary text rather than special tokens, so the trait
/// default `preserve_special_tokens() == false` is correct.
pub struct Olmo3PythonicToolParser {
    inner: PythonicToolParser,
}

impl Olmo3PythonicToolParser {
    /// Create an OLMo 3 tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: PythonicToolParser::with_config(OLMO3_CONFIG),
        }
    }
}

impl ToolParser for Olmo3PythonicToolParser {
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
    use thiserror_ext::AsReport;

    use super::Olmo3PythonicToolParser;
    use crate::tool::pythonic::{OLMO3_CONFIG, parse_chunkings};
    use crate::tool::test_utils::test_tools;
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    // Fixtures shared with `tests/tool_parsers/test_olmo3_tool_parser.py`.
    const SIMPLE_CALL: &str = "get_weather(city='San Francisco', metric='celsius')";
    const MORE_TYPES_CALL: &str = "register_user(name='John Doe', \
         age=37, \
         address={'city': 'San Francisco', 'state': 'CA'}, \
         role=None, \
         passed_test=True, \
         aliases=['John', 'Johnny'])";
    const MORE_TYPES_CALL_JSON_LITERALS: &str = "register_user(name='John Doe', \
         age=37, \
         address={'city': 'San Francisco', 'state': 'CA'}, \
         role=null, \
         passed_test=true, \
         aliases=['John', 'Johnny'])";
    const ESCAPED_STRING_CALL: &str =
        r#"get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')"#;

    fn parse(text: &str) -> ToolParserOutput {
        parse_chunkings(OLMO3_CONFIG, text)
    }

    /// Wrap `calls` in the `<function_calls>` block OLMo 3 emits.
    fn block(calls: &str) -> String {
        format!("<function_calls>{calls}</function_calls>")
    }

    #[test]
    fn olmo3_extracts_simple_call() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&block(SIMPLE_CALL)));
    }

    #[test]
    fn olmo3_converts_python_literals_to_json() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "register_user",
                            ),
                            arguments: "{\"name\":\"John Doe\",\"age\":37,\"address\":{\"city\":\"San Francisco\",\"state\":\"CA\"},\"role\":null,\"passed_test\":true,\"aliases\":[\"John\",\"Johnny\"]}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(&block(MORE_TYPES_CALL)));
    }

    #[test]
    fn olmo3_accepts_json_literals() {
        // `null` / `true` parse into the same arguments as `None` / `True`.
        let json_literals = parse(&block(MORE_TYPES_CALL_JSON_LITERALS));

        assert_eq!(json_literals, parse(&block(MORE_TYPES_CALL)));
        expect![[
            r#"{"name":"John Doe","age":37,"address":{"city":"San Francisco","state":"CA"},"role":null,"passed_test":true,"aliases":["John","Johnny"]}"#
        ]]
        .assert_eq(&json_literals.calls()[0].arguments);
    }

    #[test]
    fn olmo3_extracts_calls_on_separate_lines() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "register_user",
                            ),
                            arguments: "{\"name\":\"John Doe\",\"age\":37,\"address\":{\"city\":\"San Francisco\",\"state\":\"CA\"},\"role\":null,\"passed_test\":true,\"aliases\":[\"John\",\"Johnny\"]}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(&block(&format!("{SIMPLE_CALL}\n{MORE_TYPES_CALL}"))));
    }

    #[test]
    fn olmo3_extracts_empty_arguments() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "do_something_cool",
                            ),
                            arguments: "{\"additional_data\":{}}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 2,
                            name: Some(
                                "do_something_cool",
                            ),
                            arguments: "{\"steps\":[]}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&block(
            "get_weather()\ndo_something_cool(additional_data={})\ndo_something_cool(steps=[])",
        )));
    }

    #[test]
    fn olmo3_decodes_escaped_strings() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"Martha's Vineyard\",\"metric\":\"\\\"cool units\\\"\"}",
                        },
                    ),
                ],
            }
        "#]].assert_debug_eq(&parse(&block(ESCAPED_STRING_CALL)));
    }

    #[test]
    fn olmo3_ignores_blank_lines_and_indentation() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\":1,\"y\":2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&format!(
            "<function_calls>\n\n  {SIMPLE_CALL}  \n\n  add(\n    x=1,\n    y=2,\n  )\n\n</function_calls>"
        )));
    }

    #[test]
    fn olmo3_parses_calls_without_the_wrapper() {
        // The non-streaming Python path only strips the wrapper when present.
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "add",
                            ),
                            arguments: "{\"x\":1,\"y\":2}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&format!("{SIMPLE_CALL}\nadd(x=1, y=2)")));
    }

    #[test]
    fn olmo3_closes_unterminated_block_at_the_end_of_the_stream() {
        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&format!("<function_calls>\n{SIMPLE_CALL}\n")));
    }

    #[test]
    fn olmo3_keeps_plain_text() {
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
    fn olmo3_keeps_text_before_the_block() {
        // Text before the block permanently disables tool parsing, like the
        // Python parser's `current_text.startswith("<")` check does.
        expect![[r#"
            ToolParserOutput {
                events: [
                    Text(
                        "Let me check. <function_calls>get_weather(city='San Francisco', metric='celsius')</function_calls>",
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&parse(&format!("Let me check. {}", block(SIMPLE_CALL))));
    }

    #[test]
    fn olmo3_keeps_text_that_only_looks_like_a_block() {
        for text in [
            "<function_calls></function_calls>",
            "<functions>get_weather()</functions>",
            "<function_calls>",
        ] {
            let output = parse(text);
            assert_eq!(output.normal_text(), text, "{text:?} should stay text");
            assert!(output.calls().is_empty(), "{text:?} should have no calls");
        }
    }

    #[test]
    fn olmo3_streams_name_and_argument_fragments() {
        let mut parser = Olmo3PythonicToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        parser.parse_into(&block(SIMPLE_CALL), &mut output).unwrap();
        output.append(parser.finish().unwrap());

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "{",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "\"city\":\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "San Francisco\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: ",\"metric\":\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "celsius\"",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: None,
                            arguments: "}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn olmo3_streams_calls_split_across_large_steps() {
        // Fixture from `test_streaming_tool_call_with_large_steps`.
        let mut parser = Olmo3PythonicToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();

        for chunk in [
            "<function_calls>get_weather(city='San",
            " Francisco', metric='celsius')\nget_weather()\ndo_something_cool(steps=[])\
             </function_calls>",
        ] {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"metric\":\"celsius\"}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 2,
                            name: Some(
                                "do_something_cool",
                            ),
                            arguments: "{\"steps\":[]}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output.coalesce());
    }

    #[test]
    fn olmo3_finish_fails_incomplete_call() {
        let mut parser = Olmo3PythonicToolParser::new(&test_tools());
        parser.parse_chunk("<function_calls>get_weather(city='San").unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: incomplete OLMo 3 tool call"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn olmo3_rejects_trailing_text_after_the_block() {
        let mut parser = Olmo3PythonicToolParser::new(&test_tools());
        parser.parse_chunk(&format!("{} and that is all", block(SIMPLE_CALL))).unwrap();

        let error = parser.finish().unwrap_err();

        expect!["tool parser parsing failed: trailing text after OLMo 3 tool calls"]
            .assert_eq(&error.to_report_string());
    }

    #[test]
    fn olmo3_creates_through_tool_parser_trait() {
        let mut parser = Olmo3PythonicToolParser::create(&test_tools()).unwrap();
        let output = parser
            .parse_complete("<function_calls>\nregister_user(name='Doe', age=9)\n</function_calls>")
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
