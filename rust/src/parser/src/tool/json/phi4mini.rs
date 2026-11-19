// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use serde_json::Value;
use vllm_tokenizer::{DecodedText, DynTokenizer};
use xgrammar_structural_tag::format::{Format, TagFormat};
use xgrammar_structural_tag::tool::function_parameters;

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::output_grammar::{self, OutputGrammarContext};
use crate::schema::JsonType;
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const PHI4MINI_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Phi4Mini",
    start_marker: "functools[",
    framed_start_marker: None,
    end_marker: "]",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    // Accept both key variants emitted by Phi-4 Mini tool-call templates.
    arguments_key: &["arguments", "parameters"],
};

/// Tool parser for phi-4-mini models.
///
/// Example tool-call content:
///
/// ```text
/// functools[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
/// ```
///
/// phi-4-mini emits an array of tool-call objects wrapped in a `functools[..]`
/// envelope. Each object names the function with `name` and carries its
/// arguments under `arguments` (preferred) or `parameters`. Arguments are
/// already OpenAI-style JSON text, so they are streamed as raw argument deltas
/// without schema conversion or JSON normalization.
pub struct Phi4MiniJsonToolParser {
    inner: JsonToolCallParser,
}

impl Phi4MiniJsonToolParser {
    /// Create a phi-4-mini tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(PHI4MINI_CONFIG),
        }
    }
}

impl ToolParser for Phi4MiniJsonToolParser {
    /// Create a boxed phi-4-mini tool parser.
    fn create(tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Feed one decoded text chunk through the phi-4-mini parser.
    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.inner.parse_into(chunk, output)
    }

    /// Build the `functools[..]` call list for a tool choice that obliges a call.
    ///
    /// `auto` builds no grammar.
    fn build_call_format(
        &self,
        ctx: &OutputGrammarContext<'_>,
    ) -> output_grammar::Result<Option<Format>> {
        let Some(tools) = output_grammar::normalize_tool_choice(ctx)? else {
            return Ok(None);
        };
        if !tools.choice.requires_tool_call() {
            return Ok(None);
        }

        let calls = tools
            .function_tools
            .iter()
            .map(|tool| {
                // The parser reads arguments only as an object, so free
                // arguments still have to be one.
                let arguments = match function_parameters(&tool.function) {
                    Value::Bool(true) => JsonType::Object.any_schema(),
                    schema => schema,
                };
                TagFormat::new(
                    format!(r#"{{"name": "{}", "arguments": "#, tool.function.name),
                    Format::json_schema(arguments),
                    "}",
                )
            })
            .collect();
        let single_call = tools.choice.is_forced() || !ctx.parallel_tool_calls;
        Ok(Some(Format::sequence(vec![
            Format::const_string("functools["),
            Format::tags_with_separator(calls, ", ", true, single_call),
            Format::const_string("]"),
        ])))
    }

    /// Flush any buffered partial state at end of stream.
    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish()
    }

    /// Clear parser state and return currently uncommitted buffered text.
    fn reset(&mut self) -> String {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;
    use xgrammar_structural_tag::ToolChoice;

    use super::Phi4MiniJsonToolParser;
    use crate::output_grammar::{OutputGrammarContext, ToolStrictLevel};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserTestExt as _};

    /// Build one phi-4-mini tool-call object: `{"name":..,"<args_key>":<args>}`.
    fn build_call(function_name: &str, args_key: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{function_name}","{args_key}":{arguments}}}"#)
    }

    /// Wrap tool-call objects in the `functools[..]` envelope.
    fn wrap(calls: &[String]) -> String {
        format!("functools[{}]", calls.join(","))
    }

    #[test]
    fn phi4mini_parse_complete_without_tool_call_keeps_text() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text(), "Hello, world!");
        assert!(result.calls().is_empty());
    }

    #[test]
    fn phi4mini_parse_complete_extracts_arguments_key() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo","days":"3"}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("get_weather", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].tool_index, 0);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls()[0].arguments, arguments);
    }

    #[test]
    fn phi4mini_parse_complete_falls_back_to_parameters_key() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo"}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("get_weather", "parameters", arguments)]))
            .unwrap();

        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls()[0].arguments, arguments);
    }

    #[test]
    fn phi4mini_forced_tool_choice_builds_the_call_list_grammar() {
        let tools = test_tools()[..2].to_vec();
        let parser = Phi4MiniJsonToolParser::new(&tools);
        let build = |tool_choice: ToolChoice, parallel_tool_calls| {
            parser
                .build_visible_format(&OutputGrammarContext {
                    tools: &tools,
                    tool_choice: &tool_choice,
                    tool_strict_level: ToolStrictLevel::Auto,
                    parallel_tool_calls,
                    answer: None,
                })
                .unwrap()
                .map(|format| serde_json::to_string_pretty(&format).unwrap())
        };

        // A choice that obliges a call holds the model to the `functools[..]` list.
        expect![[r#"
            {
              "type": "sequence",
              "elements": [
                {
                  "type": "const_string",
                  "value": "functools["
                },
                {
                  "type": "tags_with_separator",
                  "tags": [
                    {
                      "type": "tag",
                      "begin": "{\"name\": \"get_weather\", \"arguments\": ",
                      "content": {
                        "type": "json_schema",
                        "json_schema": {
                          "type": "object",
                          "additionalProperties": true
                        },
                        "style": "json",
                        "any_order": false,
                        "max_whitespace_cnt": null,
                        "excludes": []
                      },
                      "end": "}"
                    },
                    {
                      "type": "tag",
                      "begin": "{\"name\": \"add\", \"arguments\": ",
                      "content": {
                        "type": "json_schema",
                        "json_schema": {
                          "type": "object",
                          "additionalProperties": true
                        },
                        "style": "json",
                        "any_order": false,
                        "max_whitespace_cnt": null,
                        "excludes": []
                      },
                      "end": "}"
                    }
                  ],
                  "separator": ", ",
                  "at_least_one": true,
                  "stop_after_first": false
                },
                {
                  "type": "const_string",
                  "value": "]"
                }
              ]
            }"#]]
        .assert_eq(&build(ToolChoice::required(), true).unwrap());
        expect![[r#"
            {
              "type": "sequence",
              "elements": [
                {
                  "type": "const_string",
                  "value": "functools["
                },
                {
                  "type": "tags_with_separator",
                  "tags": [
                    {
                      "type": "tag",
                      "begin": "{\"name\": \"add\", \"arguments\": ",
                      "content": {
                        "type": "json_schema",
                        "json_schema": {
                          "type": "object",
                          "additionalProperties": true
                        },
                        "style": "json",
                        "any_order": false,
                        "max_whitespace_cnt": null,
                        "excludes": []
                      },
                      "end": "}"
                    }
                  ],
                  "separator": ", ",
                  "at_least_one": true,
                  "stop_after_first": true
                },
                {
                  "type": "const_string",
                  "value": "]"
                }
              ]
            }"#]]
        .assert_eq(&build(ToolChoice::function("add"), true).unwrap());
        assert_ne!(
            build(ToolChoice::required(), false),
            build(ToolChoice::required(), true)
        );

        // A non-strict tool's free arguments must still be an object, the only
        // shape the parser reads.
        let required: serde_json::Value =
            serde_json::from_str(&build(ToolChoice::required(), true).unwrap()).unwrap();
        assert_eq!(
            required["elements"][1]["tags"][0]["content"]["json_schema"]["type"],
            "object"
        );

        // `auto` builds no grammar, as before.
        assert!(build(ToolChoice::auto(), true).is_none());
        assert!(build(ToolChoice::auto(), false).is_none());
    }

    #[test]
    fn phi4mini_extracts_multiple_comma_delimited_calls() {
        let input = wrap(&[
            build_call("get_weather", "arguments", r#"{"location":"Shanghai"}"#),
            build_call("add", "arguments", r#"{"x":1,"y":2}"#),
        ]);
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());

        let result = parser.parse_complete(&input).unwrap();

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"location\":\"Shanghai\"}",
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
        .assert_debug_eq(&result);
    }

    /// The shared JSON core scans matched braces, so bracket-bearing argument
    /// values are forwarded intact.
    #[test]
    fn phi4mini_array_valued_arguments_are_not_truncated() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"items":[1,2],"flag":true}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("convert", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].arguments, arguments);
    }

    /// Preface text before a tool call is preserved as normal_text, consistent
    /// with the other JSON parsers in this crate.
    #[test]
    fn phi4mini_preserves_text_before_tool_call() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let input = format!(
            "Let me check.\n{}",
            wrap(&[build_call(
                "get_weather",
                "arguments",
                r#"{"location":"Tokyo"}"#
            )])
        );

        let result = parser.parse_complete(&input).unwrap();

        assert_eq!(result.normal_text(), "Let me check.\n");
        assert_eq!(result.calls().len(), 1);
    }

    #[test]
    fn phi4mini_does_not_validate_or_normalize_arguments() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments = r#"{"location":"Tokyo",}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("get_weather", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls()[0].arguments, arguments);
    }

    /// The bundled `tool_chat_template_phi4_mini.jinja` emits objects with
    /// whitespace after `:` and `,` (e.g. `{"name": "f", "arguments": {..}}`).
    /// Confirm the parser handles that real model format and preserves the
    /// inner argument spacing verbatim.
    #[test]
    fn phi4mini_accepts_real_model_whitespace_format() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let input = r#"functools[{"name": "get_weather", "arguments": {"location": "Tokyo"}}]"#;

        let result = parser.parse_complete(input).unwrap();

        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls()[0].arguments, r#"{"location": "Tokyo"}"#);
    }

    /// Argument deltas are streamed through the shared JSON core.
    #[test]
    fn phi4mini_streaming_emits_argument_deltas() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let chunks = [
            "preface functo",
            "ols[",
            r#"{"name":"get_weather","arguments":"#,
            r#"{"location":"#,
            r#""Beijing""#,
            r#"}"#,
            r#"}]"#,
            " suffix",
        ];

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text(), "preface  suffix");
        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls()[0].arguments, r#"{"location":"Beijing"}"#);
    }

    #[test]
    fn phi4mini_streaming_handles_split_markers() {
        let input = format!(
            "hello {}",
            wrap(&[build_call(
                "get_weather",
                "arguments",
                r#"{"location":"Tokyo"}"#
            )])
        );
        let chunks = split_by_chars(&input, 5);
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());

        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.normal_text(), "hello ");
        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].arguments, r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn phi4mini_finish_errors_on_truncated_tool_call() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let _ = parser
            .parse_chunk(r#"functools[{"name":"get_weather","arguments":{"location""#)
            .unwrap();
        let error = parser.finish().unwrap_err();

        assert!(
            error.to_report_string().contains("incomplete Phi4Mini tool call"),
            "finish() reports the truncated tool call as incomplete: {}",
            error.to_report_string(),
        );
    }

    #[test]
    fn phi4mini_preserve_special_tokens_is_false() {
        let parser = Phi4MiniJsonToolParser::new(&test_tools());
        assert!(!parser.preserve_special_tokens());
    }

    /// The brace-scanning core handles nested arrays and objects in arguments.
    #[test]
    fn phi4mini_parses_nested_arrays_and_objects() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let arguments =
            r#"{"array_field":["a","b","c"],"object_field":{"nested":"value"},"empty_object":{}}"#;
        let result = parser
            .parse_complete(&wrap(&[build_call("convert", "arguments", arguments)]))
            .unwrap();

        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("convert"));
        assert_eq!(result.calls()[0].arguments, arguments);
    }

    /// The chat template emits parallel calls as `},\n  {` (comma + newline +
    /// indent). Confirm the `Optional` marker whitespace and `,` delimiter
    /// parse the real multi-call layout.
    #[test]
    fn phi4mini_parses_parallel_calls_in_template_format() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let input = concat!(
            "functools[\n",
            "  {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}},\n",
            "  {\"name\": \"add\", \"arguments\": {\"x\": 1, \"y\": 2}}\n",
            "]"
        );

        let result = parser.parse_complete(input).unwrap();

        assert_eq!(result.calls().len(), 2);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls()[1].name.as_deref(), Some("add"));
    }

    /// The shared core requires an object after the start marker.
    #[test]
    fn phi4mini_empty_array_errors() {
        let mut parser = Phi4MiniJsonToolParser::new(&test_tools());
        let error = parser.parse_complete("functools[]").unwrap_err();

        assert!(
            error.to_report_string().contains("invalid Phi4Mini"),
            "empty functools[] should error: {}",
            error.to_report_string(),
        );
    }
}
