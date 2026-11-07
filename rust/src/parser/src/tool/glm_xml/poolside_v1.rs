// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer};

use super::{GlmXmlToolParser, Separator};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

/// Tool parser for Poolside's XML-style tool calls.
///
/// Poolside shares GLM-4.7's call format but does not enable a tool grammar.
/// String arguments are emitted as they arrive. Other types wait for the
/// complete value so the schema can determine their JSON type.
pub struct PoolsideV1ToolParser(GlmXmlToolParser);

impl PoolsideV1ToolParser {
    fn new(tools: &[Tool]) -> Self {
        Self(GlmXmlToolParser::new(tools, Separator::Flexible).with_incremental_strings())
    }
}

impl ToolParser for PoolsideV1ToolParser {
    fn create(tools: &[Tool], _tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
        self.0.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.0.finish()
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use xgrammar_structural_tag::ToolChoice;

    use super::PoolsideV1ToolParser;
    use crate::output_grammar::{OutputGrammarContext, ToolStrictLevel};
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tokenizer, test_tools};
    use crate::tool::{Tool, ToolParser, ToolParserTestExt as _};

    #[test]
    fn poolside_preserves_tokens_without_a_tool_grammar() {
        let tools = test_tools();
        let parser = PoolsideV1ToolParser::create(&tools, test_tokenizer()).unwrap();

        assert!(parser.preserve_special_tokens());
        assert_eq!(parser.default_tool_strict_level(), ToolStrictLevel::Auto);
        assert!(parser.structural_tag_builder().is_none());
        for choice in [
            json!("auto"),
            json!("required"),
            json!({"type": "function", "function": {"name": "get_weather"}}),
        ] {
            let tool_choice: ToolChoice = serde_json::from_value(choice).unwrap();
            let format = parser
                .build_call_format(&OutputGrammarContext {
                    tools: &tools,
                    tool_choice: &tool_choice,
                    tool_strict_level: ToolStrictLevel::Auto,
                    parallel_tool_calls: true,
                    answer: None,
                })
                .unwrap();
            assert!(format.is_none());
        }
    }

    #[test]
    fn poolside_keeps_plain_text() {
        let mut parser = PoolsideV1ToolParser::new(&test_tools());
        let output = collect_stream(&mut parser, &["Hello, ", "Paris!", " <tool"]);

        assert_eq!(output.normal_text(), "Hello, Paris! <tool");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn poolside_complete_and_stream_preserve_typed_arguments() {
        let tools = [Tool {
            name: "write_file".into(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "missing": {"type": ["string", "null"]},
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "metadata": {"type": "object"},
                    "items": {"type": "array"}
                }
            }),
            strict: None,
            defer_loading: None,
        }];
        let text = concat!(
            "I will save the file.\n<tool_call>write_file",
            "<arg_key>content</arg_key><arg_value>  print(\"café\\n\")\n</arg_value>",
            "<arg_key>note</arg_key><arg_value> 1.10\n</arg_value>",
            "<arg_key>missing</arg_key><arg_value> null </arg_value>",
            "<arg_key>count</arg_key><arg_value> 2 </arg_value>",
            "<arg_key>enabled</arg_key><arg_value>true</arg_value>",
            "<arg_key>metadata</arg_key><arg_value>{\"path\":\"a.py\"}</arg_value>",
            "<arg_key>items</arg_key><arg_value>[1, 2]</arg_value>",
            "</tool_call></assistant>\n"
        );
        let complete = PoolsideV1ToolParser::new(&tools).parse_complete(text).unwrap();
        let streamed = collect_stream(
            &mut PoolsideV1ToolParser::new(&tools),
            &split_by_chars(text, 1),
        );

        assert_eq!(streamed, complete);
        assert_eq!(complete.normal_text(), "I will save the file.\n");
        assert_eq!(complete.calls().len(), 1);
        assert_eq!(complete.calls()[0].name.as_deref(), Some("write_file"));
        assert_eq!(
            serde_json::from_str::<Value>(&complete.calls()[0].arguments).unwrap(),
            json!({
                "content": "  print(\"café\\n\")\n",
                "note": " 1.10\n",
                "missing": null,
                "count": 2,
                "enabled": true,
                "metadata": {"path": "a.py"},
                "items": [1, 2]
            })
        );
    }

    #[test]
    fn poolside_stream_handles_split_markers_and_multiple_calls() {
        let mut parser = PoolsideV1ToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "<tool_",
                "call>get_weather<arg_k",
                "ey>city</arg_key><arg_value>Paris</arg_val",
                "ue></tool_",
                "call>\n<tool",
                "_call>add\n</tool_call>",
            ],
        );

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "Paris"})
        );
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(output.calls()[1].name.as_deref(), Some("add"));
        assert_eq!(output.calls()[1].arguments, "{}");
        assert!(output.normal_text().is_empty());
    }

    #[test]
    fn poolside_emits_long_strings_before_the_value_ends() {
        let mut parser = PoolsideV1ToolParser::new(&test_tools());
        let mut output = parser
            .parse_chunk("<tool_call>get_weather<arg_key>city</arg_key><arg_value>")
            .unwrap();
        let value = "  café\n\"Paris\"\\</arg_valueX> ".repeat(200);
        let fragments = parser.parse_chunk(&value).unwrap();

        assert!(!fragments.calls().is_empty());
        assert!(fragments.calls().iter().all(|call| call.name.is_none()));
        let body = fragments.calls().iter().map(|call| call.arguments.as_str()).collect::<String>();
        let escaped = serde_json::to_string(&value).unwrap();
        assert_eq!(body, escaped[1..escaped.len() - 1]);
        output.append(fragments);
        // Even with no pending bytes, the call still needs its closing tags.
        assert!(parser.finish().is_err());
        assert!(parser.parse_chunk("</arg_val").unwrap().calls().is_empty());
        output.append(parser.parse_chunk("ue></tool_call>").unwrap());
        output.append(parser.finish().unwrap());

        let output = output.coalesce();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": value})
        );
    }

    #[test]
    fn poolside_buffers_nullable_and_unknown_values() {
        let tools = [Tool {
            name: "save".into(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "note": {"type": ["string", "null"]}
                }
            }),
            strict: None,
            defer_loading: None,
        }];
        let mut parser = PoolsideV1ToolParser::new(&tools);
        let mut output =
            parser.parse_chunk("<tool_call>save<arg_key>note</arg_key><arg_value>").unwrap();
        assert!(parser.parse_chunk(" null ").unwrap().calls().is_empty());
        output
            .append(parser.parse_chunk("</arg_value><arg_key>other</arg_key><arg_value>").unwrap());
        assert!(parser.parse_chunk("null").unwrap().calls().is_empty());
        output.append(parser.parse_chunk("</arg_value></tool_call>").unwrap());
        output.append(parser.finish().unwrap());

        assert_eq!(
            serde_json::from_str::<Value>(&output.coalesce().calls()[0].arguments).unwrap(),
            json!({"note": null, "other": null})
        );
    }

    #[test]
    fn poolside_emits_strings_from_additional_properties_early() {
        let tools = [Tool {
            name: "save".into(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "additionalProperties": {"type": "string"}
            }),
            strict: None,
            defer_loading: None,
        }];
        let mut parser = PoolsideV1ToolParser::new(&tools);
        let mut output =
            parser.parse_chunk("<tool_call>save<arg_key>note</arg_key><arg_value>").unwrap();
        let value = parser.parse_chunk(" null ").unwrap();
        assert_eq!(value.calls().len(), 1);
        assert_eq!(value.calls()[0].arguments, " null ");
        output.append(value);
        output.append(
            parser
                .parse_chunk(concat!(
                    "</arg_value><arg_key>count</arg_key><arg_value>2</arg_value>",
                    "</tool_call>"
                ))
                .unwrap(),
        );
        output.append(parser.finish().unwrap());
        assert_eq!(
            serde_json::from_str::<Value>(&output.coalesce().calls()[0].arguments).unwrap(),
            json!({"note": " null ", "count": 2})
        );
    }

    #[test]
    fn poolside_split_values_keep_empty_strings_and_duplicate_keys() {
        let text = concat!(
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value></arg_value>",
            "<arg_key>city</arg_key><arg_value>日\n\"Paris\"\\</arg_valueX></arg_value>",
            "</tool_call>"
        );
        let expected = PoolsideV1ToolParser::new(&test_tools()).parse_complete(text).unwrap();
        for chunk_size in [1, 7, 31] {
            let output = collect_stream(
                &mut PoolsideV1ToolParser::new(&test_tools()),
                &split_by_chars(text, chunk_size),
            );
            assert_eq!(output, expected);
        }
        assert!(expected.calls()[0].arguments.starts_with("{\"city\":\"\",\"city\":"));
        assert_eq!(
            serde_json::from_str::<Value>(&expected.calls()[0].arguments).unwrap(),
            json!({"city": "日\n\"Paris\"\\</arg_valueX>"})
        );
    }

    #[test]
    fn poolside_reset_clears_partial_calls_and_tool_indices() {
        let mut parser = PoolsideV1ToolParser::new(&test_tools());
        let first = parser.parse_chunk("<tool_call>add</tool_call>").unwrap();
        assert_eq!(first.calls()[0].tool_index, 0);

        let partial = parser.parse_chunk("<tool_call>get_weather").unwrap();
        assert!(partial.calls().is_empty());
        assert!(parser.finish().is_err());
        assert_eq!(parser.reset(), "get_weather");

        let output = parser.parse_complete("<tool_call>add</tool_call>").unwrap();
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[0].arguments, "{}");
    }
}
