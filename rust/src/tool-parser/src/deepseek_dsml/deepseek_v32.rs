use super::{DeepSeekDsmlToolParser, DsmlTokens};
use crate::{Result, Tool, ToolParseResult, ToolParser};

/// Tool parser for DeepSeek V3.2 models.
///
/// Example tool call content:
///
/// ```text
/// <｜DSML｜function_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜function_calls>
/// ```
///
/// Arguments are emitted only after a full `invoke` block is parsed.
///
/// DeepSeek V3.2 relies on DSML markers such as `｜DSML｜`, which are
/// represented as special tokens in the tokenizer and therefore must be
/// preserved during decode for parsing to work.
pub struct DeepSeekV32ToolParser(DeepSeekDsmlToolParser);

impl DeepSeekV32ToolParser {
    /// Create a DeepSeek V3.2 tool parser.
    pub(super) fn new(tools: &[Tool]) -> Self {
        Self(DeepSeekDsmlToolParser::new(tools, DsmlTokens::V32))
    }
}

impl ToolParser for DeepSeekV32ToolParser {
    /// Create a boxed DeepSeek V3.2 tool parser.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve DSML special tokens while decoding.
    fn preserve_special_tokens(&self) -> bool {
        true
    }

    /// Push one decoded text chunk through the DSML parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.0.push(chunk)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        self.0.finish()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::DeepSeekV32ToolParser;
    use crate::ToolParser;
    use crate::test_utils::{collect_stream, split_by_chars, test_tools};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| {
                format!(
                    r#"<｜DSML｜parameter name="{name}" string="true">{value}</｜DSML｜parameter>"#
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"{function_name}\">\n{params}\n</｜DSML｜invoke>\n</｜DSML｜function_calls>"
        )
    }

    #[test]
    fn deepseek_v32_parse_complete_without_tool_call_keeps_text() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v32_parse_complete_extracts_single_tool_call() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2024-01-16")],
            ))
            .unwrap();

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "location": "SF",
                "date": "2024-01-16"
            })
        );
    }

    #[test]
    fn deepseek_v32_parse_complete_preserves_prefix_text() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let output = format!(
            "Thinking... {}",
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let result = parser.parse_complete(&output).unwrap();

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_parse_complete_converts_schema_types() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                "<｜DSML｜function_calls>\n\
                 <｜DSML｜invoke name=\"convert\">\n\
                 <｜DSML｜parameter name=\"whole\" string=\"false\">5.0</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"flag\" string=\"false\">true</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"payload\" string=\"false\">{\"nested\":true}</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"items\" string=\"false\">[1,2]</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"empty\" string=\"false\">null</｜DSML｜parameter>\n\
                 </｜DSML｜invoke>\n\
                 </｜DSML｜function_calls>",
            )
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": null,
            })
        );
    }

    #[test]
    fn deepseek_v32_parse_complete_string_attr_overrides_schema_types() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(
                "<｜DSML｜function_calls>\n\
                 <｜DSML｜invoke name=\"convert\">\n\
                 <｜DSML｜parameter name=\"whole\" string=\"true\">5.0</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"flag\" string=\"true\">true</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"payload\" string=\"true\">{\"nested\":true}</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"items\" string=\"true\">[1,2]</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"empty\" string=\"true\">null</｜DSML｜parameter>\n\
                 </｜DSML｜invoke>\n\
                 </｜DSML｜function_calls>",
            )
            .unwrap();

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "whole": "5.0",
                "flag": "true",
                "payload": "{\"nested\":true}",
                "items": "[1,2]",
                "empty": "null",
            })
        );
    }

    #[test]
    fn deepseek_v32_parse_complete_unescapes_literal_closing_tags_in_parameter_value() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[
                    (
                        "location",
                        "Hangzhou &lt;/｜DSML｜parameter&gt;&lt;/｜DSML｜invoke&gt;&lt;/｜DSML｜function_calls&gt;",
                    ),
                    ("date", "2026-05-08"),
                ],
            ))
            .unwrap();

        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({
                "location": "Hangzhou </｜DSML｜parameter></｜DSML｜invoke></｜DSML｜function_calls>",
                "date": "2026-05-08",
            })
        );
    }

    #[test]
    fn deepseek_v32_streaming_extracts_single_tool_call() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_preserves_prefix_text() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "Thinking... ",
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
            ],
        );

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_streaming_without_tool_call_emits_text_incrementally() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &["Hello, ", "world!"]);

        assert_eq!(result.normal_text, "Hello, world!");
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v32_streaming_extracts_multiple_tool_calls_in_order() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[&format!(
                "{}\n{}",
                build_tool_call("get_weather", &[("location", "SF")])
                    .trim_end_matches("</｜DSML｜function_calls>"),
                "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>"
            )],
        );

        assert_eq!(result.calls.len(), 2);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[1].name.as_deref(), Some("get_weather"));
        assert_eq!(result.calls[0].tool_index, 0);
        assert_eq!(result.calls[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[1].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_handles_start_token_split_across_chunks() {
        let text = build_tool_call("get_weather", &[("location", "SF")]);
        let chunks = split_by_chars(&text, 5);
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(&mut parser, &chunks);

        assert_eq!(result.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_handles_bpe_chunked_dsml_opener() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜",
                "function",
                "_c",
                "all",
                "s",
                ">\n",
                "<｜DSML｜",
                "invoke",
                " name=\"",
                "get_weather",
                "\">\n",
                "<｜DSML｜",
                "parameter",
                " name=\"location\" string=\"true\">",
                "Beijing",
                "</｜DSML｜",
                "parameter>\n",
                "</｜DSML｜",
                "invoke>\n",
                "</｜DSML｜",
                "function_calls>",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "Beijing" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_truncated_parameter_does_not_leak_eos() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">Tokyo",
                "<｜end▁of▁sentence｜>",
            ],
        );

        assert!(result.calls.is_empty());
        assert!(!result.normal_text.contains("<｜end▁of▁sentence｜>"));
    }

    #[test]
    fn deepseek_v32_streaming_drops_eos_after_complete_tool_calls() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls><｜end▁of▁sentence｜>",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn deepseek_v32_streaming_ignores_text_after_complete_tool_calls() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜function_calls>",
                "trailing text",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert_eq!(result.calls.len(), 1);
    }

    #[test]
    fn deepseek_v32_streaming_does_not_emit_incomplete_invoke() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let result = collect_stream(
            &mut parser,
            &[
                "<｜DSML｜function_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n",
            ],
        );

        assert!(result.normal_text.is_empty());
        assert!(result.calls.is_empty());
    }

    #[test]
    fn deepseek_v32_parser_state_resets_after_finish() {
        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let first = parser
            .parse_complete(&build_tool_call("get_weather", &[("location", "SF")]))
            .unwrap();
        let second = parser
            .parse_complete(&build_tool_call("get_weather", &[("location", "NYC")]))
            .unwrap();

        assert_eq!(first.calls.len(), 1);
        assert_eq!(second.calls.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&second.calls[0].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn deepseek_v32_streaming_matches_parse_complete() {
        let full_text = build_tool_call("add", &[("x", "3"), ("y", "4")]);
        let chunks = split_by_chars(&full_text, 7);
        let mut streaming_parser = DeepSeekV32ToolParser::new(&test_tools());
        let streamed = collect_stream(&mut streaming_parser, &chunks);

        let mut parser = DeepSeekV32ToolParser::new(&test_tools());
        let complete = parser.parse_complete(&full_text).unwrap();

        assert_eq!(streamed.normal_text, complete.normal_text);
        assert_eq!(streamed.calls, complete.calls);
    }
}
