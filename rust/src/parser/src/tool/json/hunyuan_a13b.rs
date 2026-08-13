// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{JsonToolCallConfig, JsonToolCallParser, JsonToolCallWhitespace};
use crate::tool::{Result, Tool, ToolParser, ToolParserEvent, ToolParserOutput};

const ASSISTANT_PREFIX: &str = "助手：";

const HUNYUAN_A13B_CONFIG: JsonToolCallConfig = JsonToolCallConfig {
    parser_name: "Hunyuan A13B",
    start_marker: "<tool_calls>",
    end_marker: "</tool_calls>",
    marker_whitespace: JsonToolCallWhitespace::Optional,
    delimiter: Some(","),
    name_key: "name",
    arguments_key: &["arguments"],
};

/// Tool parser for Hunyuan A13B models.
///
/// Hunyuan A13B emits an array of OpenAI-style function calls wrapped in
/// `<tool_calls>` tags:
///
/// ```text
/// <tool_calls>[{"name":"get_weather","arguments":{"city":"Beijing"}}]</tool_calls>
/// ```
///
/// The shared JSON parser streams argument text without re-serializing it and
/// tracks JSON nesting structurally. This is important for Hunyuan outputs:
/// nested objects and argument properties named `name` must not be mistaken
/// for a second tool-call header.
pub struct HunyuanA13BToolParser {
    inner: JsonToolCallParser,
    pending_prefix: String,
    prefix_decided: bool,
}

impl HunyuanA13BToolParser {
    /// Create a Hunyuan A13B tool parser.
    fn new(_tools: &[Tool]) -> Self {
        Self {
            inner: JsonToolCallParser::new(HUNYUAN_A13B_CONFIG)
                .with_container("[", "]")
                .discard_after_end(),
            pending_prefix: String::new(),
            prefix_decided: false,
        }
    }

    /// Strip the chat-template prefix from a natural-language response while
    /// preserving ordinary leading text and arbitrary streaming boundaries.
    fn append_output(&mut self, parsed: ToolParserOutput, output: &mut ToolParserOutput) {
        for event in parsed.events {
            match event {
                ToolParserEvent::Text(text) if !self.prefix_decided => {
                    self.pending_prefix.push_str(&text);
                    if self.pending_prefix.starts_with(ASSISTANT_PREFIX) {
                        self.pending_prefix.drain(..ASSISTANT_PREFIX.len());
                        self.prefix_decided = true;
                        output.push_text(std::mem::take(&mut self.pending_prefix));
                    } else if !ASSISTANT_PREFIX.starts_with(&self.pending_prefix) {
                        self.prefix_decided = true;
                        output.push_text(std::mem::take(&mut self.pending_prefix));
                    }
                }
                ToolParserEvent::Text(text) => output.push_text(text),
                ToolParserEvent::ToolCall(call) => {
                    if !self.pending_prefix.is_empty() {
                        self.prefix_decided = true;
                        output.push_text(std::mem::take(&mut self.pending_prefix));
                    }
                    output.push_call(call);
                }
            }
        }
    }
}

impl ToolParser for HunyuanA13BToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        let mut parsed = ToolParserOutput::default();
        let result = self.inner.parse_into(chunk, &mut parsed);
        self.append_output(parsed, output);
        result
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let parsed = self.inner.finish()?;
        let mut output = ToolParserOutput::default();
        self.append_output(parsed, &mut output);
        if !self.pending_prefix.is_empty() {
            output.push_text(std::mem::take(&mut self.pending_prefix));
        }
        self.prefix_decided = false;
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let mut pending = std::mem::take(&mut self.pending_prefix);
        pending.push_str(&self.inner.reset());
        self.prefix_decided = false;
        pending
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use thiserror_ext::AsReport;

    use super::HunyuanA13BToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserTestExt as _};

    fn build_call(function_name: &str, arguments: &str) -> String {
        format!(r#"{{"name":"{function_name}","arguments":{arguments}}}"#)
    }

    fn wrap(calls: &[String]) -> String {
        format!("<tool_calls>[{}]</tool_calls>", calls.join(","))
    }

    #[test]
    fn hunyuan_a13b_parse_complete_without_tool_call_keeps_text() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let output = parser.parse_complete("How can I help you today?").unwrap();

        assert_eq!(output.normal_text(), "How can I help you today?");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hunyuan_a13b_strips_assistant_prefix_from_natural_language() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let chunks = ["助", "手", "：How can I help?"];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "How can I help?");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hunyuan_a13b_preserves_nonleading_assistant_prefix() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let input = "Quoted template text: 助手：How can I help?";

        let output = parser.parse_complete(input).unwrap();

        assert_eq!(output.normal_text(), input);
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hunyuan_a13b_extracts_parallel_calls_and_nested_arguments() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let input = wrap(&[
            build_call("get_weather", r#"{"city":"San Francisco","days":3}"#),
            build_call(
                "update_record",
                r#"{"data":{"profile":{"name":"John"},"aliases":["John","Johnny"]}}"#,
            ),
        ]);

        let output = parser.parse_complete(&input).unwrap();

        expect![[r#"
            ToolParserOutput {
                events: [
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 0,
                            name: Some(
                                "get_weather",
                            ),
                            arguments: "{\"city\":\"San Francisco\",\"days\":3}",
                        },
                    ),
                    ToolCall(
                        ToolCallDelta {
                            tool_index: 1,
                            name: Some(
                                "update_record",
                            ),
                            arguments: "{\"data\":{\"profile\":{\"name\":\"John\"},\"aliases\":[\"John\",\"Johnny\"]}}",
                        },
                    ),
                ],
            }
        "#]]
        .assert_debug_eq(&output);
    }

    #[test]
    fn hunyuan_a13b_preserves_prefix_text_and_non_ascii_arguments() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let input = format!(
            "I will call the tool now. {}",
            wrap(&[build_call("get_weather", r#"{"city":"北京"}"#)])
        );

        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.normal_text(), "I will call the tool now. ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, r#"{"city":"北京"}"#);
    }

    #[test]
    fn hunyuan_a13b_allows_whitespace_before_json_array() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let input = concat!(
            "<tool_calls>\n  [\n",
            "  {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n",
            "]\n</tool_calls>"
        );

        let chunks = split_by_chars(input, 4);
        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, r#"{"city": "Tokyo"}"#);
    }

    #[test]
    fn hunyuan_a13b_accepts_empty_tool_call_array() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());

        let output = parser.parse_complete("<tool_calls>[]</tool_calls>").unwrap();

        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hunyuan_a13b_streaming_accepts_empty_tool_call_array() {
        let input = "<tool_calls>[]</tool_calls>";
        let chunks = split_by_chars(input, 1);
        let mut parser = HunyuanA13BToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hunyuan_a13b_empty_array_preserves_prefix_and_discards_streaming_suffix() {
        let chunks = [
            "Before <tool_calls>[",
            "]</tool_calls>discarded",
            " across chunks",
        ];
        let mut parser = HunyuanA13BToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.normal_text(), "Before ");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn hunyuan_a13b_rejects_trailing_comma() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let input = format!(
            "<tool_calls>[{},]</tool_calls>",
            build_call("get_weather", r#"{"city":"Seattle"}"#)
        );

        assert!(parser.parse_complete(&input).is_err());
    }

    #[test]
    fn hunyuan_a13b_discards_text_after_tool_calls() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let input = format!(
            "{}\nThank you!",
            wrap(&[build_call("get_weather", r#"{"city":"Seattle"}"#)])
        );

        let output = parser.parse_complete(&input).unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn hunyuan_a13b_streaming_discards_text_after_tool_calls() {
        let tool_call = wrap(&[build_call("get_weather", r#"{"city":"Seattle"}"#)]);
        let chunks = [tool_call.as_str(), "\nThank", " you!"];
        let mut parser = HunyuanA13BToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn hunyuan_a13b_streaming_handles_arbitrary_chunk_boundaries() {
        let input = wrap(&[build_call(
            "get_weather",
            r#"{"city":"Boston","metadata":{"name":"forecast"}}"#,
        )]);
        let chunks = split_by_chars(&input, 5);
        let mut parser = HunyuanA13BToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            output.calls()[0].arguments,
            r#"{"city":"Boston","metadata":{"name":"forecast"}}"#
        );
    }

    #[test]
    fn hunyuan_a13b_streaming_does_not_treat_argument_name_as_another_call() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        let chunks = [
            r#"<tool_calls>[{"name":"update_record","arguments":{"name":"#,
            r#""display name","nested":{"name":"inner"}}}"#,
            "]</tool_calls>",
        ];

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("update_record"));
        assert_eq!(
            output.calls()[0].arguments,
            r#"{"name":"display name","nested":{"name":"inner"}}"#
        );
    }

    #[test]
    fn hunyuan_a13b_finish_errors_on_truncated_tool_call() {
        let mut parser = HunyuanA13BToolParser::new(&test_tools());
        parser
            .parse_chunk(r#"<tool_calls>[{"name":"get_weather","arguments":{"city""#)
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert_eq!(
            error.to_report_string(),
            "tool parser parsing failed: incomplete Hunyuan A13B tool call"
        );
    }
}
