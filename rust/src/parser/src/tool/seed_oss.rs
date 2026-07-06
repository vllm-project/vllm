use super::qwen_coder::{Qwen3CoderToolParser, QwenCoderConfig};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

const SEED_OSS_CONFIG: QwenCoderConfig = QwenCoderConfig {
    parser_name: "Seed-OSS",
    tool_call_start: "<seed:tool_call>",
    tool_call_end: "</seed:tool_call>",
};

/// Tool parser for Seed-OSS XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <seed:tool_call>
/// <function=get_weather>
/// <parameter=location>杭州</parameter>
/// </function>
/// </seed:tool_call>
/// ```
///
/// Seed-OSS shares the Qwen3 Coder grammar exactly; only the two tool-call
/// wrapper tokens differ (`<seed:tool_call>` / `</seed:tool_call>`). The inner
/// `<function=...>` / `<parameter=...>` grammar and schema-driven argument
/// conversion are byte-identical, so this delegates to a
/// [`Qwen3CoderToolParser`] configured with the Seed-OSS markers. This mirrors
/// Python `SeedOssParser(Qwen3Parser)` in `vllm/parser/seed_oss.py`.
///
/// Structured-output tags are intentionally unsupported: the trait default
/// `structural_tag_model() -> None` matches Python `SeedOssEngineToolParser`,
/// which sets `structural_tag_model = None` (xgrammar has no Seed-OSS model).
///
/// The `<seed:tool_call>` wrapper tokens are added-vocabulary tokens
/// (`special = false`), not tokenizer special tokens, so they survive decoding
/// under the production default `skip_special_tokens = true`; the trait default
/// `preserve_special_tokens() == false` is therefore correct, matching the
/// sibling `SeedOssReasoningParser`, which relies on the same for `<seed:think>`.
pub struct SeedOssToolParser {
    inner: Qwen3CoderToolParser,
}

impl SeedOssToolParser {
    /// Create a Seed-OSS tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self {
            inner: Qwen3CoderToolParser::with_config(tools, SEED_OSS_CONFIG),
        }
    }
}

impl ToolParser for SeedOssToolParser {
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
    use serde_json::{Value, json};
    use thiserror_ext::AsReport;

    use super::SeedOssToolParser;
    use crate::tool::test_utils::{collect_stream, split_by_chars, test_tools};
    use crate::tool::{ToolParser, ToolParserOutput, ToolParserTestExt as _};

    fn build_tool_call(function_name: &str, params: &[(&str, &str)]) -> String {
        let params = params
            .iter()
            .map(|(name, value)| format!("<parameter={name}>{value}</parameter>"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "<seed:tool_call>\n<function={function_name}>\n{params}\n</function>\n</seed:tool_call>"
        )
    }

    #[test]
    fn seed_oss_parse_complete_without_tool_call_keeps_text() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser.parse_complete("Hello, world!").unwrap();

        assert_eq!(output.normal_text(), "Hello, world!");
        assert!(output.calls().is_empty());
    }

    #[test]
    fn seed_oss_does_not_treat_plain_tool_call_marker_as_start() {
        // The Qwen Coder wrapper `<tool_call>` must NOT trigger the Seed-OSS
        // parser; only `<seed:tool_call>` does.
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser
            .parse_complete("<tool_call>\n<function=get_weather>\n</function>\n</tool_call>")
            .unwrap();

        assert_eq!(
            output.normal_text(),
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>"
        );
        assert!(output.calls().is_empty());
    }

    #[test]
    fn seed_oss_parse_complete_extracts_single_tool_call() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2026-04-29")],
            ))
            .unwrap();

        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF", "date": "2026-04-29" })
        );
    }

    #[test]
    fn seed_oss_parse_complete_preserves_prefix_text() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let input = format!(
            "Thinking... {}",
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let output = parser.parse_complete(&input).unwrap();

        assert_eq!(output.normal_text(), "Thinking... ");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn seed_oss_parse_complete_converts_schema_types() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "convert",
                &[
                    ("whole", "5.0"),
                    ("flag", "true"),
                    ("payload", r#"{"nested":true}"#),
                    ("items", "[1,2]"),
                    ("empty", "42"),
                ],
            ))
            .unwrap();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({
                "whole": 5.0,
                "flag": true,
                "payload": { "nested": true },
                "items": [1, 2],
                "empty": "42",
            })
        );
    }

    #[test]
    fn seed_oss_streaming_extracts_multiple_tool_calls_in_order() {
        let text = format!(
            "{}\n{}",
            build_tool_call("get_weather", &[("location", "SF")]),
            build_tool_call("get_weather", &[("location", "NYC")])
        );
        let chunks = split_by_chars(&text, 7);
        let mut parser = SeedOssToolParser::new(&test_tools());

        let output = collect_stream(&mut parser, &chunks);

        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[1].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn seed_oss_streaming_handles_end_marker_split_across_chunks() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        let mut output = ToolParserOutput::default();
        output.append(
            parser
                .parse_chunk(
                    "<seed:tool_call>\n\
                     <function=get_weather>\n\
                     <parameter=location>SF</parameter>\n\
                     </function>\n\
                     </seed:tool",
                )
                .unwrap(),
        );

        assert!(output.normal_text().is_empty());
        assert!(output.calls().is_empty());

        output.append(parser.parse_chunk("_call>").unwrap());
        output.append(parser.finish().unwrap());
        let output = output.coalesce();

        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn seed_oss_finish_fails_incomplete_tool_call() {
        let mut parser = SeedOssToolParser::new(&test_tools());
        parser
            .parse_chunk(
                "<seed:tool_call>\n<function=get_weather>\n<parameter=location>SF</parameter>",
            )
            .unwrap();

        let error = parser.finish().unwrap_err();

        assert_eq!(
            error.to_report_string(),
            "tool parser parsing failed: incomplete Seed-OSS tool call"
        );
    }
}
