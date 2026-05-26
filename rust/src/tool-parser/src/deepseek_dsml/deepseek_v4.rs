use super::{DeepSeekDsmlToolParser, DsmlTokens};
use crate::{Result, Tool, ToolParser, ToolParserOutput};

/// Tool parser for DeepSeek V4 models.
///
/// Example tool call content:
///
/// ```text
/// <｜DSML｜tool_calls>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// <｜DSML｜invoke name="get_weather">
/// <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
/// <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
/// </｜DSML｜invoke>
/// </｜DSML｜tool_calls>
/// ```
///
/// Arguments are emitted only after a full `invoke` block is parsed.
///
/// V4 reuses the V3.2 DSML invoke/parameter grammar but wraps calls in
/// `<｜DSML｜tool_calls>` instead of `<｜DSML｜function_calls>`.
///
/// DeepSeek V4 relies on DSML markers such as `｜DSML｜`, which are
/// represented as special tokens in the tokenizer and therefore must be
/// preserved during decode for parsing to work.
pub struct DeepSeekV4ToolParser(DeepSeekDsmlToolParser);

impl DeepSeekV4ToolParser {
    /// Create a DeepSeek V4 tool parser.
    fn new(tools: &[Tool]) -> Self {
        Self(DeepSeekDsmlToolParser::new(tools, DsmlTokens::V4))
    }
}

impl ToolParser for DeepSeekV4ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }
    fn preserve_special_tokens(&self) -> bool {
        true
    }
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
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

    use super::{DeepSeekV4ToolParser, ToolParser};
    use crate::test_utils::{collect_stream, test_tools};

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
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"{function_name}\">\n{params}\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        )
    }

    #[test]
    fn deepseek_v4_parse_complete_reuses_dsml_parser_with_tool_calls_token() {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools());
        let output = parser
            .parse_complete(&build_tool_call(
                "get_weather",
                &[("location", "SF"), ("date", "2024-01-16")],
            ))
            .unwrap();

        assert!(output.normal_text.is_empty());
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
            json!({
                "location": "SF",
                "date": "2024-01-16"
            })
        );
    }

    #[test]
    fn deepseek_v4_streaming_handles_tool_calls_token_split_across_chunks() {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools());
        let output = collect_stream(
            &mut parser,
            &[
                "Thinking... ",
                "<｜DSML｜",
                "tool",
                "_calls>\n",
                "<｜DSML｜invoke name=\"get_weather\">\n",
                "<｜DSML｜parameter name=\"location\" string=\"true\">Beijing</｜DSML｜parameter>\n",
                "</｜DSML｜invoke>\n",
                "</｜DSML｜",
                "tool_calls>",
            ],
        );

        assert_eq!(output.normal_text, "Thinking... ");
        assert_eq!(output.calls.len(), 1);
        assert_eq!(output.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls[0].arguments).unwrap(),
            json!({ "location": "Beijing" })
        );
    }
}
