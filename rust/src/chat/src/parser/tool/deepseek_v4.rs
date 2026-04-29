use super::deepseek_v32::{DeepSeekV32ToolParser, DsmlTokens};
use super::{Result, ToolParseResult, ToolParser};
use crate::request::ChatTool;

/// Tool parser for DeepSeek V4 models.
///
/// V4 reuses the V3.2 DSML invoke/parameter grammar but wraps calls in `<｜DSML｜tool_calls>`
/// instead of `<｜DSML｜function_calls>`.
pub struct DeepSeekV4ToolParser(DeepSeekV32ToolParser);

impl DsmlTokens {
    const V4: Self = Self {
        tool_calls_start: "<｜DSML｜tool_calls>",
        tool_calls_end: "</｜DSML｜tool_calls>",
    };
}

impl DeepSeekV4ToolParser {
    fn new(tools: &[ChatTool]) -> Self {
        Self(DeepSeekV32ToolParser::with_tokens(tools, DsmlTokens::V4))
    }
}

impl ToolParser for DeepSeekV4ToolParser {
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn adjust_request(&self, request: &mut crate::request::ChatRequest) -> Result<()> {
        self.0.adjust_request(request)
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.0.push(chunk)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        self.0.finish()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{DeepSeekV4ToolParser, ToolParser};
    use crate::request::ChatTool;

    fn test_tools() -> Vec<ChatTool> {
        vec![ChatTool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" },
                    "date": { "type": "string" }
                }
            }),
            strict: None,
        }]
    }

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

    fn collect_stream(chunks: &[&str], tools: &[ChatTool]) -> crate::parser::tool::ToolParseResult {
        let mut parser = DeepSeekV4ToolParser::new(tools);
        let mut result = crate::parser::tool::ToolParseResult::default();
        for chunk in chunks {
            result.append(parser.push(chunk).unwrap());
        }
        result.append(parser.finish().unwrap());
        result.coalesce_calls()
    }

    #[test]
    fn deepseek_v4_parse_complete_reuses_dsml_parser_with_tool_calls_token() {
        let mut parser = DeepSeekV4ToolParser::new(&test_tools());
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
    fn deepseek_v4_streaming_handles_tool_calls_token_split_across_chunks() {
        let result = collect_stream(
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
            &test_tools(),
        );

        assert_eq!(result.normal_text, "Thinking... ");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "Beijing" })
        );
    }
}
