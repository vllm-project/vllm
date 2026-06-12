// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use dynamo_parsers_v2::{
    DeepSeekV4ToolStreamParser, Tool as DynamoTool, ToolCallDelta as DynamoToolCallDelta,
    ToolParseResult as DynamoToolParseResult, ToolParser as DynamoToolParser,
};

use crate::{Result, Tool, ToolCallDelta, ToolParseResult, ToolParser};

/// Adapter for the Dynamo-owned DeepSeek V4 parser.
pub struct DynamoDeepSeekV4ToolParser {
    inner: Box<dyn DynamoToolParser>,
}

impl DynamoDeepSeekV4ToolParser {
    fn new(tools: &[Tool]) -> Result<Self> {
        let tools = tools.iter().map(to_dynamo_tool).collect::<Vec<_>>();
        let inner = DeepSeekV4ToolStreamParser::create(&tools).map_err(dynamo_error)?;
        Ok(Self { inner })
    }
}

impl ToolParser for DynamoDeepSeekV4ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)?))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.inner.push(chunk).map(to_vllm_result).map_err(dynamo_error)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        self.inner.finish().map(to_vllm_result).map_err(dynamo_error)
    }

    fn parse_complete(&mut self, output: &str) -> Result<ToolParseResult> {
        self.inner.parse_complete(output).map(to_vllm_result).map_err(dynamo_error)
    }
}

fn to_dynamo_tool(tool: &Tool) -> DynamoTool {
    DynamoTool {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
        strict: tool.strict,
    }
}

fn to_vllm_result(result: DynamoToolParseResult) -> ToolParseResult {
    ToolParseResult {
        normal_text: result.normal_text,
        calls: result.calls.into_iter().map(to_vllm_delta).collect(),
    }
}

fn to_vllm_delta(delta: DynamoToolCallDelta) -> ToolCallDelta {
    ToolCallDelta {
        tool_index: delta.tool_index,
        name: delta.name,
        arguments: delta.arguments,
    }
}

fn dynamo_error(error: anyhow::Error) -> crate::ToolParserError {
    parsing_failed!("Dynamo parser failed: {}", error)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{DynamoDeepSeekV4ToolParser, ToolParser};

    #[test]
    fn parse_complete_uses_dynamo_stream_parser() {
        let mut parser = DynamoDeepSeekV4ToolParser::new(&[]).unwrap();
        let result = parser
            .parse_complete(
                "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_weather\">\n\
<｜DSML｜parameter name=\"location\" string=\"true\">SF</｜DSML｜parameter>\n\
</｜DSML｜invoke>\n\
</｜DSML｜tool_calls>",
            )
            .unwrap();

        assert_eq!(result.normal_text, "");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "location": "SF" })
        );
    }

    #[test]
    fn parse_complete_recovers_complete_invoke_without_outer_close() {
        let mut parser = DynamoDeepSeekV4ToolParser::new(&[]).unwrap();
        let result = parser
            .parse_complete(
                "<｜DSML｜tool_calls>\n\
<｜DSML｜invoke name=\"get_datetime\">\n\
<｜DSML｜parameter name=\"timezone\" string=\"true\">Asia/Shanghai</｜DSML｜parameter>\n\
</｜DSML｜invoke>",
            )
            .unwrap();

        assert_eq!(result.normal_text, "");
        assert_eq!(result.calls.len(), 1);
        assert_eq!(result.calls[0].name.as_deref(), Some("get_datetime"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls[0].arguments).unwrap(),
            json!({ "timezone": "Asia/Shanghai" })
        );
    }
}
