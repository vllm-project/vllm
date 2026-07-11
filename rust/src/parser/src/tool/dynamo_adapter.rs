// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use dynamo_parsers_v2::{
    Tool as DynamoTool, ToolCallDelta as DynamoToolCallDelta,
    ToolParseResult as DynamoToolParseResult, ToolParser as DynamoToolParser,
    create_tool_parser_for_family,
};

use crate::tool::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

/// Shared adapter around any Dynamo-owned `dynamo-parsers-v2` stream parser.
///
/// The only per-family difference is the family name passed to the Dynamo
/// crate's `create_tool_parser_for_family` dispatch, so every family routes
/// through this one core instead of a copy-pasted adapter per parser.
struct DynamoStreamAdapter {
    family: &'static str,
    tools: Vec<DynamoTool>,
    inner: Box<dyn DynamoToolParser>,
}

impl DynamoStreamAdapter {
    fn new(family: &'static str, tools: &[Tool]) -> Result<Self> {
        let tools = tools.iter().map(to_dynamo_tool).collect::<Vec<_>>();
        let inner = create_tool_parser_for_family(family, &tools).map_err(dynamo_error)?;
        Ok(Self {
            family,
            tools,
            inner,
        })
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        output.append(self.inner.push(chunk).map(to_vllm_output).map_err(dynamo_error)?);
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.inner.finish().map(to_vllm_output).map_err(dynamo_error)
    }

    fn reset(&mut self) -> String {
        // Rebuild from the same family + tools so per-family state (e.g. the
        // Qwen3-Coder tool-schema typing) is restored. The family already
        // constructed successfully once, so this cannot fail on these tools.
        self.inner = create_tool_parser_for_family(self.family, &self.tools)
            .expect("Dynamo parser recreate on reset");
        String::new()
    }
}

/// Define a vLLM `ToolParser` that delegates to a Dynamo v2 family parser.
///
/// The family string is the only thing that varies between parsers, so it is
/// passed as data rather than duplicating the adapter body.
macro_rules! dynamo_family_parser {
    ($(#[$meta:meta])* $name:ident, $family:literal) => {
        $(#[$meta])*
        pub struct $name(DynamoStreamAdapter);

        impl $name {
            fn new(tools: &[Tool]) -> Result<Self> {
                Ok(Self(DynamoStreamAdapter::new($family, tools)?))
            }
        }

        impl ToolParser for $name {
            fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
            where
                Self: Sized + 'static,
            {
                Ok(Box::new(Self::new(tools)?))
            }

            fn preserve_special_tokens(&self) -> bool {
                self.0.preserve_special_tokens()
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
    };
}

dynamo_family_parser!(
    /// Adapter for the Dynamo-owned DeepSeek V4 parser.
    DynamoDeepSeekV4ToolParser,
    "deepseek_v4"
);

dynamo_family_parser!(
    /// Adapter for the Dynamo-owned Qwen3-Coder parser.
    DynamoQwen3CoderToolParser,
    "qwen3_coder"
);

dynamo_family_parser!(
    /// Adapter for the Dynamo-owned GLM-4.7 parser.
    DynamoGlm47ToolParser,
    "glm47"
);

dynamo_family_parser!(
    /// Adapter for the Dynamo-owned Kimi K2 parser.
    DynamoKimiK2ToolParser,
    "kimi_k2"
);

dynamo_family_parser!(
    /// Adapter for the Dynamo-owned MiniMax M2 parser.
    DynamoMiniMaxM2ToolParser,
    "minimax_m2"
);

dynamo_family_parser!(
    /// Adapter for the Dynamo-owned MiniMax M3 parser.
    DynamoMiniMaxM3ToolParser,
    "minimax_m3"
);

dynamo_family_parser!(
    /// Adapter for the Dynamo-owned Gemma 4 parser.
    DynamoGemma4ToolParser,
    "gemma4"
);

fn to_dynamo_tool(tool: &Tool) -> DynamoTool {
    DynamoTool {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
        strict: tool.strict,
    }
}

fn to_vllm_output(result: DynamoToolParseResult) -> ToolParserOutput {
    // vLLM's ToolParserOutput is event-ordered; the Dynamo result already
    // separates text from calls, so emit the text first, then each call.
    let mut output = ToolParserOutput::default();
    output.push_text(result.normal_text);
    for call in result.calls {
        output.push_call(to_vllm_delta(call));
    }
    output
}

fn to_vllm_delta(delta: DynamoToolCallDelta) -> ToolCallDelta {
    ToolCallDelta {
        tool_index: delta.tool_index,
        name: delta.name,
        arguments: delta.arguments,
    }
}

fn dynamo_error(error: anyhow::Error) -> crate::tool::ToolParserError {
    parsing_failed!("Dynamo parser failed: {}", error)
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::tool::Tool;
    use crate::tool::ToolParserTestExt as _;

    use super::{DynamoDeepSeekV4ToolParser, DynamoGemma4ToolParser, DynamoQwen3CoderToolParser};

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

        assert_eq!(result.normal_text(), "");
        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls()[0].arguments).unwrap(),
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

        assert_eq!(result.normal_text(), "");
        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_datetime"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls()[0].arguments).unwrap(),
            json!({ "timezone": "Asia/Shanghai" })
        );
    }

    fn weather_tool() -> Tool {
        Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }
    }

    #[test]
    fn qwen3_coder_parse_complete_uses_dynamo_stream_parser() {
        let mut parser = DynamoQwen3CoderToolParser::new(&[weather_tool()]).unwrap();
        let result = parser
            .parse_complete(
                "<tool_call> <function=get_weather> \
<parameter=location> NYC </parameter> </function> </tool_call>",
            )
            .unwrap();

        assert_eq!(result.normal_text(), "");
        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        // Value is schema-typed (string) and trimmed, matching the v1 batch parser.
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls()[0].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }

    #[test]
    fn qwen3_coder_preserves_prefix_text_before_block() {
        let mut parser = DynamoQwen3CoderToolParser::new(&[weather_tool()]).unwrap();
        let result = parser
            .parse_complete(
                "I will check the weather. <tool_call> <function=get_weather> \
<parameter=location>NYC</parameter> </function> </tool_call>",
            )
            .unwrap();

        assert_eq!(result.normal_text(), "I will check the weather. ");
        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
    }

    // gemma4 has no native vLLM tool parser (unified-only), so verify the
    // Dynamo parser directly on gemma4's custom `<|tool_call>...` grammar.
    #[test]
    fn gemma4_dynamo_parser_extracts_call() {
        let tools = vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }];
        let mut parser = DynamoGemma4ToolParser::new(&tools).unwrap();
        let result = parser
            .parse_complete("<|tool_call>call:get_weather{location:<|\"|>NYC<|\"|>}<tool_call|>")
            .unwrap();

        assert_eq!(result.calls().len(), 1);
        assert_eq!(result.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&result.calls()[0].arguments).unwrap(),
            json!({ "location": "NYC" })
        );
    }
}
