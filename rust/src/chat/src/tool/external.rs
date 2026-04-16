use async_trait::async_trait;
use openai_protocol::common::Tool as OpenAiTool;

use super::{Result, ToolCallDelta, ToolParseResult};
use crate::ToolParser;
use crate::request::ChatTool;

/// Adaptor that exposes the external `tool-parser` through the local [`ToolParser`] interface.
pub(crate) struct ExternalToolParserAdaptor<P> {
    pub(crate) inner: P,
    tools: Vec<OpenAiTool>,
}

impl<P> ExternalToolParserAdaptor<P> {
    pub(crate) fn new(inner: P, tools: &[ChatTool]) -> Self {
        let tools = tools.iter().map(ChatTool::to_openai_tool).collect();
        Self { inner, tools }
    }
}

impl<P> ExternalToolParserAdaptor<P>
where
    P: tool_parser::ToolParser,
{
    async fn parse_complete(&self, output: &str) -> Result<ToolParseResult> {
        self.inner
            .parse_complete(output)
            .await
            .map(|(normal_text, tool_calls)| {
                // The external `parse_complete()` path does not receive tools and may therefore
                // return calls with invalid names. Filter them here against the request-scoped tool
                // set captured at parser creation time.
                let calls = tool_calls
                    .into_iter()
                    .filter(|tool_call| {
                        self.tools
                            .iter()
                            .any(|tool| tool.function.name == tool_call.function.name)
                    })
                    .enumerate()
                    .map(|(tool_index, tool_call)| ToolCallDelta {
                        tool_index,
                        name: Some(tool_call.function.name),
                        arguments: tool_call.function.arguments,
                    })
                    .collect();

                ToolParseResult { normal_text, calls }
            })
            .map_err(Into::into)
    }

    async fn parse_incremental(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.inner
            .parse_incremental(chunk, &self.tools)
            .await
            .map(convert_parse_result)
            .map_err(Into::into)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallDelta>> {
        self.inner
            .get_unstreamed_tool_args()
            .map(|items| items.into_iter().map(convert_tool_call_item).collect())
    }
}

fn convert_tool_call_item(item: tool_parser::types::ToolCallItem) -> ToolCallDelta {
    ToolCallDelta {
        tool_index: item.tool_index,
        name: item.name,
        arguments: item.parameters,
    }
}

fn convert_parse_result(result: tool_parser::types::StreamingParseResult) -> ToolParseResult {
    ToolParseResult {
        normal_text: result.normal_text,
        calls: result
            .calls
            .into_iter()
            .map(convert_tool_call_item)
            .collect(),
    }
}

macro_rules! def_external_tool_parser {
    ($name:ident, $external:ident) => {
        def_external_tool_parser!($name, $external, new);
    };

    ($name:ident, $external:ident, $new_method:ident) => {
        #[doc = concat!(
          "Adaptor exposing the external [`tool_parser::parsers::", stringify!($external), "`] through the local [`ToolParser`] interface."
        )]
        pub struct $name(ExternalToolParserAdaptor<tool_parser::parsers::$external>);

        #[async_trait]
        impl ToolParser for $name {
            fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>> {
                Ok(Box::new(Self(ExternalToolParserAdaptor::new(
                    <tool_parser::parsers::$external>::$new_method(),
                    tools,
                ))))
            }

            async fn parse_complete(&self, output: &str) -> Result<ToolParseResult> {
                self.0.parse_complete(output).await
            }

            async fn parse_incremental(&mut self, chunk: &str) -> Result<ToolParseResult> {
                self.0.parse_incremental(chunk).await
            }

            fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallDelta>> {
                self.0.get_unstreamed_tool_args()
            }
        }
    };
}

def_external_tool_parser!(CohereToolParser, CohereParser);
def_external_tool_parser!(DeepSeekV31ToolParser, DeepSeek31Parser);
def_external_tool_parser!(DeepSeekV3ToolParser, DeepSeekParser);
def_external_tool_parser!(Glm45MoeToolParser, Glm4MoeParser, glm45);
def_external_tool_parser!(Glm47MoeToolParser, Glm4MoeParser, glm47);
def_external_tool_parser!(JsonToolParser, JsonParser);
def_external_tool_parser!(KimiK2ToolParser, KimiK2Parser);
def_external_tool_parser!(Llama3JsonToolParser, LlamaParser);
def_external_tool_parser!(MinimaxM2ToolParser, MinimaxM2Parser);
def_external_tool_parser!(MistralToolParser, MistralParser);
def_external_tool_parser!(PythonicToolParser, PythonicParser);
def_external_tool_parser!(Qwen3CoderToolParser, QwenCoderParser);
def_external_tool_parser!(Qwen3XmlToolParser, QwenParser);
def_external_tool_parser!(Step3ToolParser, Step3Parser);
