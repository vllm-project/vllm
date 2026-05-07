use std::future::Future;

use futures::FutureExt as _;
use openai_protocol::common::Tool as OpenAiTool;

use super::{Result, ToolCallDelta, ToolParseResult};
use crate::ToolParser;
use crate::request::ChatTool;

/// Adaptor that exposes the external `tool-parser` through the local
/// [`ToolParser`] interface.
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
    /// Delagating to the external `parse_complete()`.
    ///
    /// We don't rely on the default `push()+finish()` lifecycle, because some
    /// external parsers may not correctly handle the full text passed to
    /// incremental `push()` interface.
    // TODO: instead of working around like this, we should make incremental
    // `push()` robust enough to handle decoded text in arbitrary chunk sizes,
    // as optimizations like speculative decoding or batching may still make the
    // chunk "too long" to be correctly parsed in one `push()` call.
    fn parse_complete(&mut self, output: &str) -> Result<ToolParseResult> {
        let (normal_text, calls) = poll_external(self.inner.parse_complete(output))?;

        // The external `parse_complete()` path does not receive tools and may therefore
        // return calls with invalid names. Filter them here against the request-scoped
        // tool set captured at parser creation time.
        let calls = calls
            .into_iter()
            .filter(|tool_call| {
                self.tools.iter().any(|tool| tool.function.name == tool_call.function.name)
            })
            .enumerate()
            .map(|(tool_index, tool_call)| ToolCallDelta {
                tool_index,
                name: Some(tool_call.function.name),
                arguments: tool_call.function.arguments,
            })
            .collect();

        Ok(ToolParseResult { normal_text, calls })
    }

    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        poll_external(self.inner.parse_incremental(chunk, &self.tools)).map(convert_parse_result)
    }

    fn finish(&mut self) -> Result<ToolParseResult> {
        let calls = self
            .inner
            .get_unstreamed_tool_args()
            .unwrap_or_default()
            .into_iter()
            .map(convert_tool_call_item)
            .collect();

        Ok(ToolParseResult {
            normal_text: String::new(),
            calls,
        })
    }
}

/// Bridge the external async trait into our synchronous local trait.
///
/// This is intentionally a temporary compatibility layer: the current external
/// parser implementations are CPU-only and their async fns do not actually
/// suspend. As long as that dependency behavior stays unchanged,
/// `now_or_never()` is a robust adaptation strategy and we don't have to spawn
/// a thread to `block_on()` the future.
fn poll_external<T>(
    future: impl Future<Output = tool_parser::errors::ParserResult<T>>,
) -> Result<T> {
    future
        .now_or_never()
        .ok_or_else(|| {
            tool_parser::errors::ParserError::ParsingFailed(
                "external tool parser future unexpectedly yielded".to_string(),
            )
        })?
        .map_err(Into::into)
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
        calls: result.calls.into_iter().map(convert_tool_call_item).collect(),
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

        impl ToolParser for $name {
            fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>> {
                Ok(Box::new(Self(ExternalToolParserAdaptor::new(
                    <tool_parser::parsers::$external>::$new_method(),
                    tools,
                ))))
            }

            fn parse_complete(&mut self, output: &str) -> Result<ToolParseResult> {
                self.0.parse_complete(output)
            }

            fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
                self.0.push(chunk)
            }

            fn finish(&mut self) -> Result<ToolParseResult> {
                self.0.finish()
            }
        }
    };
}

// Markup-style tool-call formats.
def_external_tool_parser!(Step3ToolParser, Step3Parser);

// JSON tool-call formats.
def_external_tool_parser!(CohereToolParser, CohereParser);
def_external_tool_parser!(JsonToolParser, JsonParser);
def_external_tool_parser!(Llama3JsonToolParser, LlamaParser);
def_external_tool_parser!(MistralToolParser, MistralParser);
def_external_tool_parser!(Qwen3XmlToolParser, QwenParser);

// Custom envelopes with JSON arguments.
def_external_tool_parser!(DeepSeekV31ToolParser, DeepSeek31Parser);
def_external_tool_parser!(DeepSeekV3ToolParser, DeepSeekParser);

// Special-token or custom-syntax tool-call formats.
def_external_tool_parser!(PythonicToolParser, PythonicParser);
