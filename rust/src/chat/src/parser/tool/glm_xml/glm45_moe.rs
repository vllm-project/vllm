use super::{GlmXmlToolParser, Separator};
use crate::parser::tool::{Result, ToolParseResult, ToolParser};
use crate::request::ChatTool;

/// Tool parser for GLM-4.5/4.6 MoE XML-style tool calls.
///
/// Example tool call content:
///
/// ```text
/// <tool_call>get_weather
/// <arg_key>city</arg_key>
/// <arg_value>Hangzhou</arg_value>
/// </tool_call>
/// ```
///
/// Arguments are emitted only after a full `tool_call` block is parsed.
pub struct Glm45MoeToolParser(GlmXmlToolParser);

impl Glm45MoeToolParser {
    /// Create a GLM-4.5/4.6 MoE tool parser.
    pub(super) fn new(tools: &[ChatTool]) -> Self {
        Self(GlmXmlToolParser::new(tools, Separator::Newline))
    }
}

impl ToolParser for Glm45MoeToolParser {
    /// Create a boxed GLM-4.5/4.6 MoE tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Push one decoded text chunk through the GLM MoE parser.
    fn push(&mut self, chunk: &str) -> Result<ToolParseResult> {
        self.0.push(chunk)
    }

    /// Flush buffered text and reset parser state.
    fn finish(&mut self) -> Result<ToolParseResult> {
        self.0.finish()
    }
}
