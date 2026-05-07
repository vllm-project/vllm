use super::{DeepSeekDsmlToolParser, DsmlTokens};
use crate::parser::tool::{Result, ToolParseResult, ToolParser};
use crate::request::ChatTool;

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
    pub(super) fn new(tools: &[ChatTool]) -> Self {
        Self(DeepSeekDsmlToolParser::new(tools, DsmlTokens::V32))
    }
}

impl ToolParser for DeepSeekV32ToolParser {
    /// Create a boxed DeepSeek V3.2 tool parser.
    fn create(tools: &[ChatTool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    /// Preserve DSML special tokens when tool parsing is enabled.
    fn adjust_request(&self, request: &mut crate::request::ChatRequest) -> Result<()> {
        self.0.adjust_request(request)
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
