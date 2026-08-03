// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{GlmXmlToolParser, Separator};
use crate::tool::{Result, Tool, ToolParser, ToolParserOutput};

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
    pub(super) fn new(tools: &[Tool]) -> Self {
        Self(GlmXmlToolParser::new(tools, Separator::Newline))
    }
}

impl ToolParser for Glm45MoeToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
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
