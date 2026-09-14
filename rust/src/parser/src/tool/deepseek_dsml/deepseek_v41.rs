// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::{DeepSeekDsmlToolParser, DsmlTokens};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};

mod structural_tag;

/// Tool parser for DeepSeek V4.1's spaced DSML tags.
///
/// Arguments are emitted only after a full `invoke` block is parsed.
pub struct DeepSeekV41ToolParser(DeepSeekDsmlToolParser);

impl ToolParser for DeepSeekV41ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self(DeepSeekDsmlToolParser::new(
            tools,
            DsmlTokens::V41,
        ))))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(&structural_tag::DeepSeekV41StructuralTagBuilder)
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
mod tests;
