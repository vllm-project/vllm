// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_tokenizer::{DecodedText, DynTokenizer, Tokenizer};

use super::{DeepSeekDsmlToolParser, DsmlTokens, dsml_marker, dsml_token};
use crate::tool::{Result, StructuralTagBuilder, Tool, ToolParser, ToolParserOutput};
use crate::utils::Marker;

/// Tool parser for DeepSeek V4.1's spaced DSML tags.
///
/// Arguments stream as their parameter values arrive.
pub struct DeepSeekV41ToolParser(DeepSeekDsmlToolParser);

impl DeepSeekV41ToolParser {
    /// The tool-calls opener as this parser matches it.
    pub(crate) fn tool_calls_start(tokenizer: &dyn Tokenizer) -> Marker {
        dsml_marker(
            dsml_token(tokenizer).as_ref(),
            DsmlTokens::V41.tool_calls_start,
        )
    }
}

impl ToolParser for DeepSeekV41ToolParser {
    fn create(tools: &[Tool], tokenizer: DynTokenizer) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self(DeepSeekDsmlToolParser::new(
            tools,
            tokenizer.as_ref(),
            DsmlTokens::V41,
        ))))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn structural_tag_builder(&self) -> Option<&dyn StructuralTagBuilder> {
        Some(xgrammar_structural_tag::Model::DeepSeekV41.builder())
    }

    fn parse_into(&mut self, chunk: DecodedText, output: &mut ToolParserOutput) -> Result<()> {
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
